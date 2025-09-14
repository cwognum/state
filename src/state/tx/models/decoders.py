import logging
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ...emb.finetune_decoder import Finetune

logger = logging.getLogger(__name__)


class FinetuneVCICountsDecoder(nn.Module):
    def __init__(
        self,
        genes=None,
        adata=None,
        # checkpoint: str = "/large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt",
        # config: str = "/large_storage/ctc/userspace/aadduri/SE-600M/config.yaml",
        checkpoint: str = "/home/aadduri/vci_pretrain/vci_1.4.4/vci_1.4.4_v7.ckpt",
        config: str = "/home/aadduri/vci_pretrain/vci_1.4.4/config.yaml",
        read_depth=4.0,
        latent_dim=1034,  # dimension of pretrained vci model
        hidden_dim=512,  # hidden dimensions of the decoder
        dropout=0.1,
        basal_residual=False,
    ):
        super().__init__()
        # Initialize finetune helper and model from a single checkpoint
        if checkpoint is None:
            raise ValueError(
                "FinetuneVCICountsDecoder requires a VCI/SE checkpoint. Set kwargs.vci_checkpoint or env STATE_VCI_CHECKPOINT."
            )
        self.finetune = Finetune(cfg=OmegaConf.load(config))
        self.finetune.load_model(checkpoint)
        # Resolve genes: prefer explicit list; else infer from anndata if provided
        if genes is None and adata is not None:
            try:
                genes = self.finetune.genes_from_adata(adata)
            except Exception as e:
                raise ValueError(f"Failed to infer genes from AnnData: {e}")
        if genes is None:
            raise ValueError("FinetuneVCICountsDecoder requires 'genes' or 'adata' to derive gene names")
        self.genes = genes
        # Keep read_depth as a learnable parameter so decoded counts can adapt
        self.read_depth = nn.Parameter(torch.tensor(read_depth, dtype=torch.float), requires_grad=True)
        self.basal_residual = basal_residual

        # layers = [
        #     nn.Linear(latent_dim, hidden_dims[0]),
        # ]

        # self.gene_lora = nn.Sequential(*layers)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.genes)),
        )

        self.gene_decoder_proj = nn.Sequential(
            nn.Linear(len(self.genes), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(self.genes)),
        )

        self.binary_decoder = self.finetune.model.binary_decoder
        for param in self.binary_decoder.parameters():
            param.requires_grad = False

        # Validate that all requested genes exist in the pretrained checkpoint's embeddings
        pe = getattr(self.finetune, "protein_embeds", {})
        self.present_mask = [g in pe for g in self.genes]
        self.missing_positions = [i for i, g in enumerate(self.genes) if g not in pe]
        self.missing_genes = [self.genes[i] for i in self.missing_positions]
        total_req = len(self.genes)
        found = total_req - len(self.missing_positions)
        total_pe = len(pe) if hasattr(pe, "__len__") else -1
        miss_pct = (len(self.missing_positions) / total_req) if total_req > 0 else 0.0
        logger.info(
            f"FinetuneVCICountsDecoder gene check: requested={total_req}, found={found}, missing={len(self.missing_positions)} ({miss_pct:.1%}), all_embeddings_size={total_pe}"
        )

        # Create learnable embeddings for missing genes in the post-ESM gene embedding space
        if len(self.missing_positions) > 0:
            # Infer gene embedding output dimension by a dry-run through gene_embedding_layer
            try:
                sample_vec = next(iter(pe.values())).to(self.finetune.model.device)
                if sample_vec.dim() == 1:
                    sample_vec = sample_vec.unsqueeze(0)
                gene_embed_dim = self.finetune.model.gene_embedding_layer(sample_vec).shape[-1]
            except Exception:
                # Conservative fallback
                gene_embed_dim = 1024

            self.missing_table = nn.Embedding(len(self.missing_positions), gene_embed_dim)
            nn.init.normal_(self.missing_table.weight, mean=0.0, std=0.02)
            # For user visibility
            try:
                self.finetune.missing_genes = self.missing_genes
            except Exception:
                pass
        else:
            # Register a dummy buffer so attributes exist
            self.missing_table = None

    def gene_dim(self):
        return len(self.genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, latent_dim].
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        batch_size, seq_len, latent_dim = x.shape
        x = x.view(batch_size * seq_len, latent_dim)

        # Get gene embeddings
        gene_embeds = self.finetune.get_gene_embedding(self.genes)
        # Replace missing gene rows with learnable embeddings
        if self.missing_table is not None and len(self.missing_positions) > 0:
            device = gene_embeds.device
            learned = self.missing_table.weight.to(device)
            idx = torch.tensor(self.missing_positions, device=device, dtype=torch.long)
            gene_embeds = gene_embeds.clone()
            gene_embeds.index_copy_(0, idx, learned)

        # Handle RDA task counts
        use_rda = getattr(self.finetune.model.cfg.model, "rda", False)
        # Define your sub-batch size (tweak this based on your available memory)
        sub_batch_size = 16
        logprob_chunks = []  # to store outputs of each sub-batch

        for i in range(0, x.shape[0], sub_batch_size):
            # Get the sub-batch of latent vectors
            x_sub = x[i : i + sub_batch_size]

            # Create task_counts for the sub-batch if needed
            if use_rda:
                task_counts_sub = torch.ones((x_sub.shape[0],), device=x.device) * self.read_depth
            else:
                task_counts_sub = None

            # Compute merged embeddings for the sub-batch
            # resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None)
            cell_embeds = x_sub[:, :-10]
            ds_emb = x_sub[:, -10:]
            merged_embs_sub = self.finetune.model.resize_batch(
                cell_embeds=cell_embeds, task_embeds=gene_embeds, task_counts=task_counts_sub, ds_emb=ds_emb
            )

            # Run the binary decoder on the sub-batch
            logprobs_sub = self.binary_decoder(merged_embs_sub)

            # Squeeze the singleton dimension if needed
            if logprobs_sub.dim() == 3 and logprobs_sub.size(-1) == 1:
                logprobs_sub = logprobs_sub.squeeze(-1)

            # Collect the results
            logprob_chunks.append(logprobs_sub)

        # Concatenate the sub-batches back together
        logprobs = torch.cat(logprob_chunks, dim=0)

        # Reshape back to [B, S, gene_dim]
        decoded_gene = logprobs.view(batch_size, seq_len, len(self.genes))
        decoded_gene = decoded_gene + self.gene_decoder_proj(decoded_gene)

        # add logic for basal_residual:
        decoded_x = self.latent_decoder(x)
        decoded_x = decoded_x.view(batch_size, seq_len, len(self.genes))

        # Pass through the additional decoder layers
        return torch.nn.functional.relu(decoded_gene + decoded_x)
