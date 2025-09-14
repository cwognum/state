import logging

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
        # model_loc="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/step=950000.ckpt",
        # config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/tahoe_config.yaml",
        model_loc="/home/aadduri/vci_pretrain/vci_1.4.2.ckpt",
        config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/large_1e-4_rda_tabular_counts_2048/crossds_config.yaml",
        read_depth=1200,
        latent_dim=1024,  # dimension of pretrained vci model
        hidden_dims=[512, 512, 512],  # hidden dimensions of the decoder
        dropout=0.1,
        basal_residual=False,
    ):
        super().__init__()
        # Initialize finetune helper and model
        self.model_loc = model_loc
        self.config = config
        self.finetune = Finetune(OmegaConf.load(self.config))
        self.finetune.load_model(self.model_loc)
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
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], len(self.genes)),
            nn.ReLU(),
        )

        self.gene_decoder_proj = nn.Sequential(
            nn.Linear(len(self.genes), 128),
            nn.Linear(128, len(self.genes)),
        )

        self.binary_decoder = self.finetune.model.binary_decoder
        for param in self.binary_decoder.parameters():
            param.requires_grad = False

        # Validate that all requested genes exist in the pretrained checkpoint's embeddings
        pe = getattr(self.finetune, "protein_embeds", {})
        present = [g for g in self.genes if g in pe]
        missing = [g for g in self.genes if g not in pe]
        if len(missing) > 0:
            total_req = len(self.genes)
            total_pe = len(pe) if hasattr(pe, "__len__") else -1
            found = total_req - len(missing)
            miss_pct = (len(missing) / total_req) if total_req > 0 else 1.0
            logger.error(
                f"FinetuneVCICountsDecoder gene check: requested={total_req}, found={found}, missing={len(missing)} ({miss_pct:.1%}), all_embeddings_size={total_pe}"
            )
            logger.error(
                f"Examples missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}; examples present: {present[:10]}{' ...' if len(present) > 10 else ''}"
            )
            raise ValueError(
                f"FinetuneVCICountsDecoder: {len(missing)} gene(s) not found in pretrained all_embeddings. "
                f"Requested={total_req}, Found={found}, Missing={len(missing)} ({miss_pct:.1%}). "
                f"First missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}."
            )

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
                # task_counts_sub = torch.full(
                #     (x_sub.shape[0],), self.read_depth, device=x.device
                # )
                task_counts_sub = torch.ones((x_sub.shape[0],), device=x.device) * self.read_depth
            else:
                task_counts_sub = None

            # Compute merged embeddings for the sub-batch
            merged_embs_sub = self.finetune.model.resize_batch(x_sub, gene_embeds, task_counts_sub)

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
        # decoded_gene = torch.nn.functional.relu(decoded_gene)

        # Normalize the sum of decoded_gene to be read depth (learnable)
        # Guard against divide-by-zero by adding small epsilon
        eps = 1e-6
        decoded_gene = decoded_gene / (decoded_gene.sum(dim=2, keepdim=True) + eps) * self.read_depth

        # decoded_gene = self.gene_lora(decoded_gene)
        # TODO: fix this to work with basal counts

        # add logic for basal_residual:
        decoded_x = self.latent_decoder(x)
        decoded_x = decoded_x.view(batch_size, seq_len, len(self.genes))

        # Pass through the additional decoder layers
        return decoded_gene + decoded_x
