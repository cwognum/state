import logging
import torch
from torch import nn
from omegaconf import OmegaConf

from vci.nn.model import StateEmbeddingModel
from vci.train.trainer import get_embeddings
from vci.utils import get_embedding_cfg

log = logging.getLogger(__name__)


class Finetune:
    def __init__(self, cfg=None, learning_rate=1e-4):
        """
        Initialize the Finetune class for fine-tuning the binary decoder of a pre-trained model.

        Parameters:
        -----------
        cfg : OmegaConf
            Configuration object containing model settings
        learning_rate : float
            Learning rate for fine-tuning the binary decoder
        """
        self.model = None
        self.collator = None
        self.protein_embeds = None
        self._vci_conf = cfg
        self.learning_rate = learning_rate
        self.cached_gene_embeddings = {}
        self.device = None

    def load_model(self, checkpoint: str):
        """
        Load a pre-trained SE model from a single checkpoint path and prepare
        it for use. Mirrors the transform/inference loader behavior: extract
        config and embeddings from the checkpoint if present, otherwise fallbacks.
        """
        if self.model:
            raise ValueError("Model already initialized")

        # Resolve configuration: prefer embedded cfg in checkpoint
        cfg_to_use = self._vci_conf
        if cfg_to_use is None:
            try:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "cfg_yaml" in ckpt:
                    cfg_to_use = OmegaConf.create(ckpt["cfg_yaml"])  # type: ignore
                elif isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
                    hp = ckpt.get("hyper_parameters", {}) or {}
                    # Some checkpoints may have a cfg-like structure in hyper_parameters
                    if isinstance(hp, dict) and len(hp) > 0:
                        try:
                            cfg_to_use = OmegaConf.create(hp["cfg"]) if "cfg" in hp else OmegaConf.create(hp)
                        except Exception:
                            cfg_to_use = OmegaConf.create(hp)
            except Exception as e:
                log.warning(f"Could not extract config from checkpoint: {e}")
        if cfg_to_use is None:
            raise ValueError("No config found in checkpoint and no override provided. Provide SE cfg or a full checkpoint.")

        self._vci_conf = cfg_to_use

        # Load model; allow passing cfg to constructor like inference
        self.model = StateEmbeddingModel.load_from_checkpoint(
            checkpoint, dropout=0.0, strict=False, cfg=self._vci_conf
        )
        self.device = self.model.device

        # Try to extract packaged protein embeddings from checkpoint
        packaged_pe = None
        try:
            ckpt2 = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt2, dict) and "protein_embeds_dict" in ckpt2:
                packaged_pe = ckpt2["protein_embeds_dict"]
        except Exception:
            pass

        # Resolve protein embeddings for pe_embedding weights
        all_pe = packaged_pe or get_embeddings(self._vci_conf)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.device)

        # Keep a mapping from gene name -> protein embedding vector
        self.protein_embeds = packaged_pe
        if self.protein_embeds is None:
            # Fallback to configured path
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings, weights_only=False)

        # Freeze SE model and decoder
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.binary_decoder.parameters():
            p.requires_grad = False
        self.model.binary_decoder.eval()

    def _auto_detect_gene_column(self, adata):
        """Auto-detect the gene column with highest overlap with protein embeddings.

        Returns None to indicate var.index, or a string column name in var.
        """
        if self.protein_embeds is None:
            log.warning("No protein embeddings available for auto-detection, using index")
            return None

        protein_genes = set(self.protein_embeds.keys())
        best_column = None
        best_overlap = 0

        # Check index first
        index_genes = set(getattr(adata.var, "index", []))
        overlap = len(protein_genes.intersection(index_genes))
        if overlap > best_overlap:
            best_overlap = overlap
            best_column = None  # None means use index

        # Check all columns in var
        for col in adata.var.columns:
            try:
                col_vals = adata.var[col].dropna().astype(str)
            except Exception:
                continue
            col_genes = set(col_vals)
            overlap = len(protein_genes.intersection(col_genes))
            if overlap > best_overlap:
                best_overlap = overlap
                best_column = col

        return best_column

    def genes_from_adata(self, adata):
        """Return list of gene names from AnnData using auto-detected column/index."""
        col = self._auto_detect_gene_column(adata)
        if col is None:
            return list(map(str, adata.var.index.values))
        return list(adata.var[col].astype(str).values)

    def get_gene_embedding(self, genes):
        """
        Get embeddings for a list of genes, with caching to avoid recomputation.

        Parameters:
        -----------
        genes : list
            List of gene names/identifiers

        Returns:
        --------
        torch.Tensor
            Tensor of gene embeddings
        """
        # Cache key based on genes tuple
        cache_key = tuple(genes)

        # Return cached embeddings if available
        if cache_key in self.cached_gene_embeddings:
            return self.cached_gene_embeddings[cache_key]

        # Compute gene embeddings; fallback to zero vectors for missing genes.
        missing = [g for g in genes if g not in self.protein_embeds]
        if len(missing) > 0:
            try:
                embed_size = next(iter(self.protein_embeds.values())).shape[-1]
            except Exception:
                embed_size = 5120
            # Log once per call to aid debugging
            log.warning(
                f"Finetune.get_gene_embedding: {len(missing)} gene(s) missing from pretrained embeddings; using zeros as placeholders. "
                f"First missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}."
            )

        protein_embeds = [
            self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(embed_size)
            for x in genes
        ]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        gene_embeds = self.model.gene_embedding_layer(protein_embeds)

        # Cache and return
        self.cached_gene_embeddings[cache_key] = gene_embeds
        return gene_embeds

    def get_counts(self, cell_embs, genes, read_depth=None, batch_size=32):
        """
        Generate predictions with the binary decoder with gradients enabled.

        Parameters:
        - cell_embs: A tensor or array of cell embeddings.
        - genes: List of gene names.
        - read_depth: Optional read depth for RDA normalization.
        - batch_size: Batch size for processing.

        Returns:
        A single tensor of shape [N, num_genes] where N is the total number of cells.
        """

        # Convert cell_embs to a tensor on the correct device.
        cell_embs = torch.tensor(cell_embs, dtype=torch.float, device=self.device)

        # Check if RDA is enabled.
        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 4.0

        # Retrieve gene embeddings (cached if available).
        gene_embeds = self.get_gene_embedding(genes)

        # List to collect the output predictions for each batch.
        output_batches = []

        # Loop over cell embeddings in batches.
        for i in range(0, cell_embs.size(0), batch_size):
            # Determine batch indices.
            end_idx = min(i + batch_size, cell_embs.size(0))
            cell_embeds_batch = cell_embs[i:end_idx]

            # Set up task counts if using RDA.
            if use_rda:
                task_counts = torch.full((cell_embeds_batch.shape[0],), read_depth, device=self.device)
            else:
                task_counts = None

            # Resize the batch using the model's method.
            merged_embs = self.model.resize_batch(cell_embeds_batch, gene_embeds, task_counts)

            # Forward pass through the binary decoder.
            logprobs_batch = self.model.binary_decoder(merged_embs)

            # If the output has an extra singleton dimension (e.g., [B, gene_dim, 1]), squeeze it.
            if logprobs_batch.dim() == 3 and logprobs_batch.size(-1) == 1:
                logprobs_batch = logprobs_batch.squeeze(-1)

            output_batches.append(logprobs_batch)

        # Concatenate all batch outputs along the first dimension.
        return torch.cat(output_batches, dim=0)
