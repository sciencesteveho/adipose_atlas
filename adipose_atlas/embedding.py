"""Dimensionality reduction and embedding computation."""

from typing import Literal, Sequence

import anndata as ad  # type: ignore
import harmonypy as hm  # type: ignore
import numpy as np  # type: ignore
import scanpy as sc  # type: ignore
from loguru import logger

from adipose_atlas.utils.config import EmbeddingConfig

EmbeddingMethod = Literal["umap", "tsne"]


class DimensionalityReducer:
    """Dimensionality reduction and embedding computation for single-cell count
    matrices.

    Supports two workflows:
    1. Existing PCA — derive embeddings from an existing pre-called PCA. Call
        `compute_neighbors` then `compute_embedding`.
    2. Full reproduction from adata.raw.X — call `preprocess_from_raw` to
        rebuild Harmony-corrected PCA from raw counts then `compute_neighbors`
        and `compute_embedding`.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initializes the dimensionality reducer.

        Args:
            config: Params from config dataclass for embedding computation.
        """
        self.config = config

    def preprocess_from_raw(
        self,
        adata: ad.AnnData,
        batch_key: str = "biosample_id",
        regress_keys: Sequence[str] = ("mt_percent", "ribo_percent", "total_counts"),
        target_sum: float = 1e4,
        harmony_max_iter: int = 50,
        harmony_max_iter_kmeans: int = 30,
        harmony_tau: int = 5,
        n_pcs: int = 40,
        scale_max_value: float = 10.0,
        seed: int = 42,
    ) -> None:
        """Rebuild Harmony-corrected PCA from raw counts.

        Pipeline:
            raw counts → normalize to 10k → log1p normalization → subset to HVG
            → regress out → scale → PCA → Harmony → neighbors

        Mutates adata in place. After this function:
          - adata.X contains regressed and scaled HVG matrix
          - adata.obsm['X_pca'] contains Harmony-corrected PCs

        Args:
            adata: AnnData with raw integer counts in adata.raw.X.
            batch_key: obs column for Harmony batch correction.
            regress_keys: obs columns to regress out after log1p (on HVGs).
            target_sum: Target total counts per cell.
            n_pcs: Number of principal components.
            harmony_max_iter: Maximum Harmony iterations.
            harmony_max_iter_kmeans: Maximum k-means iterations in Harmony.
            harmony_tau: Harmony overclustering protection parameter.
            seed: Random seed for PCA and Harmony.
            scale_max_value: Max value for sc.pp.scale clipping.

        Raises:
            ValueError: If adata.raw is None or obs keys are missing.
        """
        self._validate_preprocessing(adata, batch_key, regress_keys)

        logger.info("Using raw counts (adata.raw.X).")
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.raw.X
        adata.X = adata.layers["counts"]

        logger.info(f"Normalizing to {target_sum:.0f} counts per cell...")
        sc.pp.normalize_total(adata, target_sum=float(target_sum))

        logger.info("Log1p transforming...")
        sc.pp.log1p(adata)
        adata.raw = adata

        logger.info("Computing highly variable genes...")
        sc.pp.highly_variable_genes(
            adata,
        )
        n_hvg = int(adata.var["highly_variable"].sum())
        logger.info(f"Found {n_hvg} highly variable genes.")

        logger.info(f"Subsetting to {n_hvg} HVGs...")
        adata._inplace_subset_var(adata.var["highly_variable"])

        keys = list(regress_keys)
        logger.info(f"Regressing out {keys} (on HVGs)...")
        sc.pp.regress_out(adata, keys)

        logger.info(f"Scaling (max_value={scale_max_value})...")
        sc.pp.scale(adata, max_value=float(scale_max_value))

        logger.info(f"Computing PCA (n_comps={n_pcs})...")
        sc.tl.pca(
            adata,
            svd_solver="arpack",
            n_comps=int(n_pcs),
            random_state=int(seed),
        )

        logger.info(f"Running Harmony (batch_key={batch_key})...")
        ho = hm.run_harmony(
            adata.obsm["X_pca"],  # type: ignore
            adata.obs,  # type: ignore
            [batch_key],
            epsilon_harmony=-float("Inf"),
            epsilon_cluster=-float("Inf"),
            tau=harmony_tau,
            max_iter_kmeans=int(harmony_max_iter_kmeans),
            max_iter_harmony=int(harmony_max_iter),
            random_state=int(seed),
            device="cpu",  # force cpu to avoid apple silicon GPU issues
        )

        Z = self._coerce_harmony_output(
            ho.Z_corr,
            n_obs=adata.n_obs,
        )

        logger.info(f"Harmony corrected PCA shape: {Z.shape}")
        adata.obsm["X_pca"] = Z
        logger.info("Harmony correction applied to adata.obsm['X_pca'].")

    def compute_neighbors(
        self,
        adata: ad.AnnData,
        n_neighbors: int = 30,
        n_pcs: int = 40,
        metric: str = "euclidean",
        method: str = "umap",
        random_state: int = 0,
    ) -> None:
        """Compute neighborhood graph using PCA space."""
        if self.config.pca_key not in adata.obsm:
            raise ValueError(f"PCA key '{self.config.pca_key}' not found in adata.obsm")

        logger.info(
            "Computing neighbors "
            f"(rep={self.config.pca_key}, n_pcs={n_pcs}, n_neighbors={n_neighbors}, "
            f"metric={metric}, method={method}, random_state={random_state})..."
        )
        sc.pp.neighbors(
            adata=adata,
            use_rep=str(self.config.pca_key),
            n_pcs=int(n_pcs),
            n_neighbors=int(n_neighbors),
            metric=metric,  # type: ignore
            method=method,  # type: ignore
            random_state=int(random_state),
        )

    def compute_embeddings(
        self,
        adata: ad.AnnData,
        method: EmbeddingMethod = "umap",
        force_recompute: bool = False,
    ) -> None:
        """Computes 2D embedding."""
        key = f"X_{method}"
        if not force_recompute and key in adata.obsm:
            logger.info(f"Embedding {key} already exists. Skipping computation.")
            return

        # # leiden clustering
        # if leiden:
        #     sc.tl.leiden(
        #         adata=adata,
        #         resolution=0.25,
        #         random_state=int(self.config.umap_random_state),
        #         n_iterations=2,
        #         flavor="igraph",
        #         key_added="leiden_clusters",
        #     )

        if method == "umap":
            logger.info(
                f"Computing UMAP (min_dist={self.config.umap_min_dist}, spread={self.config.umap_spread})..."
            )
            sc.tl.umap(
                adata=adata,
                min_dist=float(self.config.umap_min_dist),
                spread=float(self.config.umap_spread),
                random_state=int(self.config.umap_random_state),
                init_pos="spectral",
            )
        elif method == "tsne":
            logger.info(
                f"Computing t-SNE (perplexity={self.config.tsne_perplexity})..."
            )
            sc.tl.tsne(
                adata,
                use_rep=str(self.config.pca_key),
                perplexity=float(self.config.tsne_perplexity),
                random_state=int(self.config.tsne_random_state),
            )
        else:
            raise ValueError(f"Unknown embedding method: {method}")

    @staticmethod
    def _validate_preprocessing(
        adata: ad.AnnData,
        batch_key: str,
        regress_keys: Sequence[str],
    ) -> None:
        if adata.raw is None:
            raise ValueError(
                "adata.raw is None. Expected raw integer counts in adata.raw.X."
            )
        required = list(regress_keys) + [batch_key]
        missing = [k for k in required if k not in adata.obs.columns]
        if missing:
            raise ValueError(f"Missing required obs columns: {missing}")

    @staticmethod
    def _coerce_harmony_output(
        Z_corr: np.ndarray,
        n_obs: int,
    ) -> np.ndarray:
        """Coerce Harmony output to a NumPy array with shape (n_obs, n_pcs).

        Args:
            Z_corr: Harmony-corrected embedding from Harmony output (ho.Z_corr).
            n_obs: Expected number of observations (cells).
        """
        Z = Z_corr
        if hasattr(Z, "cpu"):
            Z = Z.detach().cpu().numpy()  # type: ignore
        else:
            Z = np.asarray(Z)

        # harmonypy sometimes returns (pcs, cells)
        if Z.ndim != 2:
            raise ValueError(
                f"Harmony output must be 2D. Got shape={getattr(Z, 'shape', None)}"
            )

        if Z.shape[0] != n_obs:
            if Z.shape[1] == n_obs:
                Z = Z.T
            else:
                raise ValueError(
                    f"Harmony  output has unexpected shape {Z.shape}; expected first or second "
                    f"dimension to equal n_obs={n_obs}."
                )

        return Z
