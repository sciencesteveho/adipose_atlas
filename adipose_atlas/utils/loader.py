"""Data loading and preprocessing utilities."""

import anndata as ad  # type: ignore
import scanpy as sc  # type: ignore
from loguru import logger

from adipose_atlas.utils.config import InitConfig


class AnnDataLoader:
    """Load and process AnnData objects.

    Attributes:
        config: Specific configuration dataclass. See InitConfig class for
        details.

    Examples:
    >>> # load the config
    >>> from at_single_cell.utils.config import InitConfig
    >>> config = InitConfig(
    ...     raw_h5ad_path="data/raw_data.h5ad",
    ...     subsample_n=1000,
    ...     seed=42,
    ... )
    >>> # Instantiate the ann data loader
    >>> loader = AnnDataLoader(config)
    """

    def __init__(self, config: InitConfig) -> None:
        """Initialize the AnnDataLoader."""
        self.config = config

    def load_single_nucleus(self) -> ad.AnnData:
        """Load the dataset."""
        path = self.config.h5ad_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"h5ad not found at: {path}")

        logger.info(f"Loading AnnData from {path}...")
        adata = sc.read_h5ad(path)

        if self.config.subsample_n is not None:
            adata = self._subsample_copy(adata)

        return adata

    def load_xenium(self) -> ad.AnnData:
        """Load Xenium spatial AnnData."""
        if self.config.xenium_h5ad_path is None:
            raise ValueError("xenium_h5ad_path is None")

        path = self.config.xenium_h5ad_path
        if not path.exists():
            raise FileNotFoundError(f"Xenium h5ad not found at: {path}")

        logger.info(f"Loading Xenium AnnData from {path}...")
        adata = sc.read_h5ad(path)

        return adata

    def _subsample_copy(self, adata: ad.AnnData) -> ad.AnnData:
        """Subsample the AnnData object.

        Returns:
          A subsampled copy of the AnnData object.
        """
        n_total = adata.n_obs
        subsample_n = self.config.subsample_n
        if subsample_n is None:
            return adata

        if n_total > subsample_n:
            logger.info(
                f"Subsampling from {n_total} to {subsample_n} "
                f"cells (seed={self.config.seed})."
            )
            return sc.pp.subsample(
                adata,
                n_obs=subsample_n,
                random_state=self.config.seed,
                copy=True,
            )  # type: ignore

        logger.info(
            f"Requested subsample ({subsample_n}) >= total cells ({n_total}). "
            "Unable to subsample"
        )
        return adata
