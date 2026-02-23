"""Analysis pipelines for AT single-cell reproduction."""

# Force deterministic environment: only needed as part of this repoduction repo.
# Any future attempts at re-utilizing this code should re-evaluate the computing
# environment.
import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import dataclasses
from typing import List

import scanpy as sc  # type: ignore
from loguru import logger
from scanpy import AnnData  # type: ignore

from adipose_atlas.embedding import DimensionalityReducer
from adipose_atlas.spatial_proximity import CellStateProximityMatrix
from adipose_atlas.spatial_stress import compute_spatial_stress_heatmaps
from adipose_atlas.utils.config import (
    GlobalEmbeddingConfig,
    MyeloidLineageConfig,
    SpatialAnalysisConfig,
    _write_resolved_yaml,
)
from adipose_atlas.utils.loader import AnnDataLoader
from adipose_atlas.utils.seed import _set_global_seed
from adipose_atlas.visualization import GLOBAL_STATE_COLOR_MAP, MYELOID_COLOR_MAP
from adipose_atlas.visualization.cell_type_abundance import plot_celltype_abundance
from adipose_atlas.visualization.cell_type_composition import plot_cell_type_composition
from adipose_atlas.visualization.embedding_projection import plot_embedding
from adipose_atlas.visualization.marker_genes_dotplot import marker_genes_dotplot
from adipose_atlas.visualization.spatial_proximity_matrix import (
    plot_proximity_matrix_heatmap,
)
from adipose_atlas.visualization.stress_heatmaps import plot_spatial_stress_heatmaps

sc.settings.n_jobs = 1


def _ensure_obs_keys(adata: AnnData, keys: List[str]) -> None:
    """Ensure obs has required keys."""
    missing = [k for k in keys if k not in adata.obs.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required obs keys: {missing}")


def _ensure_obsm_keys(adata: AnnData, keys: List[str]) -> None:
    """Ensure obsm has required keys."""
    missing = [k for k in keys if k not in adata.obsm]
    if len(missing) > 0:
        raise ValueError(f"Missing required obsm keys: {missing}")


def _prepare_analysis_dir(
    config: GlobalEmbeddingConfig | MyeloidLineageConfig | SpatialAnalysisConfig,
) -> None:
    """Prep analysis by ensuring directory exists and using configuration
    seed.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _set_global_seed(config.loader.seed)


def run_global_atlas_embedding(config: GlobalEmbeddingConfig) -> None:
    """Reproduce global atlas embeddings and cohort cell-type composition."""
    _prepare_analysis_dir(config)
    _write_resolved_yaml(
        payload=dataclasses.asdict(config),
        output_path=config.output_dir / "analysis_config.yaml",
    )
    loader = AnnDataLoader(config.loader)
    adata = loader.load_single_nucleus()

    _ensure_obs_keys(
        adata=adata,
        keys=[
            config.cell_type_key,
        ],
    )
    _ensure_obsm_keys(
        adata=adata,
        keys=[
            "X_umap",
            config.embedding.pca_key,
        ],
    )

    dimensionality_reducer = DimensionalityReducer(config.embedding)
    dimensionality_reducer.preprocess_from_raw(
        adata,
        batch_key="biosample_id",
        seed=config.embedding.umap_random_state,
    )

    dimensionality_reducer.compute_neighbors(
        adata,
        n_neighbors=config.embedding.n_neighbors,
        n_pcs=40,
        metric="euclidean",
        method="umap",
        random_state=config.embedding.umap_random_state,
    )

    dimensionality_reducer.compute_embeddings(
        adata,
        method="umap",
        force_recompute=True,
    )
    dimensionality_reducer.compute_embeddings(
        adata,
        method="tsne",
        force_recompute=True,
    )

    plot_embedding(
        adata,
        output_dir=config.output_dir,
        method="umap",
        color=config.cell_type_key,
        filename="global_atlas_umap_recomputed_from_raw.png",
        figsize=(8, 6.5),
        color_map=GLOBAL_STATE_COLOR_MAP,
    )
    plot_embedding(
        adata,
        output_dir=config.output_dir,
        method="tsne",
        color=config.cell_type_key,
        filename="global_atlas_tsne_recomputed_from_raw.png",
        figsize=(8, 6.5),
        color_map=GLOBAL_STATE_COLOR_MAP,
    )

    plot_cell_type_composition(
        adata=adata,
        output_dir=config.output_dir,
        filename="global_atlas_cell_type_composition.png",
        sample_key="donor_id",
        condition_order=["Lean", "Obese", "Weightloss"],
        condition_labels=["LN", "OB", "WL"],
    )


def run_myeloid_lineage_analysis(config: MyeloidLineageConfig) -> None:
    """Subset myeloid lineage, reproduce embeddings, and generate dotplot and
    abundance figures.
    """
    _prepare_analysis_dir(config)
    _write_resolved_yaml(
        payload=dataclasses.asdict(config),
        output_path=config.output_dir / "analysis_config.yaml",
    )
    loader = AnnDataLoader(config.loader)
    adata = loader.load_single_nucleus()

    _ensure_obs_keys(
        adata=adata,
        keys=[
            config.cell_type_key,
            config.cell_state_key,
        ],
    )
    _ensure_obsm_keys(adata=adata, keys=[config.embedding.pca_key])

    logger.info(
        f"Subsetting myeloid lineage: {config.cell_type_key} == {config.myeloid_label}"
    )
    adata = adata[adata.obs[config.cell_type_key] == config.myeloid_label].copy()
    if adata.n_obs == 0:
        raise ValueError("Myeloid subset is empty; check cell_type_key/myeloid_label.")

    logger.info("Re-integrating myeloid subset.")
    dimensionality_reducer = DimensionalityReducer(config.embedding)

    logger.info("Rebuilding PCs on subset.")
    dimensionality_reducer.preprocess_from_raw(
        adata,
        batch_key="biosample_id",
        seed=config.loader.seed,
    )
    dimensionality_reducer.compute_neighbors(
        adata,
        n_neighbors=config.embedding.n_neighbors,
        n_pcs=40,
        metric="euclidean",
        method="umap",
        random_state=config.loader.seed,
    )
    dimensionality_reducer.compute_embeddings(
        adata,
        method="umap",
        force_recompute=True,
    )

    plot_embedding(
        adata,
        output_dir=config.output_dir,
        method="umap",
        color=config.cell_state_key,
        filename="myeloid_umap_recomputed.png",
        figsize=(5.2, 5),
        color_map=MYELOID_COLOR_MAP,
    )

    marker_genes_dotplot(
        adata=adata,
        genes=list(config.dotplot_genes),
        groupby=config.cell_state_key,
        filename="myeloid_gene_marker_dotplot.png",
        figsize=(8, 1.35),
        standard_scale="var",
        output_dir=config.output_dir,
        exclude_labels=("unassigned",),
    )

    plot_celltype_abundance(
        adata=adata,
        output_dir=config.output_dir,
        filename="myeloid_state_abundance_by_condition.png",
        sample_key="biosample_id",
        category_key=config.cell_state_key,
        group_key="condition",
        group_order=["Lean", "Obese", "Weightloss"],
        category_order=[
            "MYE1",
            "MYE2",
            "MYE3",
            "MYE4",
            "MYE5",
            "MYE6",
            "MYE7",
            "MYE8",
            "MYE9",
            "MYE10",
            "B-cells",
        ],
        exclude_categories=("unassigned", "Unassigned"),
        show_points=True,
        figsize=(2.15, 1.85),
        dpi=450,
        ylim=(-5, 105),
    )


def run_spatial_analysis(config: SpatialAnalysisConfig) -> None:
    """Run spatial analyses: proximity matrix + stress stacked heatmaps."""
    _prepare_analysis_dir(config)
    _write_resolved_yaml(
        payload=dataclasses.asdict(config),
        output_path=config.output_dir / "analysis_config.yaml",
    )

    loader = AnnDataLoader(config.loader)
    xenium = loader.load_xenium()

    logger.info("Computing cell state proximity matrix.")
    proximity_matrix = CellStateProximityMatrix(
        adata=xenium,
        spatial=config.spatial,
        cell_state_key=config.proximity.cell_state_key,
        radius_um=config.proximity.radius_um,
        sample_key=config.proximity.sample_key,
        exclude_states=config.proximity.exclude_states,
        state_order=config.proximity.state_order,
    )

    logger.info("Plotting cell state proximity matrix.")
    plot_proximity_matrix_heatmap(
        proximity_matrix.matrix.mean_distance_um,
        output_path=config.output_dir / "cell_state_proximity_matrix.png",
        figsize=(3.5, 3.5),
        cbar_ticklabels=["0", "50", "100", "150", "200", "250", "300"],
    )

    quantile_rows = ("Q4", "Q3", "Q2", "Q1")
    niche_rows = ("AD", "Stress", "Stem", "Arterial", "Venous")

    logger.info("Computing spatial stress heatmaps.")
    stress_matrices = compute_spatial_stress_heatmaps(
        xenium,
        config=config.stress,
        stress_markers=list(config.stress_markers),
        spatial_xy_key=config.spatial.spatial_xy_key,
        quantile_rows=quantile_rows,
        niche_rows=niche_rows,
    )

    plot_spatial_stress_heatmaps(
        states=stress_matrices.states,
        stress_display=stress_matrices.stress_display,
        quantile_proportions=stress_matrices.quantile_proportions,
        niche_scaled=stress_matrices.niche_scaled,
        stressed_states=config.stress.stressed_states,
        output_path=config.output_dir / "spatial_stress_stacked_heatmaps.png",
        quantile_rows=quantile_rows,
        niche_rows=niche_rows,
    )
