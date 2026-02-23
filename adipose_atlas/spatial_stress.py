"""Spatial stress scoring and matrix construction."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
import scipy.sparse as sp  # type: ignore

from adipose_atlas.utils.config import StressHeatmapConfig


@dataclass(frozen=True)
class StressHeatmapMatrices:
    """Dataclass to handle stress heatmap outputs."""

    states: Tuple[str, ...]
    stress_plus1: pd.Series
    stress_display: pd.Series
    quantile_proportions: pd.DataFrame
    niche_scaled: pd.DataFrame
    matched_markers: Tuple[str, ...]


def compute_spatial_stress_heatmaps(
    adata: ad.AnnData,
    *,
    quantile_rows: Tuple[str, ...],
    niche_rows: Tuple[str, ...],
    config: StressHeatmapConfig,
    stress_markers: Sequence[str],
    spatial_xy_key: str,
) -> StressHeatmapMatrices:
    """Stress computation: bining, scoring, attach bin scores to cells, and
    compute matrices.
    """
    keep = ~adata.obs[config.cell_state_key].astype(str).isin(config.exclude_states)
    adata2 = adata[keep].copy()

    bin_adata, bin_ids, bin_codes, valid = _aggregate_cells_into_spatial_bins(
        adata2,
        bin_um=config.bin_um,
        condition_key=config.condition_key,
        spatial_xy_key=spatial_xy_key,
        layer_for_raw=config.layer_for_raw,
    )

    bin_adata, matched = _score_stress(
        bin_adata,
        condition_key=config.condition_key,
        markers=stress_markers,
        score_key=config.score_key,
    )

    _attach_bin_scores_to_cells(
        adata2,
        bin_adata=bin_adata,
        bin_ids=bin_ids,
        bin_codes=bin_codes,
        valid=valid,
        score_key=config.score_key,
    )

    stress_plus1, stress_disp, prop_q, niche_scaled, states = _build_heatmap_matrices(
        adata2,
        quantile_rows=quantile_rows,
        niche_rows=niche_rows,
        config=config,
    )

    return StressHeatmapMatrices(
        states=states,
        stress_plus1=stress_plus1,
        stress_display=stress_disp,
        quantile_proportions=prop_q,
        niche_scaled=niche_scaled,
        matched_markers=matched,
    )


def _normalize_genesymbol(gene: object) -> str:
    """Normalize a gene symbol by stripping whitespace, uppercasing, and
    truncating at delimiters.
    """
    normalized = str(gene).strip().upper()
    for sep in (".", " ", "\t", "|", "/"):
        if sep in normalized:
            normalized = normalized.split(sep, 1)[0]
    return normalized


def _map_markers_to_varnames(
    variable_names: pd.Index, markers: Sequence[str]
) -> Tuple[str, ...]:
    """Map marker gene symbols to their matching variable names, returning only
    those found.
    """
    normalized_map: dict[str, str] = {}
    for variable_name in pd.Index(variable_names.astype(str)):
        key = _normalize_genesymbol(variable_name)
        if key not in normalized_map:
            normalized_map[key] = str(variable_name)

    mapped: list[str] = []
    for marker in markers:
        key = _normalize_genesymbol(marker)
        if key in normalized_map:
            mapped.append(normalized_map[key])
    return tuple(mapped)


def _get_spatial_coords(adata: ad.AnnData, spatial_xy_key: str) -> np.ndarray:
    """Extract spatial coordinates from obsm or obs columns, returning an (n, 2)
    array. Tries obsm first, then falls back to common obs column name pairs.
    """
    if spatial_xy_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_xy_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"Expected obsm['{spatial_xy_key}'] to be (n,2+) but got {coords.shape}."
            )
        return coords[:, :2]

    for x_key, y_key in [("x_location", "y_location"), ("x", "y"), ("X", "Y")]:
        if x_key in adata.obs.columns and y_key in adata.obs.columns:
            return np.column_stack(
                [
                    pd.to_numeric(adata.obs[x_key], errors="coerce").to_numpy(),
                    pd.to_numeric(adata.obs[y_key], errors="coerce").to_numpy(),
                ]
            )

    raise KeyError(
        f"Could not find XY coordinates. Tried obsm['{spatial_xy_key}'] "
        "and obs columns (x_location,y_location)/(x,y)/(X,Y)."
    )


def _resolve_raw_matrix(
    adata: ad.AnnData, layer_for_raw: Optional[str]
) -> sp.csr_matrix:
    """Return available raw count matrix as a CSR matrix, preferring an explicit
    layer over adata.raw then adata.X.
    """
    if layer_for_raw is not None and layer_for_raw in adata.layers:
        X = adata.layers[layer_for_raw]
    elif adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X
    return sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()  # type: ignore


def _build_bin_obs(
    bin_uniques: np.ndarray, condition_key: str, bin_um: float
) -> pd.DataFrame:
    """Decode bin keys back into condition and grid coordinates for obs metadata."""
    bin_condition = [s.split("__", 1)[1] for s in bin_uniques]
    bin_grid_x = [int(s.split("__", 1)[0].split("_")[0]) for s in bin_uniques]
    bin_grid_y = [int(s.split("__", 1)[0].split("_")[1]) for s in bin_uniques]
    return pd.DataFrame(
        {
            condition_key: bin_condition,
            "bx": bin_grid_x,
            "by": bin_grid_y,
            "x_center": np.array(bin_grid_x, dtype=float) * bin_um,
            "y_center": np.array(bin_grid_y, dtype=float) * bin_um,
        },
        index=pd.Index(bin_uniques.astype(str), name="bin_id"),
    )


def _aggregate_cells_into_spatial_bins(
    adata: ad.AnnData,
    *,
    bin_um: float,
    condition_key: str,
    spatial_xy_key: str,
    layer_for_raw: Optional[str],
) -> Tuple[ad.AnnData, pd.Index, np.ndarray, np.ndarray]:
    """Pseudo-bulk cells into fixed-size spatial bins per condition.

    Coordinates are rounded to a grid of bin_um-sized tiles; cells sharing a
    tile and condition are summed into one pseudo-bulk observation via sparse
    matrix multiplication. Binning per condition prevents expression from
    different samples bleeding together spatially.

    Returns:
        bin_adata: AnnData with X set to binned raw counts.
        bin_ids: Index of bin identifiers (aligned with bin_adata.obs_names).
        bin_codes: For each valid cell, integer code mapping to a bin id.
        valid: Boolean mask of cells that had finite coordinates.
    """
    coords = _get_spatial_coords(adata, spatial_xy_key=spatial_xy_key)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    conditions = adata.obs[condition_key].astype(str).to_numpy()

    valid = np.isfinite(x_coords) & np.isfinite(y_coords)
    if int(valid.sum()) == 0:
        raise ValueError("No valid x/y coordinates to bin.")

    expression_csr = _resolve_raw_matrix(adata, layer_for_raw)

    # Round coordinates to bin grid and combine with condition to form unique
    # bin keys
    bin_x = np.round(x_coords[valid] / float(bin_um)).astype(int)
    bin_y = np.round(y_coords[valid] / float(bin_um)).astype(int)
    valid_conditions = conditions[valid]

    bin_keys = pd.Series(
        bin_x.astype(str) + "_" + bin_y.astype(str) + "__" + valid_conditions
    )
    bin_codes, bin_uniques = pd.factorize(bin_keys, sort=False)
    n_bins = int(len(bin_uniques))

    # Sparse aggregation matrix sums expression across all cells sharing a bin
    aggregation_matrix = sp.csr_matrix(
        (
            np.ones(int(valid.sum()), dtype=np.float32),
            (bin_codes, np.arange(int(valid.sum()), dtype=int)),
        ),
        shape=(n_bins, int(valid.sum())),
    )
    binned_expression = (aggregation_matrix @ expression_csr[valid, :]).tocsr()

    bin_obs = _build_bin_obs(bin_uniques, condition_key, bin_um)  # type: ignore
    bin_adata = ad.AnnData(X=binned_expression, obs=bin_obs, var=adata.var.copy())  # type: ignore
    bin_adata.layers["raw"] = binned_expression.copy()
    bin_adata.X = bin_adata.layers["raw"].copy()

    return bin_adata, pd.Index(bin_uniques.astype(str)), bin_codes, valid


def _score_stress(
    bin_adata: ad.AnnData,
    *,
    condition_key: str,
    markers: Sequence[str],
    score_key: str,
) -> Tuple[ad.AnnData, Tuple[str, ...]]:
    """Score genes per condition with scanpy then assign global quartile labels."""
    mapped_markers = _map_markers_to_varnames(bin_adata.var_names, markers)
    if len(mapped_markers) == 0:
        raise ValueError("No stress markers matched to bin_adata.var_names.")

    condition_scores: list[pd.DataFrame] = []
    for condition in pd.unique(bin_adata.obs[condition_key].astype(str)):
        condition_subset = bin_adata[
            bin_adata.obs[condition_key].astype(str) == condition
        ].copy()
        sc.tl.score_genes(
            condition_subset,
            gene_list=list(mapped_markers),
            ctrl_size=50,
            score_name=score_key,
            use_raw=None,
        )
        condition_scores.append(condition_subset.obs[[score_key]])  # type: ignore

    merged_scores = pd.concat(condition_scores, axis=0).reindex(bin_adata.obs_names)
    bin_adata.obs[score_key] = merged_scores[score_key].astype(float)
    bin_adata.obs["Stress_Score_quant_all"] = pd.qcut(
        bin_adata.obs[score_key],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    ).astype(str)

    return bin_adata, mapped_markers


def _attach_bin_scores_to_cells(
    adata: ad.AnnData,
    *,
    bin_adata: ad.AnnData,
    bin_ids: pd.Index,
    bin_codes: np.ndarray,
    valid: np.ndarray,
    score_key: str,
) -> None:
    """Attach bin-level stress score and global quantiles back to each cell."""
    stress_scores = np.full(int(adata.n_obs), np.nan, dtype=float)
    quantile_labels: np.ndarray = np.full(int(adata.n_obs), None, dtype=object)

    cell_bin_ids = bin_ids.to_numpy()[bin_codes]
    stress_map = bin_adata.obs[score_key].to_dict()

    quantile_map = bin_adata.obs["Stress_Score_quant_all"].to_dict()
    stress_scores[valid] = pd.Series(cell_bin_ids).map(stress_map).to_numpy(dtype=float)
    quantile_labels[valid] = (
        pd.Series(cell_bin_ids).map(quantile_map).to_numpy(dtype=object)
    )

    adata.obs["bin_stress_raw"] = stress_scores
    adata.obs["bin_stress_quant_all"] = pd.Categorical(
        quantile_labels, categories=["Q1", "Q2", "Q3", "Q4"], ordered=True
    )


def _get_ordered_present_states(
    adata: ad.AnnData,
    *,
    cell_state_key: str,
    state_order: Optional[Sequence[str]],
    exclude_states: Sequence[str],
) -> Tuple[str, ...]:
    """Return observed cell states in order, excluding specified states."""
    observed_states = set(adata.obs[cell_state_key].astype(str))
    observed_states = {
        state for state in observed_states if state not in set(exclude_states)
    }
    if state_order is None:
        return tuple(sorted(observed_states))

    return tuple(state for state in state_order if state in observed_states)


def _build_heatmap_matrices(
    adata: ad.AnnData,
    quantile_rows: Tuple[str, ...] = ("Q4", "Q3", "Q2", "Q1"),
    niche_rows: Tuple[str, ...] = ("AD", "Stress", "Stem", "Arterial", "Venous"),
    *,
    config: StressHeatmapConfig,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, Tuple[str, ...]]:
    """Compute heatmap matrices: mean stress per state (shifted by +1 before
    min-max normalizing to handle near-zero scores), quantile composition,
    and niche proportions scaled to each niche's maximum for visual
    comparability across states.
    """
    states = _get_ordered_present_states(
        adata,
        cell_state_key=config.cell_state_key,
        state_order=config.state_order,
        exclude_states=config.exclude_states,
    )

    obs_stress = adata.obs[[config.cell_state_key, "bin_stress_raw"]].copy()
    obs_stress[config.cell_state_key] = obs_stress[config.cell_state_key].astype(str)
    obs_stress = obs_stress[obs_stress[config.cell_state_key].isin(states)]

    mean_stress = (
        obs_stress.groupby(config.cell_state_key)["bin_stress_raw"]  # type: ignore
        .mean()
        .reindex(list(states))
    )
    stress_plus1 = mean_stress + 1.0
    stress_normalized = (stress_plus1 - np.nanmin(stress_plus1)) / (
        np.nanmax(stress_plus1) - np.nanmin(stress_plus1) + 1e-12
    )

    obs_quantile = adata.obs[[config.cell_state_key, "bin_stress_quant_all"]].copy()
    obs_quantile[config.cell_state_key] = obs_quantile[config.cell_state_key].astype(
        str
    )
    obs_quantile["quantile"] = obs_quantile["bin_stress_quant_all"].astype(str)
    obs_quantile = obs_quantile[obs_quantile[config.cell_state_key].isin(states)]

    quantile_counts = (
        pd.crosstab(obs_quantile["quantile"], obs_quantile[config.cell_state_key])
        .reindex(index=["Q1", "Q2", "Q3", "Q4"], fill_value=0)
        .reindex(columns=list(states), fill_value=0)
    )
    quantile_proportions = (
        quantile_counts / quantile_counts.sum(axis=0).replace(0, np.nan)
    ).fillna(0.0)
    quantile_proportions = quantile_proportions.reindex(index=list(quantile_rows))

    obs_niche = adata.obs[[config.cell_state_key, config.niche_key]].copy()
    obs_niche[config.cell_state_key] = obs_niche[config.cell_state_key].astype(str)
    obs_niche[config.niche_key] = obs_niche[config.niche_key].astype(str)
    obs_niche = obs_niche[obs_niche[config.cell_state_key].isin(states)]

    niche_counts = (
        pd.crosstab(obs_niche[config.niche_key], obs_niche[config.cell_state_key])
        .reindex(index=list(niche_rows), fill_value=0)
        .reindex(columns=list(states), fill_value=0)
    )
    niche_proportions = (niche_counts.T / niche_counts.sum(axis=1).replace(0, np.nan)).T
    niche_proportions = niche_proportions.fillna(0.0)
    niche_scaled = (
        niche_proportions / niche_proportions.max(axis=0).replace(0, np.nan)
    ).fillna(0.0)

    return stress_plus1, stress_normalized, quantile_proportions, niche_scaled, states
