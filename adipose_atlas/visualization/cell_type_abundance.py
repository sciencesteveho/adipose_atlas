"""Grouped boxplot of cell-state / cell-type abundance across conditions."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from adipose_atlas.visualization import _set_matplotlib_publication_parameters


def plot_celltype_abundance(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    filename: str,
    sample_key: str,
    category_key: str,
    group_key: str,
    group_order: Sequence[str],
    group_labels: Optional[Sequence[str]] = None,
    category_order: Optional[Sequence[str]] = None,
    subset_mask: Optional[pd.Series] = None,
    exclude_categories: Sequence[str] = ("unassigned", "Unassigned"),
    normalize_to: Optional[str] = None,
    show_points: bool = True,
    point_size: float = 0.175,
    jitter: float = 0.07,
    jitter_seed: int = 0,
    box_total_width: float = 0.78,
    box_linewidth: float = 0.5,
    x_tick_rotation: float = 90.0,
    figsize: Tuple[float, float] = (7.5, 2.8),
    dpi: int = 450,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot grouped boxplot of per-sample % composition.

    Args:
        adata: AnnData with cells in adata.obs.
        output_dir: Directory to save figure.
        filename: Output filename.
        sample_key: obs column for donor/patient/sample id.
        category_key: obs column for x-axis categories.
        group_key: obs column for conditions/groups.
        group_order: Explicit group ordering.
        group_labels: Optional labels to display in legend (same length as
            group_order).
        category_order: Optional explicit order for x-axis categories.
        subset_mask: Optional boolean mask to subset cells before computing
            composition.
        exclude_categories: Categories in `category_key` to drop.
        normalize_to: Optional category in `category_key` to normalize by
            per-sample.
        show_points: If True, overlay per-sample points.
        point_size: Marker size for per-sample points.
        jitter: Horizontal jitter for per-sample points.
        jitter_seed: Seed for deterministic jitter.
        box_total_width: Total width allocated to all grouped boxes at each x.
        box_linewidth: Line width for boxes/whiskers/caps/medians.
        x_tick_rotation: Rotation angle for x tick labels.
        figsize: Figure size.
        dpi: Output dpi.
        ylim: Optional y-axis limits.
    """
    _set_matplotlib_publication_parameters()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = compute_grouped_composition_table(
        adata,
        sample_key=sample_key,
        category_key=category_key,
        group_key=group_key,
        subset_mask=subset_mask,
        exclude_categories=exclude_categories,
        group_order=group_order,
        category_order=category_order,
        normalize_to=normalize_to,
    )

    if category_order is None:
        categories = [c for c in df[category_key].dropna().unique().tolist()]
    else:
        categories = list(category_order)

    groups = list(group_order)
    if len(groups) == 0:
        raise ValueError("group_order is empty.")

    if group_labels is None:
        group_labels = _make_group_labels_with_sample_count(
            df,
            sample_key=sample_key,
            group_key=group_key,
            group_order=groups,
        )
    if len(group_labels) != len(groups):
        raise ValueError("Group_labels must be the same length as group_order.")

    n_groups = len(groups)
    x = np.arange(len(categories), dtype=float)
    box_width = box_total_width / float(n_groups)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * box_width

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors: List[str] = []
    if prop_cycle is not None:
        cycle_colors = [d.get("color", "#777777") for d in prop_cycle]
    if not cycle_colors:
        cycle_colors = ["#777777"]

    group_to_color: Dict[str, str] = {
        g: cycle_colors[i % len(cycle_colors)] for i, g in enumerate(groups)
    }

    legend_handles: List[Patch] = []
    for gi, (group, label) in enumerate(zip(groups, group_labels)):
        positions: List[float] = []
        data: List[np.ndarray] = []

        for ci, cat in enumerate(categories):
            sub = df[(df[group_key] == group) & (df[category_key] == cat)]
            vals = sub["value"].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(float(x[ci] + offsets[gi]))

        if not data:
            legend_handles.append(
                Patch(
                    facecolor=group_to_color[group],
                    edgecolor="black",
                    linewidth=box_linewidth,
                    label=str(label),
                )
            )
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.92,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            manage_ticks=False,
        )

        for box in bp["boxes"]:
            box.set_facecolor(group_to_color[group])
            box.set_alpha(1.0)
            box.set_edgecolor("black")
            box.set_linewidth(box_linewidth)

        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(box_linewidth)

        legend_handles.append(
            Patch(
                facecolor=group_to_color[group],
                edgecolor="black",
                linewidth=box_linewidth,
                label=str(label),
            )
        )

    # Overlay points via deterministic jitter
    if show_points:
        rng = np.random.default_rng(jitter_seed)
        cat_to_x = {c: float(i) for i, c in enumerate(categories)}

        for gi, group in enumerate(groups):
            sub = df[df[group_key] == group].dropna(subset=[category_key])
            if sub.empty:
                continue

            xs = np.array(
                [cat_to_x.get(c, np.nan) for c in sub[category_key].astype(str)],
                dtype=float,
            )
            mask = ~np.isnan(xs)
            xs = (
                xs[mask]
                + offsets[gi]
                + rng.uniform(-jitter, jitter, size=int(mask.sum()))
            )
            ys = sub.loc[mask, "value"].to_numpy(dtype=float)

            ax.scatter(
                xs,
                ys,
                s=point_size,
                linewidths=0.3,
                zorder=3,
                color="0.25",
            )

    # Labels / ticks
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=x_tick_rotation, ha="right")
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0, pad=0.5)
    ax.tick_params(axis="y", which="both", left=False, right=False, length=0, pad=0.5)
    ax.set_ylabel("% Cell type")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        handlelength=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        columnspacing=0.8,
        alignment="right",
    )

    ax.grid(False)
    fig.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    fig.clear()


def _ensure_obs_columns(obs: pd.DataFrame, keys: Sequence[str]) -> None:
    """Ensure obs columns exist"""
    missing = [k for k in keys if k not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")


def _validate_one_group_per_sample(
    obs: pd.DataFrame, sample_key: str, group_key: str
) -> pd.Series:
    """Return a per-sample group label series."""
    mapping = obs[[sample_key, group_key]].drop_duplicates()
    counts = mapping.groupby(sample_key, observed=True)[group_key].nunique()

    bad = counts[counts > 1]
    if not bad.empty:
        examples = bad.index.tolist()[:5]
        raise ValueError(
            f"Each '{sample_key}' must map to exactly one '{group_key}'. "
            f"Found {len(bad)} samples with >1 group; examples: {examples}"
        )
    return mapping.set_index(sample_key)[group_key]


def _make_group_labels_with_sample_count(
    df: pd.DataFrame,
    *,
    sample_key: str,
    group_key: str,
    group_order: Sequence[str],
) -> List[str]:
    """Create legend labels with italiscized sample count."""
    labels: List[str] = []
    for g in group_order:
        n = int(df.loc[df[group_key] == g, sample_key].nunique())
        labels.append(f"{g} ($N$={n})")
    return labels


def compute_grouped_composition_table(
    adata: ad.AnnData,
    *,
    sample_key: str,
    category_key: str,
    group_key: str,
    subset_mask: Optional[pd.Series] = None,
    exclude_categories: Sequence[str] = ("unassigned", "Unassigned"),
    group_order: Optional[Sequence[str]] = None,
    category_order: Optional[Sequence[str]] = None,
    normalize_to: Optional[str] = None,
) -> pd.DataFrame:
    """Compute per-sample composition of category_key and attach group_key.

    Args:
        adata: AnnData
        sample_key: obs column identifying samples/donors/patients (one row per
            cell).
        category_key: obs column whose per-sample composition you want.
        group_key: obs column defining groups/conditions.
            Assumes each sample belongs to exactly one group.
        subset_mask: Optional boolean mask to subset cells.
        exclude_categories: Categories in `category_key` to drop before
            computing.
        group_order: Optional explicit order for groups. If provided, rows are
            ordered.
        category_order: Optional explicit order for categories. If provided,
            missing categories are filled with 0 and ordering is enforced.
        normalize_to: Optional category name in `category_key` to normalize by
            per-sample. If provided, output values become (% category) / (%
            normalize_to).

    Returns:
        DataFrame with columns: [sample_key, group_key, category_key, value,
            n_cells]
    """
    obs = adata.obs.copy()
    _ensure_obs_columns(obs, keys=[sample_key, category_key, group_key])  # type: ignore

    if subset_mask is not None:
        if subset_mask.index is not obs.index:
            subset_mask = subset_mask.reindex(obs.index)
        obs = obs[subset_mask.fillna(False)]

    if exclude_categories:
        obs = obs[~obs[category_key].isin(list(exclude_categories))]

    if obs.empty:  # type: ignore
        raise ValueError("No cells remain after applying subset/exclusions.")

    sample_to_group = _validate_one_group_per_sample(
        obs, sample_key=sample_key, group_key=group_key  # type: ignore
    )

    df_counts = pd.crosstab(obs[sample_key], obs[category_key])

    if category_order is not None:
        df_counts = df_counts.reindex(columns=list(category_order), fill_value=0)

    n_cells = df_counts.sum(axis=1)
    if (n_cells == 0).any():
        zero_samples = n_cells[n_cells == 0].index.tolist()[:5]
        raise ValueError(
            f"Some samples have 0 cells after filtering; examples: {zero_samples}"
        )

    df_pct = df_counts.div(n_cells, axis=0) * 100.0

    if normalize_to is not None:
        if normalize_to not in df_pct.columns:
            raise KeyError(
                f"normalize_to='{normalize_to}' not found among categories in '{category_key}'. "
                f"Available: {list(df_pct.columns)}"
            )
        denom = df_pct[normalize_to].replace(0.0, np.nan)
        df_pct = df_pct.div(denom, axis=0)

    df_long = (
        df_pct.reset_index()
        .melt(id_vars=sample_key, var_name=category_key, value_name="value")
        .merge(
            sample_to_group.rename(group_key),
            left_on=sample_key,
            right_index=True,
            how="left",
        )
    )
    df_long["n_cells"] = df_long[sample_key].map(n_cells)

    # Enforce group order
    if group_order is not None:
        df_long[group_key] = pd.Categorical(
            df_long[group_key], categories=list(group_order), ordered=True
        )
        df_long = df_long.sort_values(
            [group_key, category_key, sample_key], kind="stable"
        )

    # Enforce category order
    if category_order is not None:
        df_long[category_key] = pd.Categorical(
            df_long[category_key], categories=list(category_order), ordered=True
        )
        df_long = df_long.sort_values(
            [category_key, group_key, sample_key], kind="stable"
        )

    return df_long.reset_index(drop=True)
