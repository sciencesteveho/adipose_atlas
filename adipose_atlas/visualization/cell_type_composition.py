"""Barplot for cell-type composition."""

from pathlib import Path
from typing import List

import anndata as ad  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

from adipose_atlas.visualization import (
    GLOBAL_STATE_COLOR_MAP,
    _set_matplotlib_publication_parameters,
)


def plot_cell_type_composition(
    adata: ad.AnnData,
    output_dir: Path,
    filename: str,
    condition_order: List[str],
    condition_labels: List[str],
    sample_key: str = "donor_id",
    cell_type_key: str = "cell_type_level2",
    condition_key: str = "condition",
    dpi: int = 450,
) -> None:
    """Plot cell type composition per donor with mean composition on top.

    Args:
        adata: The AnnData to use for plotting
        output_dir: Where to output the figure
        filename: Name of the figure
        condition_order: Specific order of the conditions for plotting
        condition_labels: Labels to use for plotting the conditions
        sample_key: obs column to identify individuals / samples
        cell_type_key: obs column for grouping the individual cells
        condition_key: obs column for grouping individuals by condition

    Returns:
        None. Saves figure to output_dir.
    """
    _set_matplotlib_publication_parameters()

    plot_params = {
        "kind": "barh",
        "stacked": True,
        "width": 0.9,
        "legend": False,
        "edgecolor": "white",
        "linewidth": 0.05,
    }

    df_counts = pd.crosstab(adata.obs[sample_key], adata.obs[cell_type_key])
    if "Unassigned" in df_counts.columns:
        df_counts = df_counts.drop(columns=["Unassigned"])

    df_props = df_counts.div(df_counts.sum(axis=1), axis=0)

    conditions_to_donors = (
        adata.obs[[sample_key, condition_key]].drop_duplicates().set_index(sample_key)  # type: ignore
    )
    df_props = df_props.join(conditions_to_donors)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(3, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 3, 3, 3], "hspace": 0.1},
    )

    plot_cols = [c for c in df_props.columns if c != condition_key]
    colors = [GLOBAL_STATE_COLOR_MAP.get(c, "#333333") for c in plot_cols]

    ax_mean = axes[0]
    mean_df = df_props.groupby(condition_key)[plot_cols].mean()
    mean_df = mean_df.reindex(condition_order[::-1])
    mean_df.plot(ax=ax_mean, color=colors, **plot_params)  # type: ignore

    ax_mean.set_ylabel("")
    ax_mean.set_yticklabels(condition_labels[::-1])
    ax_mean.tick_params(axis="y", length=2)
    ax_mean.grid(False)

    ax_mean.text(
        1.01,
        0.5,
        "Mean",
        transform=ax_mean.transAxes,
        va="center",
        ha="left",
    )

    for i, (condition, label) in enumerate(zip(condition_order, condition_labels)):
        ax = axes[i + 1]
        subset = df_props[df_props[condition_key] == condition][plot_cols]

        if not subset.empty:
            subset.iloc[::-1].plot(ax=ax, color=colors, **plot_params)  # type: ignore

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.margins(y=0.1)
        ax.grid(False)

        ax.text(
            1.01,
            0.5,
            label,
            transform=ax.transAxes,
            va="center",
            ha="left",
        )

    axes[-1].set_xlim(-0.0075, 1.0075)
    axes[-1].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axes[-1].set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    axes[-1].tick_params(axis="x", which="minor", length=0)
    axes[-1].tick_params(axis="x", which="major", length=2)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", bottom=False, top=False)

    # Add borders for all subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.subplots_adjust(right=0.9)
    fig.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
    fig.clear()
