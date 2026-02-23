"""Project single-cell embeddings onto 2D space."""

from pathlib import Path
from typing import Dict, Tuple

import anndata as ad  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scanpy as sc  # type: ignore
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from adipose_atlas.visualization import _set_matplotlib_publication_parameters


def plot_embedding(
    adata: ad.AnnData,
    output_dir: Path,
    method: str,
    color: str,
    color_map: Dict[str, str],
    filename: str,
    figsize: Tuple[float, float],
    dpi: int = 450,
) -> None:
    """Visualizes embeddings via projection on 2D space.

    Args:
        adata: Annotated data matrix
        output_dir: Directory to save the plot
        method: Embedding method ('umap', 'tsne')
        color: Observation column used for coloring
        color_map: Specific mapping to reproduce publication colors
        filename: Name of the output file
        figsize: (width, height)
        dpi: Resolution for saved figure
    """
    _set_matplotlib_publication_parameters()
    sc.set_figure_params(
        frameon=True,
        vector_friendly=True,
        dpi=dpi,
    )

    full_method_key = f"X_{method}"
    if full_method_key not in adata.obsm:
        logger.warning(f"Basis '{full_method_key}' not found. Skipping plot.")
        return

    logger.info(f"Plotting {method} colored by {color} -> {filename}")

    _apply_color_map(adata=adata, color_key=color, color_map=color_map)

    fig, (ax, ax_legend) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [9, 1], "wspace": 0},
    )

    sc.pl.embedding(
        adata,
        ax=ax,
        basis=method,
        color=color,
        show=False,
        return_fig=False,
        title=None,
        size=2.0,
        legend_loc=None,
    )

    _style_axes(ax, method=method)

    _add_legend(
        adata=adata,
        ax_legend=ax_legend,
        color_key=color,
        color_map=color_map,
    )

    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    fig.clear()


def _apply_color_map(
    adata: ad.AnnData,
    color_key: str,
    color_map: Dict[str, str],
) -> None:
    """Use color scheme from publication"""
    if color_key not in adata.obs:
        return

    if not hasattr(adata.obs[color_key], "cat"):
        return

    categories = list(adata.obs[color_key].cat.categories)
    colors = [color_map.get(cat, "#999999") for cat in categories]
    adata.uns[f"{color_key}_colors"] = colors


def _style_axes(
    ax: Axes,
    method: str,
    x_padding: float = 0.025,
    y_padding: float = 0.025,
) -> None:
    """Style axes for embedding plot"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(top=False, right=False)
    ax.set_title("")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min + x_padding * x_range, x_max - x_padding * x_range)
    ax.set_ylim(y_min + y_padding * y_range, y_max - y_padding * y_range)
    ax.autoscale(False)

    method_lower = method.lower()
    if method_lower == "umap":
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    elif method_lower == "tsne":
        ax.set_xlabel("tSNE1")
        ax.set_ylabel("tSNE2")

    ax.xaxis.set_label_coords(1.00, -0.02)
    ax.xaxis.label.set_horizontalalignment("right")
    ax.yaxis.set_label_coords(-0.0525, 0.94)
    ax.yaxis.label.set_verticalalignment("top")


def _add_legend(
    adata: ad.AnnData,
    ax_legend: Axes,
    color_key: str,
    color_map: Dict[str, str],
) -> None:
    """Add legend to dedicated axes."""
    ax_legend.axis("off")

    if not color_key or color_key not in adata.obs:
        return

    categories = list(adata.obs[color_key].cat.categories)
    colors = [color_map.get(cat, "#999999") for cat in categories]

    filtered = [
        (cat, col) for cat, col in zip(categories, colors) if cat != "Unassigned"
    ]

    categories, colors = map(list, zip(*filtered))

    handles = [Patch(facecolor=c, edgecolor=c) for c in colors]
    ax_legend.legend(
        handles,
        categories,
        loc="center left",
        frameon=False,
        handlelength=1.35,
        handleheight=1.65,
        labelspacing=0.65,
    )
