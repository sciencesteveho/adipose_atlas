"""Dotplot visualization for marker genes."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, cast

import anndata as ad  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scanpy as sc  # type: ignore
from loguru import logger
from matplotlib.figure import Figure  # type: ignore

from adipose_atlas.visualization import _set_matplotlib_publication_parameters


def _filter_groups(
    adata: ad.AnnData,
    groupby: str,
    exclude_labels: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    """Return a subset of adata with labels removed. Matching is
    case-insensitive and unused categories are dropped so they don't appear as
    empty rows in the plot.
    """
    if not exclude_labels:
        return adata

    lower_exclude = {s.lower() for s in exclude_labels}
    labels = adata.obs[groupby].astype(str)
    mask = ~labels.str.lower().isin(lower_exclude)
    n_filtered = int((~mask).sum())

    if n_filtered > 0:
        logger.info(
            "Dotplot: filtered {n} cells whose {col} ∈ {labels}",
            n=n_filtered,
            col=groupby,
            labels=exclude_labels,
        )

    out = adata[mask].copy()
    if hasattr(out.obs[groupby], "cat"):
        out.obs[groupby] = out.obs[groupby].cat.remove_unused_categories()
    return out


def marker_genes_dotplot(
    adata: ad.AnnData,
    output_dir: Path,
    genes: Sequence[str],
    groupby: str,
    filename: str,
    *,
    dpi: int = 450,
    figsize: Tuple[float, float] = (10.0, 4.0),
    largest_dot: float = 20.0,
    size_exponent: float = 1.0,
    dot_edge_color: str = "0.6",
    dot_edge_lw: float = 0.2,
    legend_width: float = 0.675,
    x_tick_pad: float = 0.5,
    y_tick_pad: float = 0.5,
    shift_legend_y: float = 0.14,
    shift_legend_x: float = -0.018,
    standard_scale: Optional[str] = "var",
    use_raw: Optional[bool] = None,
    exclude_labels: Optional[Sequence[str]] = None,
) -> None:
    """Generate a Scanpy dotplot.

    Args:
        adata: Annotated data matrix.
        output_dir: Where to write the figure.
        genes: Specific genes to plot
        groupby: Column in adata.obs that defines the row grouping
        filename: Output file name
        figsize: width, height.
        standard_scale: var, group, or none.
        use_raw: use adata.raw if present.
        largest_dot: Maximum dot area
        size_exponent: Exponent for dot-size scaling
        dot_edge_color: Outline color for dots
        dot_edge_lw: Outline width for dots
        legend_width: Legend size
        x_tick_pad: Padding between x-axis ticks and labels. Positive shifts
            legend up, negative down.
        y_tick_pad: Padding (points) between y-axis ticks and label Positive
            shifts legend right, negative left.
        exclude_labels: Group labels to drop before plotting (case-insensitive)
    """
    _set_matplotlib_publication_parameters()

    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not found in adata.obs")

    # genes_present = [g for g in genes if g in adata.var_names]
    # genes_missing = [g for g in genes if g not in adata.var_names]
    # if genes_missing:
    #     logger.warning("Dotplot: missing genes, skipping: {g}", g=genes_missing)
    # if not genes_present:
    #     raise ValueError("None of the requested genes are in adata.var_names")

    # adata_plot = _filter_groups(adata, groupby, exclude_labels)

    adata_plot = _filter_groups(adata, groupby, exclude_labels)

    if use_raw is None:
        use_raw = adata_plot.raw is not None

    if use_raw:
        if adata_plot.raw is None:
            raise ValueError("use_raw=True but adata.raw is None")
        var_index = adata_plot.raw.var_names
    else:
        var_index = adata_plot.var_names

    genes_present = [g for g in genes if g in var_index]
    genes_missing = [g for g in genes if g not in var_index]
    if genes_missing:
        logger.warning("Dotplot: missing genes, skipping: {g}", g=genes_missing)
    if not genes_present:
        raise ValueError(
            "None of the requested genes are in the selected var_names (raw vs X)."
        )

    if use_raw is None:
        use_raw = adata_plot.raw is not None

    dp = sc.pl.dotplot(
        adata_plot,
        var_names=genes_present,
        groupby=groupby,
        standard_scale=standard_scale,  # type: ignore
        figsize=figsize,
        use_raw=use_raw,
        show=False,
        return_fig=True,
        dot_max=1,
    )

    dp = dp.style(  # type: ignore
        largest_dot=largest_dot,
        size_exponent=size_exponent,
        dot_edge_color=dot_edge_color,
        dot_edge_lw=dot_edge_lw,
    )
    dp = dp.legend(width=legend_width)
    dp.make_figure()

    fig = cast(Figure, dp.fig)
    axes = dp.get_axes()
    main_ax = axes.get("mainplot_ax")

    # Ensure font size consistency
    for ax in fig.axes:
        for text in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            text.set_fontsize(5)

    if main_ax is not None:
        main_ax.set_xlabel("")
        main_ax.set_ylabel("")
        main_ax.tick_params(axis="x", which="major", pad=x_tick_pad)
        main_ax.tick_params(axis="y", which="major", pad=y_tick_pad)

    size_leg_ax = axes.get("size_legend_ax")
    if size_leg_ax is not None:
        size_leg_ax.set_title(size_leg_ax.get_title(), pad=-8)
        size_leg_ax.tick_params(pad=0)
        pos = size_leg_ax.get_position()
        size_leg_ax.set_position(
            [pos.x0 + shift_legend_x, pos.y0 + shift_legend_y, pos.width, pos.height]  # type: ignore
        )

    color_leg_ax = axes.get("color_legend_ax")
    if color_leg_ax is not None:
        color_leg_ax.set_title(color_leg_ax.get_title(), pad=4)
        color_leg_ax.tick_params(pad=0)
        pos = color_leg_ax.get_position()
        color_leg_ax.set_position(
            [pos.x0 + shift_legend_x, pos.y0 + shift_legend_y, pos.width, pos.height]  # type: ignore
        )

    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote dotplot: {p}", p=output_dir / filename)
