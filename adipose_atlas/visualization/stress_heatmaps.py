"""Spatial stress stacked heatmap plotting."""

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adipose_atlas.visualization import _set_matplotlib_publication_parameters


def plot_spatial_stress_heatmaps(
    *,
    states: Sequence[str],
    stress_display: pd.Series,
    quantile_proportions: pd.DataFrame,
    niche_scaled: pd.DataFrame,
    stressed_states: Sequence[str] = (),
    output_path: Path,
    title: str = "Spatial zonations",
    quantile_rows: Tuple[str, ...],
    niche_rows: Tuple[str, ...],
) -> None:
    """Plots a three-panel heatmap stack:

    1. Mean binned stress per cell state
    2. Per-state stress proportions in global stress quantiles
    3. Per-state niche composition
    """
    _set_matplotlib_publication_parameters()

    n_states = int(len(states))
    n_top, n_mid, n_bot = (
        1,
        int(quantile_proportions.shape[0]),
        int(niche_scaled.shape[0]),
    )

    cell_in = 0.20
    heat_w = n_states * cell_in
    heat_h = (n_top + n_mid + n_bot) * cell_in

    cbar_clearance = 0.55
    fig = plt.figure(figsize=(heat_w + 0.5, heat_h + cbar_clearance))

    gs = fig.add_gridspec(2, 1, height_ratios=[heat_h, cbar_clearance], hspace=0.0)
    gs_heat = gs[0].subgridspec(3, 1, height_ratios=[n_top, n_mid, n_bot], hspace=0.025)

    ax_top = fig.add_subplot(gs_heat[0])
    ax_mid = fig.add_subplot(gs_heat[1], sharex=ax_top)
    ax_bot = fig.add_subplot(gs_heat[2], sharex=ax_top)

    im_top = ax_top.imshow(
        np.asarray(stress_display.reindex(list(states)).to_numpy(dtype=float))[None, :],
        cmap="inferno",
        vmin=0,
        vmax=1,
        origin="upper",
        aspect="equal",
        interpolation="none",
    )
    ax_top.set_yticks([0])
    ax_top.set_yticklabels([r"Stress $\mathrm{\bar{x}}$"])
    ax_top.tick_params(axis="both", length=0, pad=1.5)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title(title, pad=2)

    im_mid = ax_mid.imshow(
        quantile_proportions.reindex(
            index=list(quantile_rows), columns=list(states)
        ).to_numpy(),
        cmap="GnBu",
        vmin=0,
        vmax=1,
        origin="upper",
        aspect="equal",
        interpolation="none",
    )
    ax_mid.set_yticks(np.arange(len(quantile_rows)))
    ax_mid.set_yticklabels(list(quantile_rows))
    ax_mid.set_ylabel("Stress\nquantile", labelpad=2.5)
    ax_mid.tick_params(axis="both", length=0, pad=1.5)
    ax_mid.tick_params(axis="x", labelbottom=False)

    im_bot = ax_bot.imshow(
        niche_scaled.reindex(index=list(niche_rows), columns=list(states)).to_numpy(),
        cmap="viridis",
        vmin=0,
        vmax=1,
        origin="upper",
        aspect="equal",
        interpolation="none",
    )
    ax_bot.set_yticks(np.arange(len(niche_rows)))
    ax_bot.set_yticklabels(list(niche_rows))
    ax_bot.set_ylabel("Niche", labelpad=2.5)
    ax_bot.tick_params(axis="both", length=0, pad=1.5)

    ax_bot.set_xticks(np.arange(n_states))
    ax_bot.set_xticklabels(list(states), rotation=90, ha="center", va="top")
    stressed = set(str(x) for x in stressed_states)
    for lbl in ax_bot.get_xticklabels():
        if lbl.get_text() in stressed:
            lbl.set_fontweight("bold")

    for ax in (ax_top, ax_mid, ax_bot):
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.canvas.draw()

    heat_bottom = ax_bot.get_position().y0
    cbar_y = heat_bottom - 0.14
    cbar_h_frac = 0.018
    cbar_w = 0.21
    gap = 0.03

    total_w = 3 * cbar_w + 2 * gap
    x_start = 0.5125 - total_w / 2

    cax1 = fig.add_axes([x_start, cbar_y, cbar_w, cbar_h_frac])  # type: ignore
    cax2 = fig.add_axes([x_start + cbar_w + gap, cbar_y, cbar_w, cbar_h_frac])  # type: ignore
    cax3 = fig.add_axes([x_start + 2 * (cbar_w + gap), cbar_y, cbar_w, cbar_h_frac])  # type: ignore

    cb1 = fig.colorbar(im_top, cax=cax1, orientation="horizontal")
    cb1.set_ticks([0.0, 1.0])
    cb1.set_ticklabels(["0", "1"])
    cb1.set_label("Mean bin stress\nin (stress score + 1)", labelpad=2)
    cax1.xaxis.set_label_position("bottom")
    cax1.tick_params(axis="x", labelsize=4, length=0, pad=1)

    cb2 = fig.colorbar(im_mid, cax=cax2, orientation="horizontal")
    cb2.set_ticks([0.0, 1.0])
    cb2.set_ticklabels(["0", "1"])
    cb2.set_label("Percentage cell state\nin quantile", labelpad=2)
    cax2.xaxis.set_label_position("bottom")
    cax2.tick_params(axis="x", labelsize=4, length=0, pad=1)

    cb3 = fig.colorbar(im_bot, cax=cax3, orientation="horizontal")
    cb3.set_ticks([0.0, 1.0])
    cb3.set_ticklabels(["0", "1"])
    cb3.set_label("Proportion cell state\nin niche (scaled)", labelpad=2)
    cax3.xaxis.set_label_position("bottom")
    cax3.tick_params(axis="x", labelsize=4, length=0, pad=1)

    for cax in (cax1, cax2, cax3):
        for spine in cax.spines.values():
            spine.set_visible(False)

    fig.savefig(output_path, bbox_inches="tight", dpi=450)
    plt.close(fig)
