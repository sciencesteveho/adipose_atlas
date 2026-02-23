"""Visualize spatial single-cell proximity matrix."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

from adipose_atlas.visualization import _set_matplotlib_publication_parameters


def plot_proximity_matrix_heatmap(
    mean_distance_um: pd.DataFrame,
    *,
    output_path: Path,
    figsize: Tuple[float, float] = (10.0, 9.0),
    dpi: int = 450,
    show_every_nth_label: int = 1,
    cbar_ticklabels: Optional[List[str]] = None,
) -> None:
    """Plot proximity matrix heatmap."""
    _set_matplotlib_publication_parameters()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mat = mean_distance_um.to_numpy()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(
        mat,
        aspect="equal",
        interpolation="nearest",
        cmap="GnBu_r",
    )

    ticks = np.arange(mean_distance_um.shape[0])
    ax.set_xticks(ticks[::show_every_nth_label])
    ax.set_yticks(ticks[::show_every_nth_label])

    ax.set_xticklabels(
        list(mean_distance_um.columns)[::show_every_nth_label], rotation=90, ha="center"
    )
    ax.set_yticklabels(list(mean_distance_um.index)[::show_every_nth_label])
    ax.tick_params(axis="both", which="both", length=0, pad=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.25, aspect=8.5)
    if cbar_ticklabels:
        cbar.ax.set_yticklabels(cbar_ticklabels)
    cbar.set_label("Average distance (µm)")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
