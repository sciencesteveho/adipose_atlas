"""Visualization module init."""

from typing import Dict

import matplotlib.pyplot as plt

GLOBAL_STATE_COLOR_MAP: Dict[str, str] = {
    "Adipocytes": "#e69f00",
    "B-cells": "#56b4e9",
    "Endothelial": "#009e73",
    "ASC": "#f0e442",
    "APC": "#0072b2",
    "Lymphatic": "#d55d00",
    "Macrophages": "#cc79a7",
    "Mono_DC": "#666666",
    "Mast": "#ad7700",
    "Mural": "#1c91d4",
    "NK_cells": "#007756",
    "T-cells_CD4+": "#d5c711",
    "T-cells_CD8+": "#005685",
    "ILC_Kit+": "#a04700",
}

MYELOID_COLOR_MAP: Dict[str, str] = {
    "MYE1": "#6367a9",
    "MYE2": "#cc0744",
    "MYE3": "#d157a0",
    "MYE4": "#00a6aa",
    "MYE5": "#ff4a46",
    "MYE6": "#8fb0ff",
    "MYE7": "#a30059",
    "B-cells": "#008941",
    "MYE8": "#6a3a4c",
    "MYE9": "#7900d7",
    "MYE10": "#b903aa",
}


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "figure.titlesize": 5,
            "figure.dpi": 450,
            "font.sans-serif": ["Arial", "Nimbus Sans"],
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.minor.width": 0.25,
        }
    )
