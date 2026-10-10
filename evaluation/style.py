"""How the figures look: colors, the matplotlib style, and saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
"""Categorical colors, in this fixed order (colorblind-checked as adjacent pairs)."""

AUC_CMAP = LinearSegmentedColormap.from_list(
    "auc", [(0.0, "#8f2323"), (0.25, "#e34948"), (0.5, "#f0efec"), (0.75, "#5598e7"), (1.0, "#0d366b")]
).with_extremes(bad=SURFACE)
"""For heatmaps of AUCs: gray at chance (0.5), blue better and red worse than chance."""


def method_style(method: str) -> tuple[str, str]:
    """(color, marker) of a method. The marker differs too, so figures read without color."""
    match method:
        case "fastdetect":
            return SERIES[0], "o"
        case "detectgpt":
            return SERIES[3], "s"
        case _:
            return MUTED, "^"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "figure.titlesize": 11,
    })


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
