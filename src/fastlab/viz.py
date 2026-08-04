"""Shared matplotlib styling so every figure in the lab reads as one system.

Colors come from a validated palette (adjacent-pair CVD dE 9.1, normal-vision
22.9 on the light surface). Two slots sit below 3:1 contrast, so every chart
here ships a legend or direct labels -- identity is never carried by hue alone.

Encoding rules used throughout:
    identity  -> CATEGORICAL, assigned in fixed slot order, never cycled
    magnitude -> SEQUENTIAL_CMAP (one hue, light to dark)
    polarity  -> DIVERGING_CMAP (blue/red poles, neutral gray midpoint)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -- palette -------------------------------------------------------------

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("fastlab_seq", _BLUE_RAMP)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "fastlab_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f1f1e"]
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


def use_style() -> None:
    """Install the lab's matplotlib defaults. Call once per script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.dpi": 130,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "figure.constrained_layout.use": True,
        }
    )


def header(fig, title: str, subtitle: str | None = None) -> None:
    """Figure-level title plus an optional secondary-ink 'so what' line.

    Both are drawn above the constrained-layout box and pulled back in by the
    tight bbox at save time, which keeps them from stealing space from the axes.
    """
    fig.text(0.005, 1.055, title, ha="left", va="bottom",
             fontsize=14, fontweight="semibold", color=INK_PRIMARY)
    if subtitle:
        fig.text(0.005, 1.012, subtitle, ha="left", va="bottom",
                 fontsize=10, color=INK_SECONDARY)


# Zero must read as "nothing here", so it sits at the chart surface rather than
# taking the lightest step of the sequential ramp.
PRESENCE_CMAP = LinearSegmentedColormap.from_list(
    "fastlab_presence", [SURFACE, "#2a78d6"]
)


def annotate(ax, x, y, text: str, color: str = INK_SECONDARY, **kwargs) -> None:
    """Direct label on a mark. Text stays in ink, never the series color."""
    ax.annotate(
        text, xy=(x, y), xytext=kwargs.pop("offset", (6, 6)),
        textcoords="offset points", fontsize=9, color=color, **kwargs,
    )


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    """Write a figure to figures/<name>.png and report the path."""
    directory = out_dir or FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path
