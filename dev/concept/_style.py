"""Visual identity for concept dev plots.

Applies the PFAB palette (Zinc, Steel, Emerald + Red/Yellow accents) and
typography rules from `SKILLS - Visual Identity`. Import `apply_style()` at
the top of each concept script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------- Palette ---------

ZINC = {
    50: "#FAFAFA", 100: "#F4F4F5", 200: "#E4E4E7", 300: "#D4D4D8",
    400: "#A1A1AA", 500: "#71717A", 600: "#52525B", 700: "#3F3F46",
    800: "#27272A", 900: "#18181B",
}
STEEL = {100: "#D6E4F0", 300: "#8BB0CC", 500: "#4A7FA5", 700: "#2D5F85", 900: "#1A3A5C"}
EMERALD = {100: "#D1FAE5", 300: "#6EE7B7", 500: "#10B981", 700: "#047857", 900: "#064E3B"}
RED = "#DC2626"
YELLOW = "#EAB308"

# Data series colors in order
SERIES = [STEEL[500], EMERALD[500], ZINC[400], YELLOW, RED]


def apply_style() -> None:
    """Set rcParams for consistent paper-quality figures."""
    mpl.rcParams.update({
        # Figure
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Spines: only left + bottom
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": ZINC[500],
        "axes.linewidth": 0.8,
        # Typography
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titlecolor": ZINC[700],
        "axes.titleweight": "regular",
        "axes.labelsize": 9,
        "axes.labelcolor": ZINC[600],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": ZINC[500],
        "ytick.color": ZINC[500],
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.labelcolor": ZINC[600],
        # Grid
        "axes.grid": True,
        "grid.color": ZINC[200],
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        # Lines
        "lines.linewidth": 1.6,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


# --------- Custom colormaps from palette ---------

def uncertainty_cmap() -> LinearSegmentedColormap:
    """White → Steel, matches visual identity uncertainty convention."""
    return LinearSegmentedColormap.from_list(
        "pfab_uncertainty", ["white", STEEL[100], STEEL[300], STEEL[500], STEEL[900]], N=256,
    )


def evidence_cmap() -> LinearSegmentedColormap:
    """White → Emerald for 'actual evidence' heatmaps."""
    return LinearSegmentedColormap.from_list(
        "pfab_evidence", ["white", EMERALD[100], EMERALD[300], EMERALD[500], EMERALD[900]], N=256,
    )


def density_cmap() -> LinearSegmentedColormap:
    """Zinc spectrum for raw density D(z) (unbounded)."""
    return LinearSegmentedColormap.from_list(
        "pfab_density", ["white", ZINC[100], ZINC[300], ZINC[500], ZINC[800]], N=256,
    )


# --------- Helpers ---------

def strip_spines(ax) -> None:
    """Belt-and-suspenders: remove top/right spines even if style didn't."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plots_dir() -> Path:
    """Output path for concept plots; creates if missing."""
    p = Path(__file__).parent / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, name: str, fmt: str = "png") -> Path:
    """Save to plots_dir() with consistent filename."""
    path = plots_dir() / f"{name}.{fmt}"
    fig.savefig(path, dpi=150 if fmt == "png" else None)
    return path
