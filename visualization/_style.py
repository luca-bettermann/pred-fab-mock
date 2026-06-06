"""Visual-identity palette, rcParams, and helpers for the ISARC mock plots.

A self-contained local style module (pred-fab's `plotting._style` is a later
addition not present in `pred-fab@isarc-2026`, so it can't be imported here).
Mirrors the same Visual Identity: Zinc / Steel / Emerald spectrums, red+yellow
accents, RdYlGn for performance, thin dot/cross markers, no top/right spines.
"""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as _LSC

# ── Core palette ──────────────────────────────────────────────────────────────
ZINC_50, ZINC_100, ZINC_200, ZINC_300 = "#FAFAFA", "#F4F4F5", "#E4E4E7", "#D4D4D8"
ZINC_400, ZINC_500, ZINC_600, ZINC_700 = "#A1A1AA", "#71717A", "#52525B", "#3F3F46"
ZINC_800, ZINC_900 = "#27272A", "#18181B"
STEEL_100, STEEL_300, STEEL_500, STEEL_700, STEEL_900 = "#D6E4F0", "#8BB0CC", "#4A7FA5", "#2D5F85", "#1A3A5C"
EMERALD_100, EMERALD_300, EMERALD_500, EMERALD_700, EMERALD_900 = "#D1FAE5", "#6EE7B7", "#10B981", "#047857", "#064E3B"
ACCENT_RED = "#DC2626"
ACCENT_YELLOW = "#EAB308"

# ── Semantic assignments ──────────────────────────────────────────────────────
# Phases: baseline = neutral/random (Zinc), exploration = information (Steel),
# inference = optimized/performance (Emerald). Never orange (accent-only rule).
PHASE_COLORS = {
    "baseline": ZINC_400,
    "exploration": STEEL_500,
    "inference": EMERALD_500,
}
PHASE_LABELS = {"baseline": "Baseline", "exploration": "Exploration", "inference": "Inference"}

def _truncate(base_name: str, lo: float, hi: float, name: str) -> _LSC:
    """A softened copy of a matplotlib colormap, restricted to [lo, hi] (pred-fab convention)."""
    base = plt.get_cmap(base_name)
    return _LSC.from_list(name, base(np.linspace(lo, hi, 256)), N=256)


# Restrained spectrums (avoid the saturated extremes, like pred-fab's *_soft).
# Path deviation as a quality signal: low deviation = green, high = red.
DEVIATION_CMAP = _truncate("RdYlGn_r", 0.12, 0.88, "deviation_soft")
# System performance over a landscape: red = poor, green = good.
PERFORMANCE_CMAP = _truncate("RdYlGn", 0.18, 0.85, "performance_soft")

PUBLICATION_DPI = 600

FONT = {"title": 13, "subtitle": 8, "axis": 10, "tick": 9, "annotation": 8, "legend": 8}


def apply_style() -> None:
    """Set matplotlib rcParams to the Visual-Identity defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": FONT["title"], "axes.titlecolor": ZINC_700, "axes.titleweight": "bold",
        "axes.labelsize": FONT["axis"], "axes.labelcolor": ZINC_600,
        "axes.edgecolor": ZINC_300, "axes.linewidth": 0.8,
        "xtick.color": ZINC_500, "ytick.color": ZINC_500,
        "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "legend.fontsize": FONT["legend"],
        "grid.color": ZINC_200, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": PUBLICATION_DPI,
    })


def clean_spines(ax: Any) -> None:
    """Remove top/right spines; tone left/bottom to Zinc-300."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(ZINC_300)
    ax.spines["bottom"].set_color(ZINC_300)


def clean_3d_panes(ax: Any) -> None:
    """Transparent panes, light edges, muted ticks — 3blue1brown-style 3-D."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
        pane.set_edgecolor(ZINC_300)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(colors=ZINC_500, labelsize=FONT["tick"], pad=1)
        axis.label.set_color(ZINC_600)
        axis.line.set_color(ZINC_300)
    ax.grid(False)


def style_colorbar(cbar: Any, label: str = "") -> None:
    """Apply tick / outline styling to a colorbar."""
    cbar.ax.tick_params(colors=ZINC_500, labelsize=FONT["tick"])
    if label:
        cbar.set_label(label, fontsize=FONT["axis"], color=ZINC_600)
    if cbar.outline is not None:
        cbar.outline.set_edgecolor(ZINC_300)
        cbar.outline.set_linewidth(0.6)


def light_grid(ax: Any, axis: str = "both") -> None:
    """Minimal gridlines (Zinc-200, α=0.25)."""
    ax.grid(True, axis=axis, color=ZINC_200, linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)


def save_fig(name: str, dpi: int = PUBLICATION_DPI, tight: bool = True) -> None:
    """Save the current figure to ./plots/<name>.png at publication DPI."""
    import os
    os.makedirs("./plots", exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(f"./plots/{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close()
