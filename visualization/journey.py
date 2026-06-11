"""Journey plot: combined score vs experiment count across all phases."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pred_fab.plotting._style import (
    STEEL_500, EMERALD_500, ZINC_300, ZINC_400, ZINC_500, ZINC_600, ZINC_700,
    YELLOW, FONT, clean_spines,
)

from .helpers import save_fig

PHASE_STYLE = {
    "baseline":    (ZINC_400,     "o", "Baseline"),
    "exploration": (EMERALD_500,  "o", "Exploration"),
    "inference":   (YELLOW,       "x", "Inference"),
    "adaptation":  (YELLOW,       "x", "Adaptation"),
}


def plot_journey(
    save_path: str,
    phases: list[str],
    scores: list[float],
    optimum_score: float | None = None,
) -> None:
    """Per-experiment combined scores, running best, and the physics optimum.

    The one-figure pitch: how few experiments the agent needs to close
    the gap to the (normally unknowable) physics optimum.
    """
    n = len(scores)
    xs = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Running best — the journey line
    best, running = float("-inf"), []
    for s in scores:
        best = max(best, s)
        running.append(best)
    ax.step(xs, running, where="post", color=STEEL_500, lw=1.8, zorder=2)

    # Per-experiment scores, styled per phase
    for x, phase, score in zip(xs, phases, scores):
        color, marker, _ = PHASE_STYLE.get(phase, (ZINC_400, "o", phase))
        if marker == "x":
            ax.scatter(x, score, c=color, marker="x", s=70, linewidths=1.2, zorder=4)
        else:
            ax.scatter(x, score, c=color, marker="o", s=22,
                       edgecolors="white", linewidths=0.5, zorder=3)

    if optimum_score is not None:
        ax.axhline(optimum_score, color=ZINC_300, ls="--", lw=0.8, alpha=0.7, zorder=1)
        ax.annotate(f"physics optimum  {optimum_score:.3f}",
                    xy=(1, optimum_score), xytext=(0, 4), textcoords="offset points",
                    fontsize=FONT["annotation"], color=ZINC_500)
        gap = optimum_score - running[-1]
        ax.annotate(f"gap {gap:+.3f}",
                    xy=(n, running[-1]), xytext=(4, -10), textcoords="offset points",
                    fontsize=FONT["annotation"], color=ZINC_600)

    seen = {p for p in phases if p in PHASE_STYLE}
    handles = [
        Line2D([], [], color=STEEL_500, lw=1.8, label="best so far"),
    ] + [
        Line2D([], [], color=PHASE_STYLE[p][0], marker=PHASE_STYLE[p][1], ls="",
               markersize=6, label=PHASE_STYLE[p][2])
        for p in ("baseline", "exploration", "inference", "adaptation") if p in seen
    ]
    ax.legend(handles=handles, loc="lower right",
              fontsize=FONT["legend"], frameon=False)

    ax.set_xlabel("Experiment", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_ylabel("Combined score", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_title("Journey: score vs experiments", fontsize=FONT["title"],
                 color=ZINC_700)
    ax.grid(alpha=0.2, lw=0.8)
    ax.set_xticks([x for x in xs if x == 1 or x % 5 == 0])
    clean_spines(ax)

    save_fig(save_path)
