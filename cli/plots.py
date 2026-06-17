"""Showcase plots for the ADVEI 2026 mock.

All plots go through pred-fab's schema-agnostic plotting helpers (the acquisition
topology slices the *real* pipeline via ``compute_acquisition_grids`` — same path
the optimizer uses). Each is saved under ``plots/`` and rendered inline in the
terminal via the iTerm2 inline-image protocol (no-op + path on other terminals).
"""
from __future__ import annotations

import base64
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

from pred_fab.plotting import AxisSpec, radar_chart, subplot_topology, save_fig, apply_style
from pred_fab.plotting._style import (
    ACCENT_YELLOW, ACCENT_RED, STEEL_500, EMERALD_500, ZINC_300, ZINC_400,
    ZINC_500, ZINC_600, ZINC_700, FONT, clean_spines,
)

from models.schema import (
    ParamCode, AttributeCode, PARAM_BOUNDS, N_NODES,
    display_name, derive_n_layers,
)

PLOTS_DIR = "plots"

_BOUNDS = {c: (lo, hi) for c, lo, hi in PARAM_BOUNDS}
# Showcase axis pair — the extrusion plane (calibration × speed) is the most
# active trade-off surface (material deposition vs. extrusion stability vs. energy).
_X, _Y = ParamCode.CALIBRATION_FACTOR, ParamCode.PRINT_SPEED
_ATTRS = [
    AttributeCode.STRUCTURAL_INTEGRITY, AttributeCode.MATERIAL_DEPOSITION,
    AttributeCode.EXTRUSION_STABILITY, AttributeCode.ENERGY_FOOTPRINT,
    AttributeCode.FABRICATION_TIME,
]


def ensure_plots_dir() -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return PLOTS_DIR


def _smooth(grid, passes: int = 6):
    """Light separable 3-tap blur — cosmetic only.

    The passive CCF reference points sit on a regular lattice, so pred-fab's
    evidence KDE shows a visible bump at each one (a checkerboard). The
    optimizer uses the true field; this only cleans the *display* of the
    evidence / acquisition surfaces.
    """
    import numpy as np

    g = np.asarray(grid, dtype=float)
    for _ in range(passes):
        out = g.copy()
        out[1:-1, :] = 0.25 * g[:-2, :] + 0.5 * g[1:-1, :] + 0.25 * g[2:, :]
        g = out
        out = g.copy()
        out[:, 1:-1] = 0.25 * g[:, :-2] + 0.5 * g[:, 1:-1] + 0.25 * g[:, 2:]
        g = out
    return g


def show_inline(path: str) -> None:
    """Render an image inline (iTerm2 protocol); always print the path as fallback."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        sys.stdout.write(f"\033]1337;File=inline=1;size={len(data)}:{b64}\a\n")
        sys.stdout.flush()
    except Exception:
        pass
    print(f"  → {path}")


def _fixed_mid() -> dict[str, Any]:
    """The non-swept params held at their mid-range, plus derived dimensions."""
    fixed = {c: (lo + hi) / 2 for c, lo, hi in PARAM_BOUNDS}
    fixed.pop(_X)
    fixed.pop(_Y)
    fixed["n_nodes"] = N_NODES
    fixed["n_layers"] = derive_n_layers(fixed[ParamCode.LAYER_HEIGHT])
    return fixed


def acquisition_topology(agent: Any, dataset: Any, proposal: dict[str, Any] | None,
                         kappa: float, tag: str) -> str:
    """3-panel performance | evidence | combined acquisition on the extrusion plane.

    Composes pred-fab's ``subplot_topology`` directly instead of ``plot_acquisition``:
    in advei-2026 the latter's combined panel names an unregistered ``"mixed"`` cmap
    (the registry key is ``"acquisition"``), so we render the panels here with the
    correct semantic cmaps.
    """
    import matplotlib.pyplot as plt

    ensure_plots_dir()
    cal = agent.calibration_system
    fixed = _fixed_mid()
    xs, ys, ev, perf, acq = cal.compute_acquisition_grids(
        _X, _Y, _BOUNDS[_X], _BOUNDS[_Y], fixed_params=fixed, kappa=kappa, resolution=40,
    )
    codes = sorted(dataset.get_experiment_codes())
    points = [dataset.get_experiment(c).parameters.get_values_dict() for c in codes]
    x_axis = AxisSpec(_X, display_name(_X), bounds=_BOUNDS[_X])
    y_axis = AxisSpec(_Y, display_name(_Y), unit="m/s", bounds=_BOUNDS[_Y])

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Evidence/acquisition are KDE surfaces over lattice points → smooth + drop
    # the iso-line overlay so they read clean. Performance is already smooth and
    # keeps its contours to mark the optimum.
    for ax, grid, label, cmap_name, overlay in [
        (axes[0], perf, "Performance", "performance", True),
        (axes[1], _smooth(ev), "Evidence", "evidence", False),
        (axes[2], _smooth(acq), f"Acquisition (κ={kappa:g})", "acquisition", False),
    ]:
        subplot_topology(ax, x_axis, y_axis, xs, ys, grid, cmap_name=cmap_name,
                         label=label, points=points, codes=codes, point_size=18,
                         contour_overlay=overlay)
    if proposal is not None:
        axes[2].plot(proposal[_X], proposal[_Y], "x", color=ACCENT_YELLOW, ms=10,
                     markeredgewidth=2, zorder=8, label="Proposed")
        axes[2].legend(fontsize=FONT["legend"], loc="upper left", framealpha=0.8)
    path = os.path.join(PLOTS_DIR, f"acquisition_{tag}.png")
    save_fig(path)
    return path


_ACTIVE_PHASES = (
    ("discovery", ZINC_400, "o", "Discovery"),
    ("exploration", EMERALD_500, "o", "Exploration"),
    ("inference", ACCENT_YELLOW, "x", "Inference"),
)


def journey(active: list[tuple[str, float]], reference_best: float | None,
            reference_label: str) -> str:
    """Combined score vs experiment count for the active loop, with running best.

    ``active`` is an ordered ``[(phase, score), …]`` list across discovery →
    exploration → inference. ``reference_best`` is the best combined score the
    passive CCF grid reaches — the baseline the active loop is measured against.
    The headline ADVEI claim: the active loop matches it in far fewer runs.
    """
    import matplotlib.pyplot as plt

    ensure_plots_dir()
    apply_style()
    xs = list(range(1, len(active) + 1))
    scores = [s for _, s in active]

    best, running = -1.0, []
    for s in scores:
        best = max(best, s)
        running.append(best)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(xs, running, where="post", color=STEEL_500, lw=1.8, zorder=2)
    seen: set[str] = set()
    for x, (phase, score) in zip(xs, active):
        style = next((p for p in _ACTIVE_PHASES if p[0] == phase), None)
        color, marker = (style[1], style[2]) if style else (ZINC_400, "o")
        seen.add(phase)
        if marker == "x":
            ax.scatter(x, score, c=color, marker="x", s=80, linewidths=1.4, zorder=4)
        else:
            ax.scatter(x, score, c=color, marker="o", s=24, edgecolors="white",
                       linewidths=0.5, zorder=3)

    if reference_best is not None:
        ax.axhline(reference_best, color=ACCENT_RED, ls="--", lw=1.0, alpha=0.7, zorder=1)
        ax.annotate(f"{reference_label}  {reference_best:.2f}",
                    xy=(1, reference_best), xytext=(0, 4), textcoords="offset points",
                    fontsize=FONT["annotation"], color=ACCENT_RED)
        crossed = next((i + 1 for i, b in enumerate(running) if b >= reference_best), None)
        if crossed is not None:
            ax.annotate(f"matched in {crossed}", xy=(crossed, reference_best),
                        xytext=(4, -12), textcoords="offset points",
                        fontsize=FONT["annotation"], color=ZINC_600)

    handles = [Line2D([], [], color=STEEL_500, lw=1.8, label="best so far")] + [
        Line2D([], [], color=c, marker=m, ls="", markersize=6, label=lab)
        for key, c, m, lab in _ACTIVE_PHASES if key in seen
    ]
    if reference_best is not None:
        handles.append(Line2D([], [], color=ACCENT_RED, ls="--", lw=1.0, label=reference_label))
    ax.legend(handles=handles, loc="lower right", fontsize=FONT["legend"], frameon=False)

    ax.set_xlabel("Experiment", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_ylabel("Combined score", fontsize=FONT["axis_label"], color=ZINC_600)
    ax.set_title("Journey: active learning vs passive grid", fontsize=FONT["title"], color=ZINC_700)
    ax.grid(alpha=0.2, lw=0.8)
    ax.set_xticks([x for x in xs if x == 1 or x % 5 == 0])
    clean_spines(ax)

    path = os.path.join(PLOTS_DIR, "journey.png")
    save_fig(path)
    return path


def performance_radar(perf: dict[str, float], ref_mean: dict[str, float] | None,
                      tag: str, title: str) -> str:
    """Radar of the five performance attributes, optionally vs. a reference mean."""
    import matplotlib.pyplot as plt

    ensure_plots_dir()
    names = [display_name(a) for a in _ATTRS]
    values = [float(perf.get(a, 0.0)) for a in _ATTRS]
    ref = [float(ref_mean[a]) for a in _ATTRS] if ref_mean else None
    fig, ax = plt.subplots(figsize=(6.5, 6.0), subplot_kw={"projection": "polar"})
    radar_chart(
        ax, names, values,
        score=sum(values) / len(values),
        ref_values=ref,
        label=title, ref_label="dataset mean" if ref else None,
    )
    path = os.path.join(PLOTS_DIR, f"radar_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
