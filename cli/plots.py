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

from pred_fab.plotting import AxisSpec, radar_chart, subplot_topology, save_fig, apply_style
from pred_fab.plotting._style import ACCENT_YELLOW, FONT

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
    for ax, grid, label, cmap_name in [
        (axes[0], perf, "Performance", "performance"),
        (axes[1], ev, "Evidence", "evidence"),
        (axes[2], acq, f"Acquisition (κ={kappa:g})", "acquisition"),
    ]:
        subplot_topology(ax, x_axis, y_axis, xs, ys, grid, cmap_name=cmap_name,
                         label=label, points=points, codes=codes, point_size=18)
    if proposal is not None:
        axes[2].plot(proposal[_X], proposal[_Y], "x", color=ACCENT_YELLOW, ms=10,
                     markeredgewidth=2, zorder=8, label="Proposed")
        axes[2].legend(fontsize=FONT["legend"], loc="upper left", framealpha=0.8)
    path = os.path.join(PLOTS_DIR, f"acquisition_{tag}.png")
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
