"""Per-stage plotting helpers for the extrusion printing showcase."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pred_fab.core import ExperimentData, DataModule
from pred_fab import PfabAgent

os.makedirs("./plots", exist_ok=True)

_PHASE_COLORS = {"baseline": "#4C72B0", "exploration": "#DD8452", "inference": "#55A868"}

# ── Shared helpers ────────────────────────────────────────────────────────────

def _save_and_show(name: str) -> None:
    plt.tight_layout()
    plt.savefig(f"./plots/{name}.png", dpi=120, bbox_inches="tight")
    plt.close()


def _annotate_heatmap(ax: Any, grid: np.ndarray, fmt: str = "{:.4f}") -> None:
    """Overlay formatted value text on each cell of a heatmap."""
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(
                j, i, fmt.format(float(grid[i, j])),
                ha="center", va="center",
                fontsize=7, color="white",
                fontweight="bold",
            )


# ── Phase 1: As-Printed vs As-Designed (per-layer 2D grid) ───────────────────

def plot_path_comparison(
    exp_data: ExperimentData,
    camera: Any,
    params: Dict[str, Any],
) -> None:
    """Per-layer 2D grid: designed (grey dashed at y=0) vs as-printed (coloured by deviation).

    One subplot per layer. The designed path is always y=0 (perfectly straight),
    so any vertical spread in the measured line is path deviation. The fill between
    the two makes deviation immediately visible; layer drift accumulates top-to-bottom.
    """
    N_LAYERS, N_SEGMENTS, N_PTS = 5, 4, 5
    SEG_LENGTH = (N_PTS - 1) * 0.01   # 0.04 m per segment
    SEG_GAP    = 0.008                 # gap between segments

    # Pre-fetch all segment data and find the global deviation range for a shared colormap
    cache: Dict[Tuple[int, int], Dict] = {}
    all_devs: List[float] = []
    for li in range(N_LAYERS):
        for si in range(N_SEGMENTS):
            data = camera.get_segment_data(params, li, si)
            cache[(li, si)] = data
            for mp, dp in zip(data["measured_path"], data["designed_path"]):
                all_devs.append(abs(mp[1] - dp[1]))

    vmax = max(all_devs) * 1.1 if all_devs else 1e-4
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]

    fig, axes = plt.subplots(1, N_LAYERS, figsize=(14, 3.2), sharey=True)

    design   = params.get("design",      "?")
    material = params.get("material",    "?")
    speed    = params.get("print_speed", 0.0)
    fig.suptitle(
        f"As-Printed vs As-Designed  ·  {exp_data.code}\n"
        f"design={design}   material={material}   speed={speed:.1f} mm/s",
        fontsize=11, fontweight="bold",
    )

    for li, ax in enumerate(axes):
        x_off = 0.0
        all_x: List[float] = []
        all_y_meas: List[float] = []
        all_y_des: List[float]  = []

        for si in range(N_SEGMENTS):
            data = cache[(li, si)]
            for (xd, yd), (_, ym) in zip(data["designed_path"], data["measured_path"]):
                all_x.append(xd + x_off)
                all_y_des.append(yd)
                all_y_meas.append(ym)
            x_off += SEG_LENGTH + SEG_GAP

        devs     = [abs(ym - yd) for ym, yd in zip(all_y_meas, all_y_des)]
        mean_dev = float(np.mean(devs))
        line_col = cmap(norm(mean_dev))

        # Reference (designed) path is always y=0
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.2,
                   label="Designed" if li == 0 else None)
        # Measured path and shaded deviation area
        ax.plot(all_x, all_y_meas, color=line_col, linewidth=1.8,
                label="Measured" if li == 0 else None)
        ax.fill_between(all_x, all_y_des, all_y_meas, color=line_col, alpha=0.25)

        ax.set_title(f"Layer {li}\nΔ = {mean_dev * 1000:.2f} mm", fontsize=8)
        ax.set_xlabel("Along-path [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        if li == 0:
            ax.set_ylabel("Lateral offset [m]", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")

    # Shared colorbar on the right
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[-1], pad=0.03, fraction=0.06)
    cb.set_label("Mean deviation [m]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    _save_and_show("path_comparison")


# ── Phase 1: Physics landscape ────────────────────────────────────────────────

def plot_physics_landscape(params: Dict[str, Any]) -> None:
    """U-shaped deviation vs print_speed curve for the experiment's conditions.

    Shows where the actual experiment speed sits relative to the theoretical optimum.
    The U-shape arises from two competing effects: high-speed inertia overshoot and
    low-speed material sag. This plot gives the user intuition for why the optimizer
    converges to a specific speed range for each design/material combination.
    """
    from sensors.physics import (  # type: ignore[import-not-found]
        path_deviation as _phys_dev,
        DELTA, THETA, DESIGN_COMPLEXITY, MATERIAL_VISCOSITY, KAPPA, W_OPTIMAL_WATER,
    )

    design      = str(params.get("design",      "B"))
    material    = str(params.get("material",    "standard"))
    water_ratio = float(params.get("water_ratio", 0.40))
    act_speed   = float(params.get("print_speed", 40.0))
    n_segments  = int(params.get("n_segments",    4))

    speeds = np.linspace(20.0, 60.0, 200)
    # Average deviation over segments at layer 0 (no layer-drift offset)
    mean_devs = np.array([
        float(np.mean([_phys_dev(spd, design, si, water_ratio, material, layer_idx=0)
                       for si in range(n_segments)]))
        for spd in speeds
    ])

    # Theoretical optimal speed (from physics formula, clipped to search bounds)
    complexity = DESIGN_COMPLEXITY[design]
    viscosity  = MATERIAL_VISCOSITY[material]
    w_opt      = W_OPTIMAL_WATER[material]
    flow       = max(0.1, 1.0 - KAPPA * (water_ratio - w_opt) ** 2)
    spd_opt    = float(np.clip(np.sqrt(THETA * viscosity / (DELTA * complexity * flow)), 20.0, 60.0))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(
        f"Path Deviation vs Print Speed\n"
        f"design={design}   material={material}   water_ratio={water_ratio:.2f}",
        fontsize=11, fontweight="bold",
    )

    ax.plot(speeds, mean_devs, color=_PHASE_COLORS["inference"], linewidth=2.2,
            label="Physics model (mean over segments)")

    # Vertical: actual experiment speed
    act_dev = float(np.interp(act_speed, speeds, mean_devs))
    ax.axvline(act_speed, color=_PHASE_COLORS["exploration"], linestyle="-", linewidth=1.5,
               label=f"Experiment speed ({act_speed:.1f} mm/s)")
    ax.scatter([act_speed], [act_dev], color=_PHASE_COLORS["exploration"], s=55, zorder=5)

    # Vertical: theoretical optimum
    opt_dev = float(np.interp(spd_opt, speeds, mean_devs))
    ax.axvline(spd_opt, color=_PHASE_COLORS["baseline"], linestyle="--", linewidth=1.5,
               label=f"Optimal speed ({spd_opt:.1f} mm/s)")
    ax.scatter([spd_opt], [opt_dev], color=_PHASE_COLORS["baseline"], s=55, zorder=5)

    # Shade the low-speed (sag) and high-speed (overshoot) regions
    ax.axvspan(20.0, spd_opt, alpha=0.06, color="#3377CC", label="Sag-dominated")
    ax.axvspan(spd_opt, 60.0, alpha=0.06, color="#CC3333", label="Inertia-dominated")

    ax.set_xlabel("Print Speed [mm/s]", fontsize=10)
    ax.set_ylabel("Mean Path Deviation [m]", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    _save_and_show("physics_landscape")


# ── Phase 1: Feature heatmaps ─────────────────────────────────────────────────

def plot_feature_heatmaps(exp_data: ExperimentData) -> None:
    """2-panel (5×4) heatmaps of path_deviation and energy_per_segment with value labels."""
    n_layers, n_segments = 5, 4
    features_cfg = [
        ("path_deviation",     "Path Deviation [m]",      "RdYlGn_r"),
        ("energy_per_segment", "Energy per Segment [J]",  "plasma"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Feature Heatmaps — {exp_data.code}", fontsize=12, fontweight="bold")

    for ax, (fname, label, cmap_name) in zip(axes, features_cfg):
        grid = exp_data.features.get_value(fname)  # type: ignore[return-value]
        im = ax.imshow(grid, aspect="auto", cmap=cmap_name)
        ax.set_xlabel("Segment", fontsize=9)
        ax.set_ylabel("Layer",   fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(n_segments))
        ax.set_xticklabels([f"S{i}" for i in range(n_segments)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        _annotate_heatmap(ax, grid)

    _save_and_show("feature_heatmaps")


# ── Phase 2: Prediction accuracy ─────────────────────────────────────────────

def plot_prediction_accuracy(agent: PfabAgent, datamodule: DataModule) -> None:
    """Scatter of predicted vs actual for both output features with R² annotation."""
    from sklearn.metrics import r2_score
    from pred_fab.utils import SplitType  # type: ignore[attr-defined]

    pred_system = agent.pred_system
    outputs     = pred_system.get_system_outputs()

    val_batches = datamodule.get_batches(SplitType.VAL)
    if not val_batches:
        print("[plot_prediction_accuracy] No validation batches available.")
        return

    # Each prediction model may use a different subset of input columns —
    # replicate the same filtering the PredictionSystem uses during training.
    all_val_batches = list(val_batches)
    y_val = np.vstack([b[1] for b in all_val_batches])

    y_pred_cols: List[np.ndarray] = []
    for model in pred_system.models:
        model_batches = pred_system._filter_batches_for_model(  # type: ignore[attr-defined]
            all_val_batches, model
        )
        X_m = np.vstack([b[0] for b in model_batches])
        y_pred_cols.append(model.forward_pass(X_m))
    y_pred = np.hstack(y_pred_cols) if y_pred_cols else y_val

    n_out = min(y_val.shape[1], len(outputs))
    fig, axes = plt.subplots(1, n_out, figsize=(6 * n_out, 5))
    if n_out == 1:
        axes = [axes]  # type: ignore[assignment]

    fig.suptitle("Prediction Accuracy — validation set", fontsize=12, fontweight="bold")

    for i, (ax, name) in enumerate(zip(axes, outputs)):
        if i >= y_pred.shape[1] or i >= y_val.shape[1]:
            break
        r2  = r2_score(y_val[:, i], y_pred[:, i])
        lim = [
            min(float(y_val[:, i].min()), float(y_pred[:, i].min())),
            max(float(y_val[:, i].max()), float(y_pred[:, i].max())),
        ]
        ax.scatter(y_val[:, i], y_pred[:, i], alpha=0.7,
                   color=_PHASE_COLORS["exploration"], edgecolors="white", linewidths=0.4)
        ax.plot(lim, lim, "k--", linewidth=1.2, label="Perfect prediction")
        ax.set_xlabel("Actual",    fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{name}\nR² = {r2:.3f}", fontsize=10)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=8)

    _save_and_show("prediction_accuracy")


# ── Phase 3: Parameter space ──────────────────────────────────────────────────

def plot_parameter_space(
    all_params: List[Dict[str, Any]],
    phases: List[str],
    perf_history: Optional[List[Tuple[Dict[str, Any], Dict[str, float]]]] = None,
) -> None:
    """2-D scatter of water_ratio vs print_speed.

    Points are sized by experiment order (early=small) and coloured by combined
    performance score when perf_history is provided, otherwise by phase.
    Phase is encoded in marker shape: baseline=circle, exploration=star.
    Design is encoded in edge colour so all four axes of variation are visible.
    """
    water_ratios = [p["water_ratio"] for p in all_params]
    print_speeds = [p["print_speed"] for p in all_params]
    n = len(all_params)

    # Build combined score array if perf_history supplied
    scores: Optional[List[float]] = None
    if perf_history is not None and len(perf_history) == n:
        def _combined(perf: Dict[str, float]) -> float:
            acc = perf.get("path_accuracy", 0.0)
            eff = perf.get("energy_efficiency", 0.0)
            return 0.5 * acc + 0.5 * eff

        scores = [_combined(ph[1]) for ph in perf_history]

    _MARKERS   = {"baseline": "o", "exploration": "*", "inference": "D"}
    _DESIGN_EC = {"A": "#2266AA", "B": "#AA6622", "C": "#226622"}  # edge colour by design

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Parameter Space — Baseline & Exploration", fontsize=12, fontweight="bold")

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    vmin = min(scores) if scores else 0.0
    vmax = max(scores) if scores else 1.0
    _last_sc = None

    for phase, marker in _MARKERS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if not idx:
            continue
        sizes      = [40 + 5 * i for i in idx]
        edge_cols  = [_DESIGN_EC.get(str(all_params[i].get("design", "A")), "#333333") for i in idx]
        if scores is not None:
            sc = ax.scatter(
                [water_ratios[i] for i in idx],
                [print_speeds[i]  for i in idx],
                c=[scores[i] for i in idx],
                cmap=cmap, vmin=vmin, vmax=vmax,
                marker=marker, s=sizes,
                edgecolors=edge_cols, linewidths=1.2,
                alpha=0.9, zorder=3, label=phase.capitalize(),
            )
            _last_sc = sc
        else:
            ax.scatter(
                [water_ratios[i] for i in idx],
                [print_speeds[i]  for i in idx],
                c=_PHASE_COLORS[phase], marker=marker, s=sizes,
                edgecolors=edge_cols, linewidths=1.2,
                alpha=0.85, zorder=3, label=phase.capitalize(),
            )

    if scores is not None and _last_sc is not None:
        cb = fig.colorbar(_last_sc, ax=ax, pad=0.02)
        cb.set_label("Combined Score [0–1]", fontsize=9)

    # Design legend patches
    design_patches = [
        mpatches.Patch(edgecolor=ec, facecolor="none", linewidth=1.5, label=f"Design {d}")
        for d, ec in _DESIGN_EC.items()
    ]
    phase_legend  = ax.legend(fontsize=9, title="Phase", title_fontsize=8, loc="upper left")
    ax.add_artist(phase_legend)
    ax.legend(handles=design_patches, fontsize=8, title="Design", title_fontsize=8,
              loc="lower right")

    ax.set_xlabel("Water Ratio",        fontsize=10)
    ax.set_ylabel("Print Speed [mm/s]", fontsize=10)
    ax.grid(True, alpha=0.25)
    _save_and_show("parameter_space")


# ── Phase 4: Performance trajectory ──────────────────────────────────────────

def plot_performance_trajectory(
    exp_params_and_perf: List[Tuple[Dict[str, Any], Dict[str, float]]],
    phases: List[str],
    exp_codes: Optional[List[str]] = None,
) -> None:
    """Dual-line plot with shaded phase bands, phase labels, and optional x-tick experiment codes."""
    path_acc   = [pp[1].get("path_accuracy",    float("nan")) for pp in exp_params_and_perf]
    energy_eff = [pp[1].get("energy_efficiency", float("nan")) for pp in exp_params_and_perf]
    xs = list(range(1, len(path_acc) + 1))

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle("Performance Trajectory", fontsize=12, fontweight="bold")

    # Shade phase regions
    phase_alpha = {"baseline": 0.08, "exploration": 0.10, "inference": 0.10}
    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            ax.axvspan(idx[0] + 0.5, idx[-1] + 1.5,
                       color=color, alpha=phase_alpha.get(phase, 0.08), label=f"_{phase}_band")

    ax.plot(xs, path_acc,   marker="o", color=_PHASE_COLORS["exploration"],
            linewidth=1.8, markersize=5, label="Path Accuracy")
    ax.plot(xs, energy_eff, marker="s", color=_PHASE_COLORS["inference"],
            linewidth=1.8, markersize=5, label="Energy Efficiency")

    # Phase labels at top
    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            mid = (idx[0] + idx[-1]) / 2 + 1
            ax.text(mid, 1.03, phase.capitalize(), ha="center", va="bottom",
                    color=color, fontsize=8, transform=ax.get_xaxis_transform())

    # X-tick labels: experiment codes if supplied, otherwise numbers
    if exp_codes and len(exp_codes) == len(xs):
        ax.set_xticks(xs)
        ax.set_xticklabels(exp_codes, rotation=35, ha="right", fontsize=7)
    else:
        ax.set_xlabel("Experiment #", fontsize=10)

    ax.set_ylabel("Score [0–1]",  fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _save_and_show("performance_trajectory")


# ── Phase 5: Online adaptation ────────────────────────────────────────────────

def plot_adaptation(
    layer_speeds: List[float],
    deviations: List[float],
    no_adapt_deviations: Optional[List[float]] = None,
) -> None:
    """Two stacked subplots: print_speed per layer (top) and path deviation (bottom).

    The two-subplot layout avoids dual-axis confusion. The counterfactual line
    (fixed speed=40, no adaptation) makes the benefit of adaptation clearly visible.
    """
    layers = list(range(len(layer_speeds)))

    fig, (ax_spd, ax_dev) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.suptitle("Online Adaptation — Layer-by-Layer", fontsize=12, fontweight="bold")

    # ── Top: print_speed ──────────────────────────────────────────────────────
    ax_spd.plot(layers, layer_speeds, marker="o", color=_PHASE_COLORS["baseline"],
                linewidth=1.8, markersize=6, label="Print Speed (adapted)")
    ax_spd.set_ylabel("Print Speed [mm/s]", fontsize=10)
    ax_spd.legend(fontsize=9, loc="upper right")
    ax_spd.grid(True, alpha=0.25)

    # ── Bottom: deviation ─────────────────────────────────────────────────────
    ax_dev.plot(layers, deviations, marker="s", color=_PHASE_COLORS["exploration"],
                linewidth=1.8, markersize=6, label="Deviation (adapted)")
    if no_adapt_deviations is not None:
        ax_dev.plot(layers, no_adapt_deviations, marker="^", color="#CC4444",
                    linestyle="--", linewidth=1.5, markersize=5, alpha=0.80,
                    label="Deviation (no adaptation, speed=40)")
        # Shade the saving between curves
        ax_dev.fill_between(
            layers, no_adapt_deviations, deviations,
            where=[n > d for n, d in zip(no_adapt_deviations, deviations)],
            alpha=0.15, color="#228833", label="Deviation saved",
        )
    ax_dev.set_xlabel("Layer", fontsize=10)
    ax_dev.set_ylabel("Avg Path Deviation [m]", fontsize=10)
    ax_dev.legend(fontsize=9, loc="upper left")
    ax_dev.grid(True, alpha=0.25)
    ax_dev.set_xticks(layers)
    ax_dev.set_xticklabels([f"L{i}" for i in layers])

    _save_and_show("adaptation")
