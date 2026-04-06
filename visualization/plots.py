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
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import r2_score

from pred_fab.core import ExperimentData, DataModule
from pred_fab import PfabAgent
from pred_fab.utils import SplitType  # type: ignore[attr-defined]

# Local imports (type: ignore to avoid pyright issues with local modules)
from sensors.physics import FILAMENT_RADIUS, path_deviation, energy_per_segment 
from models.evaluation_models import PathAccuracyModel, EnergyConsumptionModel, ProductionRateModel 

os.makedirs("./plots", exist_ok=True)

_PHASE_COLORS = {"baseline": "#4C72B0", "exploration": "#DD8452", "inference": "#55A868"}

# ── Shared helpers ────────────────────────────────────────────────────────────

def _save(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
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


# ── Filament tube helper ──────────────────────────────────────────────────────

def _make_filament_tube(
    xs: List[float],
    ys: List[float],
    z_center: float,
    radius: float,
    n_circ: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) surface arrays for a cylindrical tube following (xs, ys) at z_center.

    The tube cross-section is circular with given radius. Suitable for plot_surface.
    Shape of each output array: (n_circ, len(xs)).
    """
    n_pts = len(xs)
    phi = np.linspace(0, 2.0 * np.pi, n_circ, endpoint=True)
    PHI = phi[:, np.newaxis]                       # (n_circ, 1)
    XS  = np.array(xs, dtype=float)[np.newaxis, :] # (1, n_pts)
    YS  = np.array(ys, dtype=float)[np.newaxis, :] # (1, n_pts)
    X = np.repeat(XS, n_circ, axis=0)              # (n_circ, n_pts)
    Y = YS + radius * np.cos(PHI)                  # (n_circ, n_pts)
    Z = z_center + radius * np.sin(PHI)             # (n_circ, n_pts)
    return X, Y, Z




# ── Phase 2: Prediction accuracy ─────────────────────────────────────────────

def plot_prediction_accuracy(
    agent: PfabAgent,
    datamodule: DataModule,
    save_dir: str = "./plots/phase_2_training",
) -> Dict[str, float]:
    """Scatter of predicted vs actual for both output features with R² annotation.

    Returns a dict mapping feature name → R² on the validation set.
    """
    pred_system = agent.pred_system
    outputs     = pred_system.get_system_outputs()

    val_batches = datamodule.get_batches(SplitType.VAL)
    if not val_batches:
        print("[plot_prediction_accuracy] No validation batches available.")
        return {}

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

    r2_scores: Dict[str, float] = {}
    for i, (ax, name) in enumerate(zip(axes, outputs)):
        if i >= y_pred.shape[1] or i >= y_val.shape[1]:
            break
        r2  = r2_score(y_val[:, i], y_pred[:, i])
        r2_scores[name] = float(r2)
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

    _save(os.path.join(save_dir, "prediction_accuracy.png"))
    return r2_scores


# ── Phase 3: Parameter space ──────────────────────────────────────────────────

def plot_parameter_space(
    all_params: List[Dict[str, Any]],
    phases: List[str],
    perf_history: Optional[List[Tuple[Dict[str, Any], Dict[str, float]]]] = None,
    save_dir: str = "./plots/phase_3_exploration",
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
    # Collect designs actually present in the data
    present_designs = sorted({str(p.get("design", "A")) for p in all_params})
    _DESIGN_EC_ALL = {"A": "#2266AA", "B": "#AA6622", "C": "#226622"}
    _DESIGN_EC = {d: _DESIGN_EC_ALL[d] for d in present_designs if d in _DESIGN_EC_ALL}

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
    _save(os.path.join(save_dir, "parameter_space.png"))


# ── Phase 4: Performance trajectory ──────────────────────────────────────────

def plot_performance_trajectory(
    exp_params_and_perf: List[Tuple[Dict[str, Any], Dict[str, float]]],
    phases: List[str],
    exp_codes: Optional[List[str]] = None,
    save_dir: str = "./plots/phase_4_inference",
) -> None:
    """Dual-line plot with shaded phase bands, phase labels, and optional x-tick experiment codes."""
    path_acc   = [pp[1].get("path_accuracy",    float("nan")) for pp in exp_params_and_perf]
    energy_eff = [pp[1].get("energy_efficiency", float("nan")) for pp in exp_params_and_perf]
    prod_rate  = [pp[1].get("production_rate",   float("nan")) for pp in exp_params_and_perf]
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
    ax.plot(xs, prod_rate,  marker="^", color="#8856a7",
            linewidth=1.2, markersize=4, alpha=0.7, label="Production Rate")

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
    _save(os.path.join(save_dir, "performance_trajectory.png"))


# ── Phase 5: Online adaptation ────────────────────────────────────────────────

def plot_adaptation(
    layer_speeds: List[float],
    deviations: List[float],
    no_adapt_deviations: Optional[List[float]] = None,
    save_dir: str = "./plots/phase_5_adaptation",
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

    _save(os.path.join(save_dir, "adaptation.png"))


# ── Phase 1: 3D tube comparison ───────────────────────────────────────────────

def plot_path_comparison_3d(
    exp_data: ExperimentData,
    camera: Any,
    params: Dict[str, Any],
    save_dir: str = "./plots/phase_1_baseline",
) -> None:
    """3D stacked tube view: designed (grey wireframe) vs as-printed (coloured solid).

    Each filament is a cylindrical tube. The designed path is rendered as a wireframe
    ghost so the measured path (solid, coloured by deviation) reads clearly in front.
    Layer drift accumulates upward — the colour shift from green → red tells the story.
    """

    N_LAYERS, N_SEGMENTS, N_PTS = 5, 4, 5
    SEG_LENGTH = (N_PTS - 1) * 0.01
    SEG_GAP    = 0.008

    from sensors.physics import FILAMENT_RADIUS  # type: ignore[import-not-found]
    radius     = FILAMENT_RADIUS
    LAYER_STEP = radius * 2.6   # gap > diameter so layers don't touch visually

    cache: Dict[Tuple[int, int], Dict] = {}
    all_devs: List[float] = []
    for li in range(N_LAYERS):
        for si in range(N_SEGMENTS):
            d = camera.get_segment_data(params, li, si)
            cache[(li, si)] = d
            for mp, dp in zip(d["measured_path"], d["designed_path"]):
                all_devs.append(abs(mp[1] - dp[1]))

    vmax = max(all_devs) * 1.1 if all_devs else 1e-4
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]

    fig = plt.figure(figsize=(15, 8))
    ax  = fig.add_subplot(111, projection="3d")

    Y_SCALE = 3.0  # exaggerate lateral deviation so it reads clearly in 3D

    for li in range(N_LAYERS):
        z_center = li * LAYER_STEP
        x_off    = 0.0

        for si in range(N_SEGMENTS):
            data    = cache[(li, si)]
            dp      = data["designed_path"]
            mp      = data["measured_path"]
            xs        = [p[0] + x_off for p in dp]
            ys_des    = [0.0] * len(dp)
            ys_meas_r = [p[1] for p in mp]
            ys_meas_v = [y * Y_SCALE for y in ys_meas_r]   # visual (scaled)

            mean_dev   = float(np.mean([abs(ym) for ym in ys_meas_r]))
            tube_color = cmap(norm(mean_dev))

            # Designed — wireframe ghost at y=0
            Xd, Yd, Zd = _make_filament_tube(xs, ys_des, z_center, radius, n_circ=16)
            ax.plot_wireframe(  # type: ignore[union-attr]
                Xd, Yd, Zd,
                color="#6699CC", alpha=0.35, linewidth=0.4,
                rstride=4, cstride=1,
            )
            # Designed centreline — dashed blue-white
            ax.plot(  # type: ignore[union-attr]
                xs, ys_des, [z_center] * len(xs),
                color="#AACCFF", linestyle="--", linewidth=1.0, alpha=0.80, zorder=4,
            )

            # Measured — solid coloured tube (y-scaled)
            Xm, Ym, Zm = _make_filament_tube(xs, ys_meas_v, z_center, radius, n_circ=20)
            ax.plot_surface(  # type: ignore[union-attr]
                Xm, Ym, Zm,
                color=tube_color, alpha=0.88, linewidth=0, antialiased=True, shade=True,
            )

            x_off += SEG_LENGTH + SEG_GAP

    # Fix aspect ratio: amplify y and z display relative to x so tubes look round
    ax.set_box_aspect([9, 2.5, 2.2])  # type: ignore[union-attr]

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.09, shrink=0.52, aspect=16)
    cb.set_label("Path Deviation [m]", fontsize=9)

    design   = params.get("design",      "?")
    material = params.get("material",    "?")
    speed    = params.get("print_speed", 0.0)
    water    = params.get("water_ratio", 0.0)

    ax.set_xlabel("Along-path [m]",    labelpad=10, fontsize=9)   # type: ignore[union-attr]
    ax.set_ylabel(f"Lateral offset [m] (×{Y_SCALE:.0f})", labelpad=10, fontsize=9)  # type: ignore[union-attr]
    ax.set_zticks([i * LAYER_STEP for i in range(N_LAYERS)])      # type: ignore[union-attr]
    ax.set_zticklabels([f"Layer {i}" for i in range(N_LAYERS)])        # type: ignore[union-attr]
    
    # Clean title structure
    fig.suptitle(
        f"3D Path Comparison — {exp_data.code}",
        fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"Design {design} · {material} · {speed:.1f} mm/s · water_ratio={water:.2f}",
        fontsize=9, pad=12
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#AACCFF", linestyle="--", linewidth=1.5, 
               label="Designed path"),
        Line2D([0], [0], color="#666666", linewidth=3, 
               label="As-printed (colored by deviation)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)
    
    ax.view_init(elev=28, azim=-62)  # type: ignore[union-attr]
    _save(os.path.join(save_dir, "path_comparison_3d.png"))


# ── Phase 1: Volumetric filament close-up ─────────────────────────────────────

def plot_filament_volume(
    exp_data: ExperimentData,
    camera: Any,
    params: Dict[str, Any],
    save_dir: str = "./plots/phase_1_baseline",
) -> None:
    """Close-up 3D filament volume: designed (grey wireframe) vs as-printed (coloured solid).

    Focuses on two middle segments of layers 0 and 4. The y-axis is scaled ×3 for
    visibility — the physical deviation is smaller than the filament diameter, so
    this exaggeration lets the lateral offset read clearly in 3D space.
    Deviation arrows mark the centreline offset at the midpoint of each segment.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    SEG_LENGTH   = 0.04      # 5 pts × 0.01 m spacing
    SEG_GAP      = 0.008
    SHOW_SEGS    = [1, 2]    # middle two segments — highest curvature
    SHOW_LAYERS  = [0, 4]    # first vs last layer
    LAYER_STEP   = 0.022     # vertical gap between the two displayed layers
    Y_SCALE      = 3.0       # exaggerate y for visibility

    from sensors.physics import FILAMENT_RADIUS  # type: ignore[import-not-found]
    radius = FILAMENT_RADIUS

    all_devs: List[float] = []
    for li in SHOW_LAYERS:
        for si in SHOW_SEGS:
            d = camera.get_segment_data(params, li, si)
            for mp, dp in zip(d["measured_path"], d["designed_path"]):
                all_devs.append(abs(mp[1] - dp[1]) * Y_SCALE)
    vmax = max(all_devs) * 1.2 if all_devs else 1e-4
    norm = Normalize(vmin=0.0, vmax=vmax / Y_SCALE)   # colormap in real units
    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]

    fig = plt.figure(figsize=(13, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for row_idx, li in enumerate(SHOW_LAYERS):
        z_center = row_idx * LAYER_STEP
        # x offset: start from second segment position
        x_start = SHOW_SEGS[0] * (SEG_LENGTH + SEG_GAP)
        x_off   = 0.0

        for si in SHOW_SEGS:
            d      = camera.get_segment_data(params, li, si)
            dp     = d["designed_path"]
            mp     = d["measured_path"]

            seg_radius = radius

            xs         = [p[0] + x_off for p in dp]
            ys_des     = [0.0] * len(dp)
            ys_meas_r  = [p[1] for p in mp]                         # real
            ys_meas_v  = [y * Y_SCALE for y in ys_meas_r]           # visual (scaled)

            mean_dev   = float(np.mean([abs(ym) for ym in ys_meas_r]))
            tube_color = cmap(norm(mean_dev))

            # Designed — blue wireframe ghost
            Xd, Yd, Zd = _make_filament_tube(xs, ys_des,    z_center, seg_radius, n_circ=22)
            ax.plot_wireframe(  # type: ignore[union-attr]
                Xd, Yd, Zd, color="#4488CC", alpha=0.50, linewidth=0.6,
                rstride=3, cstride=1,
            )
            ax.plot(  # type: ignore[union-attr]
                xs, ys_des, [z_center] * len(xs),
                color="#AACCFF", linestyle="--", linewidth=1.2, alpha=0.85, zorder=5,
            )

            # Measured — solid coloured tube (y-scaled)
            Xm, Ym, Zm = _make_filament_tube(xs, ys_meas_v, z_center, seg_radius, n_circ=28)
            ax.plot_surface(  # type: ignore[union-attr]
                Xm, Ym, Zm,
                color=tube_color, alpha=0.90, linewidth=0, antialiased=True, shade=True,
            )
            ax.plot(  # type: ignore[union-attr]
                xs, ys_meas_v, [z_center] * len(xs),
                color="#222222", linewidth=0.7, alpha=0.6, zorder=5,
            )

            # Deviation arrow at segment midpoint (above tube, z_center+seg_radius)
            mid      = len(xs) // 2
            x_mid    = xs[mid]
            y_meas_m = ys_meas_v[mid]
            z_arrow  = z_center + seg_radius * 1.1   # sit just above the tube
            if abs(y_meas_m) > 1e-6:
                ax.quiver(  # type: ignore[union-attr]
                    x_mid, 0.0, z_arrow,
                    0, y_meas_m, 0,
                    color="#FF2222", linewidth=2.0, arrow_length_ratio=0.25,
                    alpha=0.95,
                )

            x_off += SEG_LENGTH + SEG_GAP

    # Box aspect: x wide, y and z enlarged relative to data
    ax.set_box_aspect([5, 2.5, 2.0])  # type: ignore[union-attr]

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.09, shrink=0.50, aspect=16)
    cb.set_label("Path Deviation [m]", fontsize=9)

    design   = params.get("design",      "?")
    material = params.get("material",    "?")
    speed    = params.get("print_speed", 0.0)
    water    = params.get("water_ratio", 0.0)

    ax.set_xlabel("Along-path [m]",          labelpad=10, fontsize=9)  # type: ignore[union-attr]
    ax.set_ylabel(f"Lateral offset [m] (×{Y_SCALE:.0f})", labelpad=10, fontsize=9)  # type: ignore[union-attr]
    ax.set_zticks([i * LAYER_STEP for i in range(len(SHOW_LAYERS))])  # type: ignore[union-attr]
    ax.set_zticklabels([f"Layer {li}" for li in SHOW_LAYERS])               # type: ignore[union-attr]
    
    # Clean title structure
    fig.suptitle(
        f"Filament Volume Comparison — {exp_data.code}",
        fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"Design {design} · {material} · {speed:.1f} mm/s · water_ratio={water:.2f}",
        fontsize=9, pad=12
    )
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color="#4488CC", linestyle="--", linewidth=1.5, 
               label="Designed path"),
        Line2D([0], [0], color="#666666", linewidth=3, 
               label="As-printed (colored by deviation)"),
        Line2D([0], [0], color="#FF2222", marker=">", linestyle="None",
               markersize=8, label="Deviation arrows"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)
    
    ax.view_init(elev=18, azim=-72)  # type: ignore[union-attr]
    _save(os.path.join(save_dir, "filament_volume.png"))


# ── Phase 4: Inference convergence ───────────────────────────────────────────

def plot_inference_convergence(
    infer_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
    design_intent: Dict[str, Any],
    save_dir: str = "./plots/phase_4_inference",
) -> None:
    """2D parameter space for the inference intent showing optimizer convergence.

    Background contour: combined physics score (deviation + energy) across
    (water_ratio, print_speed) for the fixed design and material. The 3 inference
    experiments are plotted as a connected trajectory. The physics optimum (minimum
    deviation, ignoring energy) is marked as a white star. Together these show whether
    the model converged toward the true optimum and how quickly.
    """
    from sensors.physics import (  # type: ignore[import-not-found]
        path_deviation as _phys_dev,
        energy_per_segment as _phys_eng,
        DELTA, THETA, DESIGN_COMPLEXITY, MAT_SAG, KAPPA, W_OPTIMAL,
    )

    design   = str(design_intent.get("design",   "B"))
    material = str(design_intent.get("material", "clay"))

    # Physics background grid
    n_grid = 35
    water_vals = np.linspace(0.30, 0.50, n_grid)
    speed_vals = np.linspace(20.0, 60.0, n_grid)
    W, S = np.meshgrid(water_vals, speed_vals)

    N_SEGS = 4
    dev_grid = np.zeros_like(W)
    eng_grid = np.zeros_like(W)
    for i in range(n_grid):
        for j in range(n_grid):
            spd, w = float(S[i, j]), float(W[i, j])
            dev_grid[i, j] = float(np.mean([
                _phys_dev(spd, design, si, w, material, layer_idx=0)
                for si in range(N_SEGS)
            ]))
            eng_grid[i, j] = float(np.mean([
                _phys_eng(spd, material, design, w, si, 0)
                for si in range(N_SEGS)
            ]))

    # Normalise each to [0,1] then combine
    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-12)

    acc_grid  = 1.0 - _norm01(dev_grid)   # higher = less deviation = better
    eff_grid  = 1.0 - _norm01(eng_grid)   # higher = less energy = better
    combined  = 0.5 * acc_grid + 0.5 * eff_grid

    # Theoretical optimal speed at the material's best water_ratio
    w_opt      = W_OPTIMAL[material]
    complexity = DESIGN_COMPLEXITY[design]
    sag_f      = MAT_SAG[material]
    spd_opt    = float(np.clip(np.sqrt(THETA * sag_f / (DELTA * complexity)), 20.0, 60.0))

    fig, ax = plt.subplots(figsize=(9, 6))
    intent_str = f"design={design}   material={material}"
    fig.suptitle(
        f"Inference Convergence  ·  {intent_str}\n"
        f"Background = combined physics score (path accuracy + energy efficiency)",
        fontsize=11, fontweight="bold",
    )

    cf = ax.contourf(W, S, combined, levels=30, cmap="RdYlGn", alpha=0.85)
    ax.contour(W, S, combined, levels=8, colors="white", alpha=0.18, linewidths=0.5)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("Combined Score [0–1]", fontsize=9)

    # Physics optimum star
    ax.scatter([w_opt], [spd_opt], marker="*", s=350, color="white",
               edgecolors="#333333", linewidths=0.8, zorder=10,
               label=f"Physics optimum  (w={w_opt:.2f}, spd={spd_opt:.1f})")

    # Inference trajectory
    ws    = [p.get("water_ratio", 0.0) for _, p, _ in infer_log]
    spds  = [p.get("print_speed",  0.0) for _, p, _ in infer_log]
    codes = [c for c, _, _ in infer_log]
    perfs = [pf for _, _, pf in infer_log]

    # Alternate label offsets to avoid overlap when points are close
    offsets = [(10, 8), (-10, -18), (10, -18)]
    for i, (w, spd, code, perf) in enumerate(zip(ws, spds, codes, perfs)):
        comb_val = 0.5 * perf.get("path_accuracy", 0.0) + 0.5 * perf.get("energy_efficiency", 0.0)
        ax.scatter([w], [spd], s=100, marker="D",
                   color=_PHASE_COLORS["inference"],
                   edgecolors="white", linewidths=0.8, zorder=9)
        ofs = offsets[i % len(offsets)]
        ax.annotate(
            f"{code}\n({comb_val:.2f})",
            (w, spd), xytext=ofs, textcoords="offset points",
            fontsize=7, color="white", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="white", lw=0.5, alpha=0.5),
        )
        if i > 0:
            ax.annotate(
                "", xy=(ws[i], spds[i]), xytext=(ws[i - 1], spds[i - 1]),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6),
            )

    ax.scatter([], [], marker="D", color=_PHASE_COLORS["inference"],
               edgecolors="white", linewidths=0.8, label="Inference experiments")

    ax.set_xlabel("Water Ratio",        fontsize=10)
    ax.set_ylabel("Print Speed [mm/s]", fontsize=10)
    ax.set_xlim(0.30, 0.50)
    ax.set_ylim(20.0, 60.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.15)
    _save(os.path.join(save_dir, "inference_convergence.png"))


# ── Acquisition topology (exploration & inference) ────────────────────────────

def plot_acquisition_topology(
    agent: PfabAgent,
    w_explore: float,
    proposed: Dict[str, Any],
    history: List[Dict[str, Any]],
    fixed: Dict[str, Any],
    label: str,
    save_dir: str = "./plots",
) -> None:
    """3-panel acquisition landscape: performance | uncertainty | combined.

    Evaluates the model on a 2D grid over (water_ratio, print_speed) with
    design/material fixed. Marks existing experiments and the proposed next point.
    Saved as ``{label}_topology.png`` inside save_dir.

    Args:
        agent: PfabAgent with fitted prediction and calibration systems.
        w_explore: Exploration weight used for this step (for combined panel).
        proposed: Parameter dict for the proposed next experiment.
        history: List of all parameter dicts evaluated so far.
        fixed: Dict with at least 'design', 'material', 'n_layers', 'n_segments'.
        label: Short identifier (e.g. 'explore_03') for title and filename.
        save_dir: Directory to save the plot.
    """
    dm = agent.calibration_system._active_datamodule  # type: ignore[attr-defined]
    if dm is None:
        return  # model not yet fitted

    design   = str(fixed.get("design",     proposed.get("design",   "B")))
    material = str(fixed.get("material",   proposed.get("material", "clay")))
    n_layers = int(fixed.get("n_layers",   proposed.get("n_layers",   5)))
    n_segs   = int(fixed.get("n_segments", proposed.get("n_segments", 4)))

    N_W, N_S = 20, 20
    water_grid = np.linspace(0.30, 0.50, N_W)
    speed_grid = np.linspace(20.0, 60.0, N_S)
    W_mesh, S_mesh = np.meshgrid(water_grid, speed_grid)

    perf_grid  = np.zeros((N_S, N_W))
    unc_grid   = np.zeros((N_S, N_W))

    base_params = {
        "design": design, "material": material,
        "n_layers": n_layers, "n_segments": n_segs,
    }
    
    # Extract performance weights from calibration system
    perf_weights = agent.calibration_system.performance_weights
    perf_names = agent.calibration_system.perf_names_order
    total_weight = sum(perf_weights.values())

    for si in range(N_S):
        for wi in range(N_W):
            p = {**base_params,
                 "water_ratio": float(W_mesh[si, wi]),
                 "print_speed": float(S_mesh[si, wi])}
            try:
                perf_dict = agent.calibration_system.perf_fn(p)
                # Compute weighted sum using actual calibration weights
                weighted_sum = sum(
                    perf_weights.get(name, 0.0) * float(perf_dict.get(name, 0.0) or 0.0)
                    for name in perf_names
                )
                perf_grid[si, wi] = weighted_sum / total_weight
            except Exception:
                perf_grid[si, wi] = 0.0
            try:
                X_norm = dm.params_to_array(p)
                unc_grid[si, wi] = agent.pred_system.uncertainty(X_norm)
            except Exception:
                unc_grid[si, wi] = 1.0

    # Min-max normalize both surfaces so w_explore has calibrated influence
    # (matches the normalization used in the actual acquisition function)
    p_min, p_max = float(perf_grid.min()), float(perf_grid.max())
    u_min, u_max = float(unc_grid.min()), float(unc_grid.max())
    p_span = p_max - p_min if (p_max - p_min) > 1e-10 else 1.0
    u_span = u_max - u_min if (u_max - u_min) > 1e-10 else 1.0
    perf_norm = (perf_grid - p_min) / p_span
    unc_norm  = (unc_grid  - u_min) / u_span
    combined_grid = (1.0 - w_explore) * perf_norm + w_explore * unc_norm

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    phase_label = "Inference" if w_explore == 0.0 else "Exploration"
    fig.suptitle(
        f"{phase_label} Topology — {label}  ·  design={design}  material={material}  "
        f"w_explore={w_explore:.1f}",
        fontsize=11, fontweight="bold",
    )

    combined_title = "Combined (performance only)" if w_explore == 0.0 else "Combined exploration"
    panels = [
        (axes[0], perf_grid,     "Performance (predicted)",    "YlGn",   False),
        (axes[1], unc_grid,      "Uncertainty (evidence gap)", "PuBu",   False),
        (axes[2], combined_grid, combined_title,               "RdYlGn", True),
    ]

    # Split history into same design+material vs others
    same_set = [
        p for p in history
        if p.get("design") == design and p.get("material") == material
    ]
    other_set = [p for p in history if p not in same_set]

    other_w   = [p.get("water_ratio", 0.0) for p in other_set]
    other_spd = [p.get("print_speed",  0.0) for p in other_set]
    same_w    = [p.get("water_ratio", 0.0) for p in same_set]
    same_spd  = [p.get("print_speed",  0.0) for p in same_set]

    prop_w   = float(proposed.get("water_ratio", 0.0))
    prop_spd = float(proposed.get("print_speed",  0.0))

    for panel_idx, (ax, grid, title, cmap_name, mark_proposed) in enumerate(panels):
        cf = ax.contourf(W_mesh, S_mesh, grid, levels=24, cmap=cmap_name, alpha=0.92)
        ax.contour(W_mesh, S_mesh, grid, levels=6, colors="white", alpha=0.20, linewidths=0.5)
        fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.045)

        is_last = panel_idx == 2
        # Other history — small grey open circles (context)
        if other_w:
            ax.scatter(
                other_w, other_spd, s=18, c="white", edgecolors="#999999",
                linewidths=0.8, alpha=0.5, zorder=4,
                label="Evaluated (other)" if is_last else None,
            )
        # Same design+material history — filled black circles
        if same_w:
            ax.scatter(
                same_w, same_spd, s=28, c="black", edgecolors="white",
                linewidths=0.8, zorder=5,
                label="Evaluated (same combo)" if is_last else None,
            )

        if mark_proposed:
            ax.scatter([prop_w], [prop_spd], s=180, marker="x", color="#FFD700",
                       linewidths=3.0, zorder=10,
                       label=f"Proposed  (w={prop_w:.2f}, spd={prop_spd:.1f})")
            ax.legend(fontsize=7, loc="lower right", markerscale=0.5)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Water Ratio",        fontsize=9)
        ax.set_ylabel("Print Speed [mm/s]", fontsize=9)
        ax.set_xlim(0.30, 0.50)
        ax.set_ylim(20.0, 60.0)
        ax.grid(True, alpha=0.12)

    _save(os.path.join(save_dir, f"{label}_topology.png"))


# ── Phase 1: Physics topology grid ────────────────────────────────────────────

def plot_physics_topology(
    agent: PfabAgent,
    save_dir: str = "./plots/phase_1_baseline",
) -> None:
    """4×4 grid showing true physics performance landscapes for all design+material combos.

    Rows: (A,clay), (A,concrete), (B,clay), (B,concrete).
    Columns: path_accuracy, energy_efficiency, production_rate, combined (weighted).
    Uses the physics functions directly — no ML model involved.
    
    Args:
        agent: PfabAgent with configured calibration system (for performance weights).
        save_dir: Directory to save the plot.
    """
    # Extract weights from calibration system
    perf_weights = agent.calibration_system.performance_weights
    w_acc = perf_weights.get("path_accuracy", 1.0)
    w_eff = perf_weights.get("energy_efficiency", 1.0)
    w_rate = perf_weights.get("production_rate", 1.0)
    total_weight = w_acc + w_eff + w_rate
    
    # Format weights for display
    weights_str = f"({w_acc:.0f}:{w_eff:.0f}:{w_rate:.0f})"

    COMBOS = [("A", "clay"), ("A", "concrete"), ("B", "clay"), ("B", "concrete")]
    N_W, N_S = 40, 40  # increased resolution for smoother topology
    water_vals = np.linspace(0.30, 0.50, N_W)
    speed_vals = np.linspace(20.0, 60.0, N_S)
    W_mesh, S_mesh = np.meshgrid(water_vals, speed_vals)

    max_dev  = PathAccuracyModel.MAX_DEVIATION
    target_e = EnergyConsumptionModel.TARGET_ENERGY
    max_e    = EnergyConsumptionModel.MAX_ENERGY
    max_spd  = ProductionRateModel.MAX_RATE

    n_rows = len(COMBOS)
    n_cols = 4
    col_titles = ["Path Accuracy", "Energy Efficiency", "Production Rate", f"Combined {weights_str}"]
    row_labels  = [f"Design {d}\nMaterial: {m}" for d, m in COMBOS]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 13))
    fig.suptitle("Physics Performance Topology — All Design × Material Combos",
                 fontsize=14, fontweight="bold", y=0.995)

    # Use plasma colormap for individual metrics (better topology visibility)
    # and RdYlGn for combined
    individual_cmap = "plasma"
    combined_cmap = "RdYlGn"

    # Store all grids to compute global bounds per column
    all_grids: List[List[np.ndarray]] = []
    
    # Precompute grids for each combo and column
    for row_idx, (design, material) in enumerate(COMBOS):
        acc_grid  = np.zeros((N_S, N_W))
        eff_grid  = np.zeros((N_S, N_W))
        rate_grid = np.zeros((N_S, N_W))

        for si in range(N_S):
            for wi in range(N_W):
                spd = float(S_mesh[si, wi])
                w   = float(W_mesh[si, wi])

                # Path accuracy: avg deviation over 5 layers × 4 segments
                devs = [
                    path_deviation(spd, design, seg_i, w, material, li)
                    for li in range(5) for seg_i in range(4)
                ]
                avg_dev = float(np.mean(devs))
                acc_grid[si, wi] = max(0.0, 1.0 - avg_dev / max_dev)

                # Energy efficiency
                energies = [
                    energy_per_segment(spd, material, design, w, seg_i, li)
                    for li in range(5) for seg_i in range(4)
                ]
                avg_e = float(np.mean(energies))
                eff_grid[si, wi] = max(0.0, 1.0 - abs(avg_e - target_e) / max_e)

                # Production rate
                rate_grid[si, wi] = spd / max_spd

        # Use actual calibration weights for combined score
        combined_grid = (w_acc * acc_grid + w_eff * eff_grid + w_rate * rate_grid) / total_weight
        all_grids.append([acc_grid, eff_grid, rate_grid, combined_grid])
    
    # Compute global bounds for each column (same bounds vertically stacked)
    col_vmins = []
    col_vmaxs = []
    for col_idx in range(n_cols):
        col_values = np.concatenate([all_grids[row_idx][col_idx].flatten() for row_idx in range(n_rows)])
        col_vmins.append(float(np.min(col_values)))
        col_vmaxs.append(float(np.max(col_values)))
    
    # Plot all grids with consistent column bounds
    for row_idx in range(n_rows):
        grids = all_grids[row_idx]
        cmaps = [individual_cmap, individual_cmap, individual_cmap, combined_cmap]
        
        for col_idx, (grid, cmap_name) in enumerate(zip(grids, cmaps)):
            ax = axes[row_idx, col_idx]
            
            vmin = col_vmins[col_idx]
            vmax = col_vmaxs[col_idx]
            
            # Use more levels for smoother appearance
            cf = ax.contourf(W_mesh, S_mesh, grid, levels=30, cmap=cmap_name, 
                           vmin=vmin, vmax=vmax)
            
            # Add contour lines for better topology visibility
            contour_levels = 8
            ax.contour(W_mesh, S_mesh, grid, levels=contour_levels, 
                      colors="white", alpha=0.4, linewidths=0.7)
            
            # Add colorbar with better sizing
            cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046, aspect=20)
            cb.ax.tick_params(labelsize=7)

            # Titles and labels
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)
            
            # Row label (bold) separate from axis label (normal weight)
            if col_idx == 0:
                ax.set_ylabel("Print Speed [mm/s]", fontsize=8)
                ax.text(
                    -0.28, 0.5, row_labels[row_idx],
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    ha="center", va="center", rotation=90,
                )
            else:
                ax.set_ylabel("Print Speed [mm/s]", fontsize=8)
            
            ax.set_xlabel("Water Ratio", fontsize=8)
            ax.set_xlim(0.30, 0.50)
            ax.set_ylim(20.0, 60.0)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.12, linestyle=":", linewidth=0.5)

    plt.tight_layout(rect=(0.06, 0, 1, 0.985))
    _save(os.path.join(save_dir, "physics_topology.png"))


# ── Phase 1: Baseline parameter scatter ───────────────────────────────────────

def plot_baseline_scatter(
    experiments: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
    save_dir: str = "./plots/phase_1_baseline",
) -> None:
    """2D scatter: water_ratio vs print_speed, colored by design+material combination.

    LHS stratification is 4D (water_ratio, print_speed, design, material), so the
    2D projection onto continuous axes may show gaps — this is expected.
    Design is encoded by marker shape, material by color.
    """
    DESIGN_MARKERS = {"A": "o", "B": "s"}         # circle vs square
    MATERIAL_COLORS = {"clay": "#2E7D32", "concrete": "#1976D2"}

    # Organize data by combo
    COMBOS = [("A", "clay"), ("A", "concrete"), ("B", "clay"), ("B", "concrete")]
    combo_data: Dict[Tuple[str, str], List] = {c: [] for c in COMBOS}

    for code, params, perf in experiments:
        design   = str(params.get("design",   "A"))
        material = str(params.get("material", "clay"))
        combo = (design, material)
        x = float(params.get("water_ratio", 0.0))
        y = float(params.get("print_speed",  0.0))
        combo_data[combo].append((x, y, code))

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Baseline Experiments — Parameter Space",
                 fontsize=13, fontweight="bold")

    for combo in COMBOS:
        data = combo_data[combo]
        if not data:
            continue
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        labels = [d[2] for d in data]
        design, material = combo
        ax.scatter(xs, ys, s=120,
                   c=MATERIAL_COLORS[material],
                   marker=DESIGN_MARKERS[design],
                   alpha=0.80, edgecolors="white", linewidths=1.8,
                   label=f"Design {design} / {material}", zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), xytext=(4, 4), textcoords="offset points",
                       fontsize=7, alpha=0.75)

    ax.set_xlabel("Water Ratio", fontsize=11)
    ax.set_ylabel("Print Speed [mm/s]", fontsize=11)
    ax.set_xlim(0.28, 0.52)
    ax.set_ylim(18.0, 62.0)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.95,
             title="Design × Material", title_fontsize=10)

    # Add footnote about LHS stratification
    ax.text(0.99, 0.01, "LHS stratified across 4 dimensions\n(incl. design + material)",
            transform=ax.transAxes, fontsize=7, alpha=0.5,
            ha="right", va="bottom", style="italic")

    _save(os.path.join(save_dir, "baseline_scatter.png"))
