"""Per-stage plotting helpers for the extrusion printing showcase."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
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


# ── Phase 1: As-Printed vs As-Designed (3-D stacked layers) ──────────────────

def plot_path_comparison(
    exp_data: ExperimentData,
    camera: Any,
    params: Dict[str, Any],
) -> None:
    """3-D stacked-layer view: designed (grey dashed) vs as-printed (coloured by deviation)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

    N_LAYERS, N_SEGMENTS, N_PTS = 5, 4, 5
    SEG_LENGTH  = (N_PTS - 1) * 0.01   # 0.04 m per segment
    SEG_GAP     = 0.008                 # gap between segments

    # Pre-fetch all segment data and collect deviations for colormap range
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
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection="3d")

    first_d, first_m = True, True

    for li in range(N_LAYERS):
        z = float(li)
        x_off = 0.0

        all_dx, all_dy, all_mx, all_my = [], [], [], []

        for si in range(N_SEGMENTS):
            data = cache[(li, si)]
            dp   = data["designed_path"]
            mp   = data["measured_path"]

            dx = [p[0] + x_off for p in dp]
            dy = [p[1]          for p in dp]   # always 0
            mx = [p[0] + x_off for p in mp]
            my = [p[1]          for p in mp]

            all_dx.extend(dx)
            all_dy.extend(dy)
            all_mx.extend(mx)
            all_my.extend(my)

            # Connector lines coloured by local deviation magnitude
            for x_d, x_m, y_d, y_m in zip(dx, mx, dy, my):
                dev   = abs(y_m - y_d)
                color = cmap(norm(dev))
                ax.plot(  # type: ignore[union-attr]
                    [x_d, x_m], [y_d, y_m], [z, z],
                    color=color, alpha=0.55, linewidth=0.9,
                )

            x_off += SEG_LENGTH + SEG_GAP

        # Designed path — grey dashed
        kw_d: Dict[str, Any] = dict(color="#999999", linestyle="--", linewidth=1.5)
        if first_d:
            kw_d["label"] = "Designed"
            first_d = False
        ax.plot(all_dx, all_dy, [z] * len(all_dx), **kw_d)  # type: ignore[union-attr]

        # Measured path — solid, coloured by mean layer deviation
        mean_dev = float(exp_data.features.get_value("path_deviation")[li].mean())  # type: ignore[index]
        kw_m: Dict[str, Any] = dict(color=cmap(norm(mean_dev)), linewidth=2.2)
        if first_m:
            kw_m["label"] = "Measured"
            first_m = False
        ax.plot(all_mx, all_my, [z] * len(all_mx), **kw_m)  # type: ignore[union-attr]

    # Colorbar
    cb = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.55, aspect=18)
    cb.set_label("Path Deviation [m]", fontsize=9)

    ax.set_xlabel("Along-path [m]", labelpad=8)   # type: ignore[union-attr]
    ax.set_ylabel("Lateral offset [m]", labelpad=8)  # type: ignore[union-attr]
    ax.set_zticks(list(range(N_LAYERS)))           # type: ignore[union-attr]
    ax.set_zticklabels([f"L{i}" for i in range(N_LAYERS)])  # type: ignore[union-attr]

    design   = params.get("design",      "?")
    material = params.get("material",    "?")
    speed    = params.get("print_speed", 0.0)
    ax.set_title(  # type: ignore[union-attr]
        f"As-Printed vs As-Designed  ·  {exp_data.code}\n"
        f"design={design}   material={material}   speed={speed:.1f} mm/s",
        pad=14, fontsize=11,
    )
    ax.legend(loc="upper left", framealpha=0.75)  # type: ignore[union-attr]
    ax.view_init(elev=28, azim=-55)               # type: ignore[union-attr]

    _save_and_show("path_comparison")


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
        ax.set_yticks(range(n_layers))
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

    X_val = np.vstack([b[0] for b in val_batches])
    y_val = np.vstack([b[1] for b in val_batches])

    y_pred_list = [m.forward_pass(X_val) for m in pred_system.models]
    y_pred = np.hstack(y_pred_list) if y_pred_list else y_val

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
        ax.plot(lim, lim, "k--", linewidth=1.2)
        ax.set_xlabel("Actual",    fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{name}\nR² = {r2:.3f}", fontsize=10)
        ax.set_xlim(lim); ax.set_ylim(lim)

    _save_and_show("prediction_accuracy")


# ── Phase 3: Parameter space ──────────────────────────────────────────────────

def plot_parameter_space(
    all_params: List[Dict[str, Any]],
    phases: List[str],
) -> None:
    """2-D scatter of water_ratio vs print_speed, colour-coded by phase."""
    water_ratios = [p["water_ratio"] for p in all_params]
    print_speeds = [p["print_speed"] for p in all_params]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Parameter Space Exploration", fontsize=12, fontweight="bold")

    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            ax.scatter(
                [water_ratios[i] for i in idx],
                [print_speeds[i] for i in idx],
                c=color, label=phase.capitalize(), alpha=0.85, s=65, zorder=3,
            )

    ax.set_xlabel("Water Ratio",       fontsize=10)
    ax.set_ylabel("Print Speed [mm/s]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _save_and_show("parameter_space")


# ── Phase 4: Performance trajectory ──────────────────────────────────────────

def plot_performance_trajectory(
    exp_params_and_perf: List[Tuple[Dict[str, Any], Dict[str, float]]],
    phases: List[str],
) -> None:
    """Dual-line plot with shaded phase bands and per-phase labels."""
    path_acc   = [pp[1].get("path_accuracy",    float("nan")) for pp in exp_params_and_perf]
    energy_eff = [pp[1].get("energy_efficiency", float("nan")) for pp in exp_params_and_perf]
    xs = list(range(1, len(path_acc) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
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

    # Phase transition labels at top
    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            mid = (idx[0] + idx[-1]) / 2 + 1
            ax.text(mid, 1.02, phase.capitalize(), ha="center", va="bottom",
                    color=color, fontsize=8, transform=ax.get_xaxis_transform())

    ax.set_xlabel("Experiment #", fontsize=10)
    ax.set_ylabel("Score [0–1]",  fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _save_and_show("performance_trajectory")


# ── Phase 5: Online adaptation ────────────────────────────────────────────────

def plot_adaptation(
    layer_speeds: List[float],
    deviations: List[float],
) -> None:
    """Dual-axis: print_speed per layer and path_deviation with initial-deviation reference."""
    layers = list(range(len(layer_speeds)))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    fig.suptitle("Online Adaptation — Layer-by-Layer", fontsize=12, fontweight="bold")

    ax2 = ax1.twinx()

    ax1.plot(layers, layer_speeds, marker="o", color=_PHASE_COLORS["baseline"],
             linewidth=1.8, markersize=6, label="Print Speed")
    ax1.set_xlabel("Layer",                     fontsize=10)
    ax1.set_ylabel("Print Speed [mm/s]",        fontsize=10, color=_PHASE_COLORS["baseline"])
    ax1.tick_params(axis="y",                   labelcolor=_PHASE_COLORS["baseline"])

    ax2.plot(layers, deviations, marker="s", color=_PHASE_COLORS["exploration"],
             linestyle="--", linewidth=1.8, markersize=6, label="Path Deviation")
    ax2.axhline(deviations[0], color="grey", linestyle=":", linewidth=1.0,
                alpha=0.7, label="Initial dev")
    ax2.set_ylabel("Avg Path Deviation [m]",    fontsize=10, color=_PHASE_COLORS["exploration"])
    ax2.tick_params(axis="y",                   labelcolor=_PHASE_COLORS["exploration"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.25)
    _save_and_show("adaptation")
