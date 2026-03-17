"""Per-stage plotting helpers for the extrusion printing showcase."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pred_fab.core import ExperimentData, DataModule
from pred_fab import PfabAgent

os.makedirs("./plots", exist_ok=True)

_PHASE_COLORS = {"baseline": "#4C72B0", "exploration": "#DD8452", "inference": "#55A868"}


def _save_and_show(name: str) -> None:
    plt.tight_layout()
    plt.savefig(f"./plots/{name}.png", dpi=100)
    plt.show(block=False)
    plt.close()


def plot_feature_heatmaps(exp_data: ExperimentData) -> None:
    """2×(5×4) heatmaps of path_deviation and energy_per_segment for one experiment."""
    n_layers, n_segments = 5, 4

    def _to_grid(feature_name: str) -> np.ndarray:
        # Features are stored as (n_layers, n_segments) tensors directly
        return exp_data.features.get_value(feature_name)  # type: ignore[return-value]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Feature Heatmaps — {exp_data.code}")

    for ax, fname, label in zip(
        axes,
        ["path_deviation", "energy_per_segment"],
        ["Path Deviation [m]", "Energy per Segment [J]"],
    ):
        grid = _to_grid(fname)
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Layer")
        ax.set_title(label)
        ax.set_xticks(range(n_segments))
        ax.set_yticks(range(n_layers))
        plt.colorbar(im, ax=ax)

    _save_and_show("feature_heatmaps")


def plot_prediction_accuracy(agent: PfabAgent, datamodule: DataModule) -> None:
    """Scatter of predicted vs actual for both output features with R² annotation."""
    from sklearn.metrics import r2_score
    from pred_fab.utils import SplitType  # type: ignore[attr-defined]

    pred_system = agent.pred_system
    outputs = pred_system.get_system_outputs()

    val_batches = datamodule.get_batches(SplitType.VAL)
    if not val_batches:
        print("[plot_prediction_accuracy] No validation batches available.")
        return

    X_val = np.vstack([b[0] for b in val_batches])
    y_val = np.vstack([b[1] for b in val_batches])

    y_pred_list = []
    for model in pred_system.models:
        y_pred_list.append(model.forward_pass(X_val))
    y_pred = np.hstack(y_pred_list) if y_pred_list else y_val

    n_out = min(y_val.shape[1], len(outputs))
    fig, axes = plt.subplots(1, n_out, figsize=(6 * n_out, 5))
    if n_out == 1:
        axes = [axes]  # type: ignore[assignment]

    for i, (ax, name) in enumerate(zip(axes, outputs)):
        if i >= y_pred.shape[1] or i >= y_val.shape[1]:
            break
        r2 = r2_score(y_val[:, i], y_pred[:, i])
        ax.scatter(y_val[:, i], y_pred[:, i], alpha=0.7, color=_PHASE_COLORS["exploration"])
        lims = [min(y_val[:, i].min(), y_pred[:, i].min()),
                max(y_val[:, i].max(), y_pred[:, i].max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR² = {r2:.3f}")

    fig.suptitle("Prediction Accuracy (validation set)")
    _save_and_show("prediction_accuracy")


def plot_parameter_space(
    all_params: List[Dict[str, Any]],
    phases: List[str],
) -> None:
    """2D scatter of layer_time vs print_speed, color-coded by phase."""
    layer_times = [p["layer_time"] for p in all_params]
    print_speeds = [p["print_speed"] for p in all_params]
    colors = [_PHASE_COLORS.get(ph, "gray") for ph in phases]

    fig, ax = plt.subplots(figsize=(8, 5))
    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            ax.scatter(
                [layer_times[i] for i in idx],
                [print_speeds[i] for i in idx],
                c=color, label=phase.capitalize(), alpha=0.85, s=60,
            )
    ax.set_xlabel("Layer Time [s]")
    ax.set_ylabel("Print Speed [mm/s]")
    ax.set_title("Parameter Space Exploration")
    ax.legend()
    _save_and_show("parameter_space")


def plot_performance_trajectory(
    exp_params_and_perf: List[Tuple[Dict[str, Any], Dict[str, float]]],
) -> None:
    """Dual-line plot of path_accuracy and energy_efficiency over experiment index."""
    path_acc = [pp[1].get("path_accuracy", float("nan")) for pp in exp_params_and_perf]
    energy_eff = [pp[1].get("energy_efficiency", float("nan")) for pp in exp_params_and_perf]
    xs = list(range(1, len(path_acc) + 1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, path_acc, marker="o", color=_PHASE_COLORS["exploration"], label="Path Accuracy")
    ax.plot(xs, energy_eff, marker="s", color=_PHASE_COLORS["inference"], label="Energy Efficiency")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Score [0–1]")
    ax.set_title("Performance Trajectory")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_and_show("performance_trajectory")


def plot_adaptation(
    layer_speeds: List[float],
    deviations: List[float],
) -> None:
    """Dual-axis plot of print_speed per layer and resulting path_deviation improvement."""
    layers = list(range(len(layer_speeds)))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(layers, layer_speeds, marker="o", color=_PHASE_COLORS["baseline"], label="Print Speed")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Print Speed [mm/s]", color=_PHASE_COLORS["baseline"])
    ax1.tick_params(axis="y", labelcolor=_PHASE_COLORS["baseline"])

    ax2.plot(layers, deviations, marker="s", color=_PHASE_COLORS["exploration"],
             linestyle="--", label="Path Deviation")
    ax2.set_ylabel("Avg Path Deviation [m]", color=_PHASE_COLORS["exploration"])
    ax2.tick_params(axis="y", labelcolor=_PHASE_COLORS["exploration"])

    fig.suptitle("Online Adaptation — Layer-by-Layer")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    _save_and_show("adaptation")
