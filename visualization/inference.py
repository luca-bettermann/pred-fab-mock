"""Inference convergence and trajectory plots."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .helpers import save_fig, evaluate_physics_grid, physics_combined_at


def plot_inference_result(
    save_path: str,
    waters: np.ndarray,
    speeds: np.ndarray,
    pred_grid: np.ndarray,
    proposed_water: float,
    proposed_speed: float,
    proposed_score: float,
    opt_water: float,
    opt_speed: float,
    opt_score: float,
    experiment_pts: list[dict[str, Any]] | None = None,
) -> None:
    """Single-shot inference result on the predicted performance topology."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    fig.suptitle("Inference Result", fontsize=13, fontweight="bold")

    im = ax.contourf(waters, speeds, pred_grid, levels=20, cmap="RdYlGn")
    ax.contour(waters, speeds, pred_grid, levels=10, colors="white", linewidths=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Predicted Combined Score")

    # Training experiments
    if experiment_pts:
        ew = [p["water_ratio"] for p in experiment_pts]
        es = [p["print_speed"] for p in experiment_pts]
        ax.scatter(ew, es, s=15, c="white", edgecolors="black", linewidth=0.4, zorder=4, alpha=0.6)

    # Physics optimum (star)
    ax.plot(opt_water, opt_speed, "*", color="white", ms=16,
            markeredgecolor="black", markeredgewidth=1, zorder=8,
            label=f"Physics optimum ({opt_score:.3f})")

    # Proposed point (cross)
    ax.plot(proposed_water, proposed_speed, "x", color="#EAB308", ms=14,
            markeredgewidth=2.5, zorder=9,
            label=f"Proposed ({proposed_score:.3f})")

    ax.set_xlabel("Water Ratio")
    ax.set_ylabel("Print Speed [mm/s]")
    ax.legend(fontsize=8, loc="upper left")

    save_fig(save_path)


def plot_inference_convergence(
    save_path: str,
    infer_log: list[dict[str, Any]],
    opt_speed: float,
    opt_water: float,
    resolution: int = 40,
    perf_weights: dict[str, float] | None = None,
    n_layers: int | None = None,
) -> None:
    """Convergence trajectory on the performance topology + score per round."""
    from sensors.physics import N_LAYERS as _DEFAULT_LAYERS
    nl = n_layers or _DEFAULT_LAYERS
    waters, speeds, metrics = evaluate_physics_grid(resolution, perf_weights, n_layers=nl)
    combined = list(metrics.values())[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Inference Convergence", fontsize=13, fontweight="bold")

    im = ax1.contourf(waters, speeds, combined, levels=20, cmap="RdYlGn")
    ax1.contour(waters, speeds, combined, levels=10, colors="white", linewidths=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Combined Score")

    iw = [r["water"] for r in infer_log]
    is_ = [r["speed"] for r in infer_log]
    if len(infer_log) > 1:
        ax1.plot(iw, is_, "w-", lw=2, alpha=0.8)
    for j, r in enumerate(infer_log):
        ax1.plot(r["water"], r["speed"], "wo", ms=8, markeredgecolor="black", markeredgewidth=0.8)
        ax1.annotate(f"{j+1}", (r["water"], r["speed"]), fontsize=7, ha="center",
                     va="center", color="black", fontweight="bold")

    ax1.plot(opt_water, opt_speed, "w*", ms=14, markeredgecolor="black", markeredgewidth=1,
             label=f"Optimum ({opt_water:.2f}, {opt_speed:.0f})")
    ax1.set_xlabel("Water Ratio")
    ax1.set_ylabel("Print Speed [mm/s]")
    ax1.set_title("Trajectory on Performance Topology")
    ax1.legend(fontsize=7, loc="upper left")

    scores = [r["score"] for r in infer_log]
    ax2.plot(range(1, len(scores) + 1), scores, "o-", color="#4878CF", lw=2, ms=8)
    ax2.axhline(physics_combined_at(opt_water, opt_speed, perf_weights), color="#6ACC65",
                ls="--", lw=1.5, label="Physics optimum score")
    ax2.set_xlabel("Inference Round")
    ax2.set_ylabel("Combined Score")
    ax2.set_title("Performance per Round")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1)

    save_fig(save_path)
