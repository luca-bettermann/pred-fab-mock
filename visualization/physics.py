"""Physics topology, cross-section, and baseline coverage plots."""

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

from .helpers import save_fig, evaluate_physics_grid


def plot_physics_topology(
    save_path: str,
    resolution: int = 50,
    perf_weights: dict[str, float] | None = None,
    n_layers: int | None = None,
) -> None:
    """1x4 heatmap: individual panels show star at own optimum, combined panel
    shows small dots for each metric's optimum plus a star at the combined optimum."""
    from sensors.physics import N_LAYERS as _DEFAULT_LAYERS
    nl = n_layers or _DEFAULT_LAYERS
    waters, speeds, metrics = evaluate_physics_grid(resolution, perf_weights, n_layers=nl)
    metric_names = list(metrics.keys())

    # Pre-compute each metric's optimum location
    optima = {}
    for title, data in metrics.items():
        best_idx = np.unravel_index(np.argmax(data), data.shape)
        optima[title] = (waters[best_idx[1]], speeds[best_idx[0]])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Physics Performance Topology", fontsize=14, fontweight="bold", y=1.02)

    # Map metric names to their weights for display
    pw = perf_weights or {}
    weight_keys = {
        "Path Accuracy": "path_accuracy",
        "Energy Efficiency": "energy_efficiency",
        "Production Rate": "production_rate",
    }

    for ax, (title, data) in zip(axes, metrics.items()):
        # Individual metrics use YlGn (performance); combined uses RdYlGn (objective)
        cmap = "RdYlGn" if "Combined" in title else "YlGn"
        im = ax.contourf(waters, speeds, data, levels=20, cmap=cmap)
        ax.contour(waters, speeds, data, levels=10, colors="white", linewidths=0.3, alpha=0.5)

        if "Combined" in title:
            # Combined panel: small dots for each individual metric's optimum
            for m_name in metric_names[:-1]:  # skip combined itself
                ow, os_ = optima[m_name]
                ax.plot(ow, os_, "o", color="white", ms=6,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=8)
            # Star at the combined optimum
            cw, cs = optima[title]
            ax.plot(cw, cs, "*", color="white", ms=16,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=9)
            label = title
        else:
            # Individual panels: star at this metric's own optimum
            ow, os_ = optima[title]
            ax.plot(ow, os_, "*", color="white", ms=14,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=8)
            wk = weight_keys.get(title)
            w_val = pw.get(wk, 1) if wk else None
            label = f"{title} (w={w_val:g})" if w_val is not None else title

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Water Ratio")
        ax.set_ylabel("Print Speed [mm/s]")
        plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)


def plot_cross_sections(
    save_path: str,
    opt_speed: float,
    opt_water: float,
    resolution: int = 50,
    perf_weights: dict[str, float] | None = None,
) -> None:
    """1D cross-sections through the physics optimum."""
    waters, speeds, metrics = evaluate_physics_grid(resolution, perf_weights)
    w_idx = np.argmin(np.abs(waters - opt_water))
    s_idx = np.argmin(np.abs(speeds - opt_speed))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Cross-Sections Through Physics Optimum", fontsize=13, fontweight="bold")

    for name, data in metrics.items():
        ax1.plot(speeds, data[:, w_idx], label=name, lw=2)
        ax2.plot(waters, data[s_idx, :], label=name, lw=2)

    ax1.axvline(opt_speed, color="gray", ls="--", lw=1, label=f"Optimum ({opt_speed:.1f})")
    ax1.set_xlabel("Print Speed [mm/s]")
    ax1.set_ylabel("Score [0-1]")
    ax1.set_title(f"Water = {opt_water:.2f} (fixed)")
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.2)

    ax2.axvline(opt_water, color="gray", ls="--", lw=1, label=f"Optimum ({opt_water:.2f})")
    ax2.set_xlabel("Water Ratio")
    ax2.set_ylabel("Score [0-1]")
    ax2.set_title(f"Speed = {opt_speed:.1f} mm/s (fixed)")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_baseline_overview(
    save_path: str,
    params_list: list[dict[str, Any]],
    waters_grid: np.ndarray,
    speeds_grid: np.ndarray,
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    n_baseline: int,
) -> None:
    """1x3 overview: parameter space scatter, ground truth topology, initial model topology."""
    exp_waters = np.array([p["water_ratio"] for p in params_list])
    exp_speeds = np.array([p["print_speed"] for p in params_list])
    n = len(exp_waters)

    # Use ground-truth range so physics topology always renders correctly;
    # initial model is shown on the same scale for honest comparison.
    vmin, vmax = true_grid.min(), true_grid.max()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle(f"Baseline ({n_baseline} experiments)", fontsize=14, fontweight="bold", y=1.02)

    # Panel 1: parameter space
    ax1.scatter(exp_waters, exp_speeds, s=60, c="#4A7FA5", edgecolors="white", linewidth=0.8, zorder=5)
    for i, (w, s) in enumerate(zip(exp_waters, exp_speeds)):
        ax1.annotate(f"{i+1}", (w, s), fontsize=6, ha="center", va="bottom",
                     xytext=(0, 5), textcoords="offset points", color="#666")
    ax1.set_xlim(0.30, 0.50)
    ax1.set_ylim(20.0, 60.0)
    ax1.set_xlabel("Water Ratio")
    ax1.set_ylabel("Print Speed [mm/s]")
    ax1.set_title("Parameter Space", fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: ground truth
    im2 = ax2.contourf(waters_grid, speeds_grid, true_grid, levels=20, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax2.contour(waters_grid, speeds_grid, true_grid, levels=10, colors="white", linewidths=0.3, alpha=0.5)
    ax2.scatter(exp_waters, exp_speeds, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    ax2.set_xlabel("Water Ratio")
    ax2.set_ylabel("Print Speed [mm/s]")
    ax2.set_title("Ground Truth", fontsize=10)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Panel 3: initial model
    im3 = ax3.contourf(waters_grid, speeds_grid, pred_grid, levels=20, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax3.contour(waters_grid, speeds_grid, pred_grid, levels=10, colors="white", linewidths=0.3, alpha=0.5)
    ax3.scatter(exp_waters, exp_speeds, s=20, c="white", edgecolors="black", linewidth=0.5, zorder=5)
    ax3.set_xlabel("Water Ratio")
    ax3.set_ylabel("Print Speed [mm/s]")
    ax3.set_title("Initial Model", fontsize=10)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    save_fig(save_path)
