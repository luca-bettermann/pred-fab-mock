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
) -> None:
    """1x4 heatmap: individual panels show star at own optimum, combined panel
    shows small dots for each metric's optimum plus a star at the combined optimum."""
    waters, speeds, metrics = evaluate_physics_grid(resolution, perf_weights)
    metric_names = list(metrics.keys())

    # Pre-compute each metric's optimum location
    optima = {}
    for title, data in metrics.items():
        best_idx = np.unravel_index(np.argmax(data), data.shape)
        optima[title] = (waters[best_idx[1]], speeds[best_idx[0]])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Physics Performance Topology", fontsize=14, fontweight="bold", y=1.02)

    for ax, (title, data) in zip(axes, metrics.items()):
        im = ax.contourf(waters, speeds, data, levels=20, cmap="RdYlGn")
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
        else:
            # Individual panels: star at this metric's own optimum
            ow, os_ = optima[title]
            ax.plot(ow, os_, "*", color="white", ms=14,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=8)

        ax.set_title(title, fontsize=10)
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


def plot_baseline_scatter(
    save_path: str,
    params_list: list[dict[str, Any]],
) -> None:
    """Scatter of baseline experiments + nearest-neighbor distance histogram."""
    waters = np.array([p["water_ratio"] for p in params_list])
    speeds = np.array([p["print_speed"] for p in params_list])
    n = len(waters)

    normed = np.column_stack([(waters - 0.30) / 0.20, (speeds - 20.0) / 40.0])
    nn_dists = np.array([
        np.min(np.delete(np.linalg.norm(normed - normed[i], axis=1), i))
        for i in range(n)
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline Sampling Coverage (\u03ba=1)", fontsize=13, fontweight="bold")

    ax1.scatter(waters, speeds, s=60, c="#4878CF", edgecolors="white", linewidth=0.8, zorder=5)
    for i, (w, s) in enumerate(zip(waters, speeds)):
        ax1.annotate(f"{i+1}", (w, s), fontsize=6, ha="center", va="bottom",
                     xytext=(0, 5), textcoords="offset points", color="#666")
    ax1.set_xlim(0.30, 0.50)
    ax1.set_ylim(20.0, 60.0)
    ax1.set_xlabel("Water Ratio")
    ax1.set_ylabel("Print Speed [mm/s]")
    ax1.set_title(f"Parameter Space ({n} experiments)")
    ax1.grid(True, alpha=0.2)

    ax2.bar(range(1, n + 1), nn_dists, color="#4878CF", edgecolor="white", linewidth=0.5)
    ax2.axhline(nn_dists.mean(), color="#D65F5F", ls="--", lw=1.5, label=f"Mean={nn_dists.mean():.3f}")
    ax2.set_xlabel("Experiment #")
    ax2.set_ylabel("NN Distance (normalized)")
    ax2.set_title("Spacing Uniformity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)
