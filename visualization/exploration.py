"""Exploration, uncertainty, and optimizer comparison plots."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .helpers import save_fig


def plot_uncertainty(
    save_path: str,
    waters: np.ndarray,
    speeds: np.ndarray,
    unc_grid: np.ndarray,
    bf_grid: np.ndarray,
    baseline_params: list[dict[str, Any]] | None = None,
    title: str = "KDE Uncertainty",
) -> None:
    """3-panel: raw uncertainty | boundary factor | buffered uncertainty."""
    unc_buffered = unc_grid * bf_grid

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, data, subtitle in [
        (axes[0], unc_grid, "Raw Uncertainty"),
        (axes[1], bf_grid, "Boundary Factor"),
        (axes[2], unc_buffered, "Uncertainty x Boundary"),
    ]:
        im = ax.contourf(waters, speeds, data, levels=20, cmap="Blues")
        if baseline_params:
            bw = [p["water_ratio"] for p in baseline_params]
            bs = [p["print_speed"] for p in baseline_params]
            ax.scatter(bw, bs, s=25, c="white", edgecolors="black", linewidth=0.5,
                       zorder=5, label="Baseline")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Water Ratio")
        ax.set_ylabel("Print Speed [mm/s]")
        ax.set_title(subtitle, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)


def plot_uncertainty_cross_sections(
    save_path: str,
    waters: np.ndarray,
    speeds: np.ndarray,
    unc_grid: np.ndarray,
    unc_buffered: np.ndarray,
    water_slices: list[float] | None = None,
) -> None:
    """1D cross-sections of uncertainty at fixed water ratios."""
    if water_slices is None:
        water_slices = [0.32, 0.38, 0.42, 0.48]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Uncertainty Cross-Sections", fontsize=13, fontweight="bold")

    for w_val in water_slices:
        w_idx = np.argmin(np.abs(waters - w_val))
        ax1.plot(speeds, unc_grid[:, w_idx], label=f"w={w_val:.2f}", lw=1.5)
        ax2.plot(speeds, unc_buffered[:, w_idx], label=f"w={w_val:.2f}", lw=1.5)

    ax1.set_xlabel("Print Speed [mm/s]")
    ax1.set_ylabel("Uncertainty")
    ax1.set_title("Raw Uncertainty")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel("Print Speed [mm/s]")
    ax2.set_ylabel("Uncertainty x Boundary")
    ax2.set_title("Buffered Uncertainty")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_acquisition_topology(
    save_path: str,
    waters: np.ndarray,
    speeds: np.ndarray,
    perf_grid: np.ndarray,
    unc_grid: np.ndarray,
    combined_grid: np.ndarray,
    experiment_pts: list[dict[str, Any]] | None = None,
    proposed: dict[str, float] | None = None,
    title: str = "Acquisition Topology",
) -> None:
    """3-panel: performance | uncertainty | combined acquisition."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, data, subtitle, cmap in [
        (axes[0], perf_grid, "Performance", "YlGn"),
        (axes[1], unc_grid, "Uncertainty", "Blues"),
        (axes[2], combined_grid, "Combined", "RdYlGn"),
    ]:
        im = ax.contourf(waters, speeds, data, levels=20, cmap=cmap)
        ax.set_xlabel("Water Ratio")
        ax.set_ylabel("Print Speed [mm/s]")
        ax.set_title(subtitle, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Data points on all panels
        if experiment_pts:
            ew = [p["water_ratio"] for p in experiment_pts]
            es = [p["print_speed"] for p in experiment_pts]
            ax.scatter(ew, es, s=18, c="white", edgecolors="#3F3F46",
                       linewidth=0.5, zorder=5, label="Evaluated")

    # Proposed point only on combined panel
    if proposed is not None:
        axes[2].plot(proposed["water_ratio"], proposed["print_speed"],
                     "x", color="#EAB308", ms=10,
                     markeredgewidth=2, zorder=8, label="Proposed")

    # Legend on combined panel only (rightmost)
    axes[2].legend(fontsize=7, loc="upper left", framealpha=0.8)

    save_fig(save_path)


def plot_optimizer_comparison(
    save_path: str,
    results: dict[str, list[dict[str, Any]]],
    baseline_pts: dict[str, list[dict[str, Any]]],
    title: str = "Optimizer Comparison",
) -> None:
    """Side-by-side scatter of optimizer proposals."""
    tags = list(results.keys())
    n = len(tags)
    colors = ["#DD8452", "#D65F5F", "#4878CF", "#55A868"]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, tag, color in zip(axes, tags, colors):
        rounds = results[tag]
        bp = baseline_pts[tag]
        bw = [p["water_ratio"] for p in bp]
        bs = [p["print_speed"] for p in bp]
        ax.scatter(bw, bs, s=40, c="#cccccc", edgecolors="gray", linewidth=0.5,
                   zorder=3, label="Baseline")
        for i, r in enumerate(rounds):
            ax.scatter(r["water"], r["speed"], s=60, c=color,
                       edgecolors="white", linewidth=0.8, zorder=5)
            ax.annotate(f"{i+1}", (r["water"], r["speed"]), fontsize=7,
                        ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
        total_nfev = sum(r["nfev"] for r in rounds)
        ax.set_title(f"{tag}\n{total_nfev} total evals", fontsize=10)
        ax.set_xlabel("Water Ratio")
        ax.set_ylabel("Print Speed [mm/s]")
        ax.set_xlim(0.29, 0.51)
        ax.set_ylim(19, 61)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    save_fig(save_path)
