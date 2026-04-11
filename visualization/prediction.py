"""Prediction model quality plots."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .helpers import save_fig


def plot_prediction_scatter(
    save_path: str,
    model_results: dict[str, dict[str, Any]],
    title: str = "Prediction Accuracy",
) -> None:
    """Scatter of predicted vs actual for each feature.

    model_results: {feat: {"r2": float, "y_true": ndarray, "y_pred": ndarray}}
    """
    features = list(model_results.keys())
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, feat in zip(axes, features):
        r = model_results[feat]
        ax.scatter(r["y_true"], r["y_pred"], s=20, alpha=0.6,
                   c="#DD8452", edgecolors="white", linewidth=0.3)
        lims = [min(r["y_true"].min(), r["y_pred"].min()),
                max(r["y_true"].max(), r["y_pred"].max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{feat}\nR² = {r['r2']:.3f}", fontsize=10)
        ax.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_topology_comparison(
    save_path: str,
    waters: np.ndarray,
    speeds: np.ndarray,
    grids: dict[str, np.ndarray],
    title: str = "Topology Comparison",
) -> None:
    """Side-by-side contour plots for comparing topologies."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    vmin, vmax = all_vals.min(), all_vals.max()

    for ax, (label, data) in zip(axes, grids.items()):
        im = ax.contourf(waters, speeds, data, levels=20, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.contour(waters, speeds, data, levels=10, colors="white", linewidths=0.3, alpha=0.5)
        ax.set_xlabel("Water Ratio")
        ax.set_ylabel("Print Speed [mm/s]")
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)
