"""Trajectory and adaptation plots."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from sensors.physics import N_LAYERS
from .helpers import save_fig


def plot_trajectory_comparison(
    save_path: str,
    fixed_scores: list[float],
    traj_scores: list[float],
    traj_schedules: list[dict[str, list[float]]],
    n_layers: int = N_LAYERS,
) -> None:
    """Fixed vs trajectory scores + per-layer speed schedules."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fixed vs Trajectory Exploration", fontsize=13, fontweight="bold")

    n_fixed = len(fixed_scores)
    n_traj = len(traj_scores)
    ax1.bar(range(1, n_fixed + 1), fixed_scores, color="#DD8452", label="Fixed params", alpha=0.8)
    ax1.bar(range(n_fixed + 1, n_fixed + n_traj + 1), traj_scores,
            color="#4878CF", label="Trajectory", alpha=0.8)
    ax1.set_xlabel("Exploration Round")
    ax1.set_ylabel("Combined Score")
    ax1.set_title("Performance per Round")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    for i, sched in enumerate(traj_schedules):
        if "print_speed" in sched:
            ax2.plot(range(n_layers), sched["print_speed"],
                     "o-", label=f"traj_{i+1:02d}", lw=1.5, ms=5)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Print Speed [mm/s]")
    ax2.set_title("Per-Layer Speed Schedules")
    if any("print_speed" in s for s in traj_schedules):
        ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)


def plot_adaptation(
    save_path: str,
    speeds: list[float],
    deviations: list[float],
    counterfactual: list[float] | None = None,
) -> None:
    """Layer-by-layer speed adaptation with optional counterfactual comparison."""
    n = len(speeds)
    layers = [f"L{i}" for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Online Adaptation — Layer-by-Layer", fontsize=13, fontweight="bold")

    ax1.plot(layers, speeds, "o-", color="#4878CF", lw=2, ms=6)
    ax1.set_ylabel("Print Speed [mm/s]")
    ax1.set_title("Print Speed (adapted)")
    ax1.grid(True, alpha=0.2)

    ax2.plot(layers, deviations, "o-", color="#D65F5F", lw=2, ms=6, label="Adapted")
    if counterfactual:
        ax2.plot(layers, counterfactual, "o--", color="#D65F5F", lw=1.5, ms=5,
                 alpha=0.5, label="No adaptation")
        ax2.fill_between(range(n), deviations, counterfactual,
                         alpha=0.15, color="#D65F5F", label="Deviation saved")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Avg Path Deviation [m]")
    ax2.set_title("Deviation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    save_fig(save_path)
