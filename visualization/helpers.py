"""Shared helpers for all visualization modules."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sensors.physics import (
    path_deviation, energy_per_segment, production_rate,
    N_LAYERS, N_SEGMENTS,
)
from models.evaluation_models import PathAccuracy, EnergyEfficiency, ProductionRate

def save_fig(path: str) -> None:
    """Save current figure and close."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def physics_combined_at(
    w: float, spd: float,
    perf_weights: dict[str, float] | None = None,
    n_layers: int = N_LAYERS,
) -> float:
    """Compute combined physics performance score at a single (water, speed) point."""
    pw = perf_weights or {}
    max_dev = PathAccuracy.MAX_DEVIATION
    target_e = EnergyEfficiency.TARGET_ENERGY
    max_e = EnergyEfficiency.MAX_ENERGY
    max_rate = ProductionRate.MAX_RATE

    devs = [path_deviation(spd, s, w, li) for li in range(n_layers) for s in range(N_SEGMENTS)]
    pa = max(0.0, 1.0 - np.mean(devs) / max_dev)

    energies = [energy_per_segment(spd, w, s, li) for li in range(n_layers) for s in range(N_SEGMENTS)]
    ee = max(0.0, 1.0 - abs(np.mean(energies) - target_e) / max_e)

    pr = production_rate(spd, w) / max_rate

    w_pa = pw.get("path_accuracy", 1)
    w_ee = pw.get("energy_efficiency", 1)
    w_pr = pw.get("production_rate", 1)
    return (w_pa * pa + w_ee * ee + w_pr * pr) / (w_pa + w_ee + w_pr)


def evaluate_physics_grid(
    resolution: int = 50,
    perf_weights: dict[str, float] | None = None,
    n_layers: int = N_LAYERS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate all performance metrics on a 2D grid.

    Returns (waters, speeds, metrics_dict).
    """
    pw = perf_weights or {}
    max_dev = PathAccuracy.MAX_DEVIATION
    target_e = EnergyEfficiency.TARGET_ENERGY
    max_e = EnergyEfficiency.MAX_ENERGY
    max_rate = ProductionRate.MAX_RATE

    waters = np.linspace(0.30, 0.50, resolution)
    speeds = np.linspace(20.0, 60.0, resolution)

    path_acc = np.zeros((resolution, resolution))
    energy_eff = np.zeros((resolution, resolution))
    prod_rate = np.zeros((resolution, resolution))

    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            devs = [path_deviation(spd, s, w, li) for li in range(n_layers) for s in range(N_SEGMENTS)]
            path_acc[j, i] = max(0.0, 1.0 - np.mean(devs) / max_dev)
            energies = [energy_per_segment(spd, w, s, li) for li in range(n_layers) for s in range(N_SEGMENTS)]
            energy_eff[j, i] = max(0.0, 1.0 - abs(np.mean(energies) - target_e) / max_e)
            prod_rate[j, i] = production_rate(spd, w) / max_rate

    w_pa = pw.get("path_accuracy", 1)
    w_ee = pw.get("energy_efficiency", 1)
    w_pr = pw.get("production_rate", 1)
    combined = (w_pa * path_acc + w_ee * energy_eff + w_pr * prod_rate) / (w_pa + w_ee + w_pr)

    label = f"Combined ({int(w_pa)}:{int(w_ee)}:{int(w_pr)})"
    return waters, speeds, {
        "Path Accuracy": path_acc,
        "Energy Efficiency": energy_eff,
        "Production Rate": prod_rate,
        label: combined,
    }
