"""Shared helpers for all visualization modules."""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from pred_fab.plotting import save_fig
from pred_fab.utils.metrics import combined_score

from sensors.physics import (
    path_deviation, energy_per_segment, production_rate,
    N_LAYERS, N_SEGMENTS,
)
from models.evaluation_models import PathAccuracy, EnergyEfficiency, ProductionRate
from schema import WATER_RATIO_BOUNDS, PRINT_SPEED_BOUNDS, DEFAULT_PERF_WEIGHTS

__all__ = [
    "save_fig", "physics_combined_at", "evaluate_physics_grid", "get_physics_optimum",
]


def _physics_metrics_at(w: float, spd: float, n_layers: int) -> tuple[float, float, float]:
    """Ground-truth (path_accuracy, energy_efficiency, production_rate) scores at one point.

    Aggregation matches the evaluation pipeline (_LinearTargetScore): per-segment
    scores are clipped to [0, 1] first, then averaged.
    """
    pa_scores = [
        1.0 - path_deviation(spd, s, w, li) / PathAccuracy.MAX_DEVIATION
        for li in range(n_layers) for s in range(N_SEGMENTS)
    ]
    pa = float(np.mean(np.clip(pa_scores, 0.0, 1.0)))

    ee_scores = [
        1.0 - abs(energy_per_segment(spd, w, s, li) - EnergyEfficiency.TARGET_ENERGY) / EnergyEfficiency.MAX_ENERGY
        for li in range(n_layers) for s in range(N_SEGMENTS)
    ]
    ee = float(np.mean(np.clip(ee_scores, 0.0, 1.0)))

    pr = float(np.clip(production_rate(spd, w) / ProductionRate.MAX_RATE, 0.0, 1.0))
    return pa, ee, pr


def _full_weights(perf_weights: dict[str, float] | None) -> dict[str, float]:
    """Fill missing attribute weights with the all-1.0 default."""
    return {**DEFAULT_PERF_WEIGHTS, **(perf_weights or {})}


def physics_combined_at(
    w: float, spd: float,
    perf_weights: dict[str, float] | None = None,
    n_layers: int = N_LAYERS,
) -> float:
    """Compute combined physics performance score at a single (water, speed) point."""
    pa, ee, pr = _physics_metrics_at(w, spd, n_layers)
    perf = {"path_accuracy": pa, "energy_efficiency": ee, "production_rate": pr}
    return float(combined_score(perf, _full_weights(perf_weights)))


def evaluate_physics_grid(
    resolution: int = 50,
    perf_weights: dict[str, float] | None = None,
    n_layers: int = N_LAYERS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate all performance metrics on a 2D grid.

    Returns (waters, speeds, metrics_dict); the combined score is the last entry.
    """
    weights = _full_weights(perf_weights)

    waters = np.linspace(*WATER_RATIO_BOUNDS, resolution)
    speeds = np.linspace(*PRINT_SPEED_BOUNDS, resolution)

    path_acc = np.zeros((resolution, resolution))
    energy_eff = np.zeros((resolution, resolution))
    prod_rate = np.zeros((resolution, resolution))
    combined = np.zeros((resolution, resolution))

    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            pa, ee, pr = _physics_metrics_at(w, spd, n_layers)
            path_acc[j, i], energy_eff[j, i], prod_rate[j, i] = pa, ee, pr
            combined[j, i] = combined_score(
                {"path_accuracy": pa, "energy_efficiency": ee, "production_rate": pr}, weights
            )

    label = (f"Combined ({int(weights['path_accuracy'])}:"
             f"{int(weights['energy_efficiency'])}:{int(weights['production_rate'])})")
    return waters, speeds, {
        "Path Accuracy": path_acc,
        "Energy Efficiency": energy_eff,
        "Production Rate": prod_rate,
        label: combined,
    }


def get_physics_optimum(
    perf_weights: dict[str, float] | None = None,
    n_layers: int = N_LAYERS,
    resolution: int = 50,
) -> tuple[float, float]:
    """Grid-argmax of the combined ground-truth score; returns (water_ratio, print_speed).

    Reads the physics constants live (honours per-session randomization) and
    weighs metrics like the evaluation pipeline.
    """
    waters, speeds, metrics = evaluate_physics_grid(resolution, perf_weights, n_layers=n_layers)
    combined = next(v for k, v in metrics.items() if k.startswith("Combined"))
    opt_idx = np.unravel_index(np.argmax(combined), combined.shape)
    return float(waters[opt_idx[1]]), float(speeds[opt_idx[0]])
