"""Ground-truth performance landscape the agent calibrates toward.

Computes system performance S(water_ratio, print_speed) directly from the
deterministic physics and the evaluation scoring (no surrogate, no noise) for a
fixed design intent — used to draw the performance topology and mark the
theoretical optimum. Scoring constants are imported from the evaluation models
so there is a single source of truth.
"""

from typing import Dict, Tuple

import numpy as np

from sensors.fabrication import FabricationSystem
from sensors.physics import path_deviation, energy_per_segment
from models.evaluation_models import PathAccuracyModel, EnergyConsumptionModel

_N_SEGMENTS = 4


def _score(value: float, target: float, scale: float) -> float:
    """Evaluation scoring: 1 − |value − target| / scale, clamped to [0, 1]."""
    return float(np.clip(1.0 - abs(value - target) / scale, 0.0, 1.0))


def system_performance(
    water_ratio: float, print_speed: float, design: str, material: str,
    fab: FabricationSystem, weights: Dict[str, float],
) -> float:
    """Noiseless system performance S at one operating point."""
    mean_dev = float(np.mean([
        path_deviation(water_ratio, print_speed, design, material, si) for si in range(_N_SEGMENTS)
    ]))
    path_acc = _score(mean_dev, 0.0, PathAccuracyModel.MAX_DEVIATION)
    energy = energy_per_segment(
        print_speed, fab.get_layer_height(design), material, fab.get_layer_time(design, print_speed),
    )
    energy_eff = _score(energy, EnergyConsumptionModel.TARGET_ENERGY, EnergyConsumptionModel.MAX_ENERGY)
    w_p = weights.get("path_accuracy", 1.0)
    w_e = weights.get("energy_efficiency", 1.0)
    return (w_p * path_acc + w_e * energy_eff) / (w_p + w_e)


def true_performance_grid(
    design: str, material: str, fab: FabricationSystem,
    weights: Dict[str, float], bounds: Dict[str, Tuple[float, float]], n: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Return (water_axis, speed_axis, S_grid[speed, water], (w*, s*, S*)).

    The optimum is the grid argmax — the theoretical best operating point.
    """
    wlo, whi = bounds["water_ratio"]
    slo, shi = bounds["print_speed"]
    water = np.linspace(wlo, whi, n)
    speed = np.linspace(slo, shi, n)
    grid = np.empty((n, n))
    for i, sp in enumerate(speed):
        for j, wr in enumerate(water):
            grid[i, j] = system_performance(float(wr), float(sp), design, material, fab, weights)
    oi, oj = divmod(int(np.argmax(grid)), n)
    optimum = (float(water[oj]), float(speed[oi]), float(grid[oi, oj]))
    return water, speed, grid, optimum
