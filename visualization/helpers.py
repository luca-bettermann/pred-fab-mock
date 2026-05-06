"""ADVEI-specific visualization helpers.

Only domain-specific data generation lives here — actual plot rendering
uses ``pred_fab.plotting`` functions.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sensors.fabrication import FabricationSystem
from sensors.physics import MAX_N_LAYERS
from models.evaluation_models import (
    StructuralIntegrityEval,
    MaterialDepositionEval,
    ExtrusionStabilityEval,
    EnergyFootprintEval,
    FabricationTimeEval,
)
from pred_fab.utils.metrics import combined_score


# Default fixed parameters when slicing the 5D space to 2D
_DEFAULT_FIXED = {
    "path_offset": 1.5,
    "layer_height": 2.5,
    "calibration_factor": 1.9,
    "print_speed": 0.006,
    "slowdown_factor": 0.3,
    "n_layers": MAX_N_LAYERS,
    "n_nodes": 7,
}


def save_fig(path: str) -> None:
    """Save current figure and close."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_physics_grid(
    resolution: int = 30,
    perf_weights: dict[str, float] | None = None,
    x_param: str = "print_speed",
    y_param: str = "calibration_factor",
    fixed: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate all 5 performance attributes across a 2D parameter slice.

    Returns (x_values, y_values, metrics_dict) where metrics_dict contains
    one grid per performance attribute plus a 'combined' grid.
    """
    from schema import PARAM_BOUNDS

    bounds = {code: (lo, hi) for code, lo, hi in PARAM_BOUNDS}
    base = dict(_DEFAULT_FIXED)
    if fixed:
        base.update(fixed)

    x_lo, x_hi = bounds[x_param]
    y_lo, y_hi = bounds[y_param]
    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)

    fab = FabricationSystem()
    perf_names = [
        "structural_integrity", "material_deposition",
        "extrusion_stability", "energy_footprint", "fabrication_time",
    ]
    grids: dict[str, np.ndarray] = {name: np.zeros((resolution, resolution)) for name in perf_names}
    grids["combined"] = np.zeros((resolution, resolution))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            params = {**base, x_param: float(x), y_param: float(y)}
            fab.run_experiment(params)
            n_layers = int(params["n_layers"])
            n_nodes = int(params["n_nodes"])

            perf = _evaluate_at(fab, params, n_layers, n_nodes)
            for name in perf_names:
                grids[name][i, j] = perf[name]
            grids["combined"][i, j] = combined_score(perf, perf_weights or {})

    return xs, ys, grids


def physics_combined_at(
    print_speed: float,
    calibration_factor: float,
    perf_weights: dict[str, float] | None = None,
    **overrides: float,
) -> float:
    """Evaluate combined physics score at a single (speed, calib) point."""
    params = {**_DEFAULT_FIXED, "print_speed": print_speed,
              "calibration_factor": calibration_factor, **overrides}
    fab = FabricationSystem()
    fab.run_experiment(params)
    n_layers = int(params["n_layers"])
    n_nodes = int(params["n_nodes"])
    perf = _evaluate_at(fab, params, n_layers, n_nodes)
    return combined_score(perf, perf_weights or {})


def _evaluate_at(
    fab: FabricationSystem,
    params: dict,
    n_layers: int,
    n_nodes: int,
) -> dict[str, float]:
    """Compute all 5 performance scores at a parameter point (bypasses agent)."""
    from sensors.physics import (
        TARGET_NODE_OVERLAP_MM, TARGET_FILAMENT_WIDTH_MM,
    )

    node_overlaps = []
    filament_widths = []
    extrusion_vals = []
    energy_vals = []
    duration_vals = []

    for li in range(n_layers):
        for ni in range(n_nodes):
            node_overlaps.append(fab.get_node_feature(params, "node_overlap", li, ni))
            filament_widths.append(fab.get_node_feature(params, "filament_width", li, ni))
        extrusion_vals.append(fab.get_layer_feature(params, "extrusion_consistency", li))
        energy_vals.append(fab.get_layer_feature(params, "robot_energy", li))
        duration_vals.append(fab.get_layer_feature(params, "printing_duration", li))

    def _score(values: list[float], target: float, scaling: float) -> float:
        scores = [max(0.0, 1.0 - abs(v - target) / scaling) for v in values]
        return float(np.mean(scores))

    return {
        "structural_integrity": _score(node_overlaps, TARGET_NODE_OVERLAP_MM, 1.0),
        "material_deposition": _score(filament_widths, TARGET_FILAMENT_WIDTH_MM, 3.0),
        "extrusion_stability": _score(extrusion_vals, 1.0, 0.5),
        "energy_footprint": _score(energy_vals, 5.0, 15.0),
        "fabrication_time": _score(duration_vals, 80.0, 120.0),
    }
