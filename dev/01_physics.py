"""01 — Physics Topology Validation.

Visualize the ground truth performance landscape to confirm:
  - The combined score has a clear, findable optimum
  - The topology is non-trivial (Pareto trade-offs between metrics)
  - The landscape is smooth enough for a model to learn
"""

import os
import numpy as np

from shared import ensure_plot_dir

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visualization.helpers import evaluate_physics_grid
from pred_fab.plotting import plot_metric_topology, AxisSpec

PERF_WEIGHTS = {
    "structural_integrity": 1.0,
    "material_deposition": 1.0,
    "extrusion_stability": 1.0,
    "energy_footprint": 1.0,
    "fabrication_time": 1.0,
}

SPEED_AXIS = AxisSpec("print_speed", "Print Speed", unit="m/s", bounds=(0.004, 0.008))
CALIB_AXIS = AxisSpec("calibration_factor", "Calibration Factor", bounds=(1.6, 2.2))


def main():
    plot_dir = ensure_plot_dir()

    speeds, calibs, metrics = evaluate_physics_grid(30, PERF_WEIGHTS)
    combined = metrics["combined"]
    best_idx = np.unravel_index(np.argmax(combined), combined.shape)
    print(f"  Grid maximum: speed={speeds[best_idx[0]]:.4f} m/s, "
          f"cal={calibs[best_idx[1]]:.2f}, combined={combined[best_idx]:.3f}")

    for name, data in metrics.items():
        if name == "combined":
            continue
        print(f"  {name:25s} at optimum: {data[best_idx]:.3f}")

    individual = {k: v for k, v in metrics.items() if k != "combined"}
    topo_path = os.path.join(plot_dir, "01_topology.png")
    plot_metric_topology(
        topo_path, SPEED_AXIS, CALIB_AXIS, speeds, calibs,
        individual, combined,
        combined_label="combined",
        weights=PERF_WEIGHTS,
    )
    print(f"\n  Saved: {topo_path}")


if __name__ == "__main__":
    main()
