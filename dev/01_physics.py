"""01 — Physics Topology Validation.

Visualize the ground truth performance landscape to confirm:
  - The combined score has a clear, findable optimum
  - The topology is non-trivial (Pareto trade-offs between metrics)
  - The landscape is smooth enough for an MLP to learn
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensors.physics import DELTA, THETA, SAG, COMPLEXITY, W_OPTIMAL
from visualization import plot_physics_topology, plot_cross_sections, evaluate_physics_grid
from shared import ensure_plot_dir

PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}


def main():
    plot_dir = ensure_plot_dir()
    spd_opt = float(np.clip(np.sqrt(THETA * SAG / (DELTA * COMPLEXITY)), 20.0, 60.0))
    w_opt = W_OPTIMAL

    print(f"  Physics optimum: speed={spd_opt:.1f} mm/s, water={w_opt:.2f}")

    waters, speeds, metrics = evaluate_physics_grid(50, PERF_WEIGHTS)
    combined = list(metrics.values())[-1]
    best_idx = np.unravel_index(np.argmax(combined), combined.shape)
    print(f"  Grid maximum:    speed={speeds[best_idx[0]]:.1f} mm/s, "
          f"water={waters[best_idx[1]]:.2f}, combined={combined[best_idx]:.3f}")

    for name, data in metrics.items():
        w_idx = np.argmin(np.abs(waters - w_opt))
        s_idx = np.argmin(np.abs(speeds - spd_opt))
        print(f"  {name:25s} at optimum: {data[s_idx, w_idx]:.3f}")

    topo_path = os.path.join(plot_dir, "01_topology.png")
    plot_physics_topology(topo_path, opt_speed=spd_opt, opt_water=w_opt, perf_weights=PERF_WEIGHTS)
    print(f"\n  Saved: {topo_path}")

    cross_path = os.path.join(plot_dir, "01_cross_sections.png")
    plot_cross_sections(cross_path, spd_opt, w_opt, perf_weights=PERF_WEIGHTS)
    print(f"  Saved: {cross_path}")


if __name__ == "__main__":
    main()
