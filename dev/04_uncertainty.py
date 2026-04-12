"""04 — KDE Uncertainty + Boundary Buffer Validation.

Verify that KDE-based uncertainty is high where data is sparse,
low near training points, and boundary buffer shapes it correctly.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensors.physics import N_LAYERS, N_SEGMENTS
from visualization import plot_uncertainty, plot_uncertainty_cross_sections
from shared import make_env, run_baseline, train_models, ensure_plot_dir

N_BASELINE = 20
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
RESOLUTION = 40
EXPLORATION_RADIUS = 0.5
BOUNDARY_BUFFER = (0.10, 0.8, 2.0)


def _compute_grids(agent, dm, res):
    """Compute uncertainty and boundary factor grids."""
    waters = np.linspace(0.30, 0.50, res)
    speeds = np.linspace(20.0, 60.0, res)
    unc = np.zeros((res, res))
    bf = np.zeros((res, res))
    cal = agent.calibration_system
    bounds = cal._get_global_bounds(dm)
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            p = {"water_ratio": w, "print_speed": spd, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
            unc[j, i] = agent.predict_uncertainty(p, dm)
            bf[j, i] = cal._boundary_factor(dm.params_to_array(p), bounds)
    return waters, speeds, unc, bf


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    plot_dir = ensure_plot_dir()

    agent, fab, dataset = make_env("04_uncertainty", verbose=False)
    agent.configure(performance_weights=PERF_WEIGHTS,
                    exploration_radius=EXPLORATION_RADIUS, boundary_buffer=BOUNDARY_BUFFER)
    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    waters, speeds, unc_grid, bf_grid = _compute_grids(agent, dm, RESOLUTION)
    unc_buffered = unc_grid * bf_grid

    # Probes
    probes = [
        ("Center (0.40, 40)",            {"water_ratio": 0.40, "print_speed": 40.0}),
        ("Near optimum (0.42, 40)",      {"water_ratio": 0.42, "print_speed": 40.0}),
        ("Top-right corner (0.49, 59)",  {"water_ratio": 0.49, "print_speed": 59.0}),
        ("Bottom-left corner (0.31, 21)", {"water_ratio": 0.31, "print_speed": 21.0}),
    ]
    print(f"\n  Uncertainty probes (radius={EXPLORATION_RADIUS}):")
    print(f"  {'Location':35s}  {'u':>6s}  {'bf':>6s}  {'u*bf':>6s}")
    print(f"  {'─' * 58}")
    cal = agent.calibration_system
    bounds = cal._get_global_bounds(dm)
    for label, p in probes:
        full_p = {**p, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
        u = agent.predict_uncertainty(full_p, dm)
        b = cal._boundary_factor(dm.params_to_array(full_p), bounds)
        print(f"  {label:35s}  {u:6.3f}  {b:6.3f}  {u*b:6.3f}")

    out = os.path.join(plot_dir, "04_uncertainty.png")
    plot_uncertainty(out, waters, speeds, unc_grid, bf_grid, baseline_params,
                     title=f"KDE Uncertainty (radius={EXPLORATION_RADIUS}, buffer={BOUNDARY_BUFFER})")
    print(f"\n  Saved: {out}")

    out = os.path.join(plot_dir, "04_uncertainty_cross.png")
    plot_uncertainty_cross_sections(out, waters, speeds, unc_grid, unc_buffered)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
