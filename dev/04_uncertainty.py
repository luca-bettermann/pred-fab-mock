"""04 — Evidence Model Uncertainty Validation.

Verify that evidence-based uncertainty is high where data is sparse
and low near training points. Boundary evidence is included automatically.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensors.physics import MAX_N_LAYERS
from shared import make_env, run_baseline, train_models, ensure_plot_dir, with_dims
from visualization.helpers import save_fig
from workflow import N_NODES

N_BASELINE = 15
PERF_WEIGHTS = {
    "structural_integrity": 1.0,
    "material_deposition": 1.0,
    "extrusion_stability": 1.0,
    "energy_footprint": 1.0,
    "fabrication_time": 1.0,
}
RESOLUTION = 25
EXPLORATION_RADIUS = 0.15


def _compute_uncertainty_grid(agent, dm, res):
    """Compute uncertainty across (print_speed × calibration_factor) slice."""
    speeds = np.linspace(0.004, 0.008, res)
    calibs = np.linspace(1.6, 2.2, res)
    unc = np.zeros((res, res))
    for i, spd in enumerate(speeds):
        for j, cal in enumerate(calibs):
            p = with_dims({
                "path_offset": 1.5, "layer_height": 2.5,
                "calibration_factor": float(cal),
                "print_speed": float(spd), "slowdown_factor": 0.3,
            })
            unc[i, j] = agent.predict_uncertainty(p, dm)
    return speeds, calibs, unc


def _plot_uncertainty(save_path, speeds, calibs, unc_grid, baseline_params, title):
    """Single-panel uncertainty heatmap with data points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    im = ax.contourf(speeds, calibs, unc_grid.T, levels=20, cmap="Blues")
    if baseline_params:
        bs = [p["print_speed"] for p in baseline_params]
        bc = [p["calibration_factor"] for p in baseline_params]
        ax.scatter(bs, bc, s=18, c="white", edgecolors="#3F3F46",
                   linewidth=0.5, zorder=5)
    ax.set_xlabel("Print Speed [m/s]")
    ax.set_ylabel("Calibration Factor")
    plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)


def main():
    warnings.filterwarnings("ignore")
    plot_dir = ensure_plot_dir()

    agent, fab, dataset = make_env("04_uncertainty", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(sigma=EXPLORATION_RADIUS)
    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    speeds, calibs, unc_grid = _compute_uncertainty_grid(agent, dm, RESOLUTION)

    probes = [
        ("Center (spd=0.006, cal=1.9)", {"print_speed": 0.006, "calibration_factor": 1.9}),
        ("Fast (spd=0.008, cal=1.9)",   {"print_speed": 0.008, "calibration_factor": 1.9}),
        ("Corner (spd=0.004, cal=2.2)", {"print_speed": 0.004, "calibration_factor": 2.2}),
    ]
    print(f"\n  Uncertainty probes (radius={EXPLORATION_RADIUS}, N={N_BASELINE}):")
    print(f"  {'Location':35s}  {'u':>6s}")
    print(f"  {'─' * 44}")
    for label, p in probes:
        full_p = with_dims({**p, "path_offset": 1.5, "layer_height": 2.5, "slowdown_factor": 0.3})
        u = agent.predict_uncertainty(full_p, dm)
        print(f"  {label:35s}  {u:6.3f}")

    out = os.path.join(plot_dir, "04_uncertainty.png")
    _plot_uncertainty(out, speeds, calibs, unc_grid, baseline_params,
                      title=f"Evidence Uncertainty (radius={EXPLORATION_RADIUS}, N={N_BASELINE})")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
