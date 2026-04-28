"""04 — Evidence Model Uncertainty Validation.

Verify that evidence-based uncertainty is high where data is sparse
and low near training points. Boundary evidence is included automatically.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensors.physics import N_LAYERS, N_SEGMENTS
from shared import make_env, run_baseline, train_models, ensure_plot_dir
from visualization.helpers import save_fig

N_BASELINE = 15
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
RESOLUTION = 40
EXPLORATION_RADIUS = 0.2


def _compute_uncertainty_grid(agent, dm, res):
    """Compute uncertainty grid (includes boundary evidence)."""
    waters = np.linspace(0.30, 0.50, res)
    speeds = np.linspace(20.0, 60.0, res)
    unc = np.zeros((res, res))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            p = {"water_ratio": w, "print_speed": spd, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
            unc[j, i] = agent.predict_uncertainty(p, dm)
    return waters, speeds, unc


def _plot_uncertainty(save_path, waters, speeds, unc_grid, baseline_params, title):
    """Single-panel uncertainty heatmap with data points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    im = ax.contourf(waters, speeds, unc_grid, levels=20, cmap="Blues")
    if baseline_params:
        bw = [p["water_ratio"] for p in baseline_params]
        bs = [p["print_speed"] for p in baseline_params]
        ax.scatter(bw, bs, s=18, c="white", edgecolors="#3F3F46",
                   linewidth=0.5, zorder=5)
    ax.set_xlabel("Water Ratio")
    ax.set_ylabel("Print Speed [mm/s]")
    plt.colorbar(im, ax=ax, shrink=0.8)

    save_fig(save_path)


def main():
    warnings.filterwarnings("ignore")
    plot_dir = ensure_plot_dir()

    agent, fab, dataset = make_env("04_uncertainty", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(radius=EXPLORATION_RADIUS)
    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    waters, speeds, unc_grid = _compute_uncertainty_grid(agent, dm, RESOLUTION)

    # Probes: verify uncertainty is low near data and high in gaps
    probes = [
        ("Center (0.40, 40)",            {"water_ratio": 0.40, "print_speed": 40.0}),
        ("Near optimum (0.42, 40)",      {"water_ratio": 0.42, "print_speed": 40.0}),
        ("Top-right corner (0.49, 59)",  {"water_ratio": 0.49, "print_speed": 59.0}),
        ("Bottom-left corner (0.31, 21)", {"water_ratio": 0.31, "print_speed": 21.0}),
    ]
    print(f"\n  Uncertainty probes (radius={EXPLORATION_RADIUS}, N={N_BASELINE}):")
    print(f"  {'Location':35s}  {'u':>6s}")
    print(f"  {'─' * 44}")
    for label, p in probes:
        full_p = {**p, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
        u = agent.predict_uncertainty(full_p, dm)
        print(f"  {label:35s}  {u:6.3f}")

    # Evidence model diagnostics
    print(f"\n  Evidence model diagnostics:")
    for kid, kde in agent.pred_system._model_kdes.items():
        model_name = kde.model.outputs[0] if kde.model.outputs else "?"
        print(f"    {model_name}: {kde.n_active_dims} active dims, "
              f"{len(kde.latent_points)} points, σ={kde.sigma:.4f}")

    out = os.path.join(plot_dir, "04_uncertainty.png")
    _plot_uncertainty(out, waters, speeds, unc_grid, baseline_params,
                      title=f"Evidence Uncertainty (radius={EXPLORATION_RADIUS}, N={N_BASELINE})")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
