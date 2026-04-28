"""03 — Prediction Model Topology Sweep.

Compare prediction quality across MLP HIDDEN topologies. Builds the same
schema and baseline data, then for each topology trains DevMLP / EnergyMLP
with that HIDDEN tuple and reports per-feature R² on the validation set.

The point: at our scale (~100 training rows, 4-6 input features), where
does adding capacity stop paying off? The answer informs whether the
production HIDDEN=(48,24,12) for DevMLP is over- or under-sized.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensors.physics import N_LAYERS, N_SEGMENTS
from visualization import plot_topology_comparison
from visualization.helpers import physics_combined_at, save_fig
from shared import make_env, run_baseline, train_models, ensure_plot_dir
from pred_fab import combined_score
from models.prediction_model import DevMLP, EnergyMLP

N_BASELINE = 20
VAL_SIZE = 0.25
RESOLUTION = 40
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}

# Topologies to sweep. (DevMLP HIDDEN, EnergyMLP HIDDEN, label)
TOPOLOGIES: list[tuple[tuple[int, ...], tuple[int, ...], str]] = [
    ((16,),         (12,),     "shallow"),
    ((48, 24, 12),  (24, 12),  "production"),
    ((96, 48, 24),  (48, 24),  "wide"),
]


def _predict_combined_grid(agent, waters, speeds):
    grid = np.zeros((len(speeds), len(waters)))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({
                    "water_ratio": w, "print_speed": spd,
                    "n_layers": N_LAYERS, "n_segments": N_SEGMENTS,
                })
                grid[j, i] = combined_score(perf, PERF_WEIGHTS)
            except Exception:
                grid[j, i] = 0.0
    return grid


def main():
    warnings.filterwarnings("ignore")
    plot_dir = ensure_plot_dir()

    waters = np.linspace(0.30, 0.50, RESOLUTION)
    speeds = np.linspace(20.0, 60.0, RESOLUTION)
    true_grid = np.array([[physics_combined_at(w, spd) for w in waters] for spd in speeds])

    metrics_per_topo: dict[str, dict] = {}
    grids_per_topo: dict[str, np.ndarray] = {"Ground Truth": true_grid}

    for dev_hidden, energy_hidden, label in TOPOLOGIES:
        DevMLP.HIDDEN = dev_hidden
        EnergyMLP.HIDDEN = energy_hidden

        agent, fab, dataset = make_env(f"03_{label}", verbose=False)
        agent.configure_performance(weights=PERF_WEIGHTS)
        run_baseline(agent, fab, dataset, N_BASELINE)
        _, val_results = train_models(agent, dataset, val_size=VAL_SIZE)
        metrics_per_topo[label] = val_results or {}
        grids_per_topo[label] = _predict_combined_grid(agent, waters, speeds)

    # Console — per-feature R² across topologies
    feature_names = sorted({f for m in metrics_per_topo.values() for f in m if not f.startswith("_")})
    print(f"\n  Topology sweep ({N_BASELINE} baseline, {VAL_SIZE:.0%} held out):")
    header = f"  {'Feature':25s}" + "".join(f"  {label:>12s}" for _, _, label in TOPOLOGIES)
    print(header)
    print(f"  {'─' * (25 + 14 * len(TOPOLOGIES))}")
    for feat in feature_names:
        row = f"  {feat:25s}"
        for _, _, label in TOPOLOGIES:
            r2 = metrics_per_topo[label].get(feat, {}).get("r2", float("nan"))
            row += f"  {r2:12.3f}"
        print(row)

    # Topology side-by-side
    out = os.path.join(plot_dir, "03_topology_sweep.png")
    plot_topology_comparison(out, waters, speeds, grids_per_topo,
                              title="Combined Performance: HIDDEN-topology Sweep")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
