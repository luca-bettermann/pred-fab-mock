"""06 — Trajectory Baseline + Exploration Validation.

Validate that trajectory-based (dimensional) proposals work correctly:
  - Step parameters vary per-layer (OFAT at layer granularity)
  - Trajectory proposals produce per-layer speed schedules
  - Comparison: trajectory vs fixed-parameter performance
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import make_env, run_baseline, train_models, with_dims, run_experiment, ensure_plot_dir
from pred_fab.utils.metrics import combined_score
from utils import params_from_spec

N_BASELINE = 10
N_EXPLORE_FIXED = 3
N_EXPLORE_TRAJ = 3
PERF_WEIGHTS = {
    "structural_integrity": 1.0,
    "material_deposition": 1.0,
    "extrusion_stability": 1.0,
    "energy_footprint": 1.0,
    "fabrication_time": 1.0,
}
KAPPA = 0.5
EXPLORATION_RADIUS = 0.15
ADAPTATION_DELTA = {"print_speed": 0.0004}


def _combined(perf: dict) -> float:
    return combined_score(perf, PERF_WEIGHTS)


def main():
    warnings.filterwarnings("ignore")
    plot_dir = ensure_plot_dir()

    agent, fab, dataset = make_env("06_trajectory", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(sigma=EXPLORATION_RADIUS)

    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    # Phase 1: Fixed-parameter exploration
    print(f"\n  Fixed-parameter exploration ({N_EXPLORE_FIXED} rounds):")
    fixed_scores: list[float] = []
    prev = baseline_params[-1]
    for i in range(N_EXPLORE_FIXED):
        spec = agent.exploration_step(dm, kappa=KAPPA)
        proposed = params_from_spec(spec)
        params = with_dims({**prev, **proposed})
        exp = run_experiment(dataset, agent, fab, params, f"fixed_{i+1:02d}")
        perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        score = _combined(perf)
        fixed_scores.append(score)
        print(f"    fixed_{i+1:02d}: combined={score:.3f}")
        dm.update()
        agent.train(dm, validate=False)
        prev = params

    # Phase 2: Trajectory exploration (print_speed per layer)
    print(f"\n  Trajectory exploration ({N_EXPLORE_TRAJ} rounds, print_speed@n_layers):")
    agent.configure_trajectory("print_speed", "n_layers", delta=ADAPTATION_DELTA["print_speed"])

    traj_scores: list[float] = []
    for i in range(N_EXPLORE_TRAJ):
        spec = agent.exploration_step(dm, kappa=KAPPA, current_params=prev)
        proposed = params_from_spec(spec)
        params = with_dims({**prev, **proposed})

        exp = run_experiment(dataset, agent, fab, params, f"traj_{i+1:02d}")
        perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        score = _combined(perf)
        traj_scores.append(score)
        print(f"    traj_{i+1:02d}: combined={score:.3f}")
        dm.update()
        agent.train(dm, validate=False)
        prev = params

    print(f"\n  Summary:")
    print(f"    Fixed avg score:      {np.mean(fixed_scores):.3f}")
    print(f"    Trajectory avg score: {np.mean(traj_scores):.3f}")


if __name__ == "__main__":
    main()
