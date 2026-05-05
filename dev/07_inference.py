"""07 — Inference Validation (Single-Shot).

After exploration, test whether a single inference proposal (kappa=0)
lands near high-performance regions — first-time-right manufacturing.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import make_env, run_baseline, train_models, with_dims, run_experiment, ensure_plot_dir
from pred_fab.utils.metrics import combined_score
from utils import params_from_spec

N_BASELINE = 15
N_EXPLORE = 5
PERF_WEIGHTS = {
    "structural_integrity": 1.0,
    "material_deposition": 1.0,
    "extrusion_stability": 1.0,
    "energy_footprint": 1.0,
    "fabrication_time": 1.0,
}
KAPPA = 0.7
EXPLORATION_RADIUS = 0.15


def _combined(perf: dict) -> float:
    return combined_score(perf, PERF_WEIGHTS)


def main():
    warnings.filterwarnings("ignore")
    ensure_plot_dir()

    agent, fab, dataset = make_env("07_inference", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(sigma=EXPLORATION_RADIUS)

    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    # Exploration phase
    prev = baseline_params[-1]
    explore_scores: list[float] = []
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, kappa=KAPPA)
        prev = with_dims({**prev, **params_from_spec(spec)})
        exp = run_experiment(dataset, agent, fab, prev, f"explore_{i+1:02d}")
        perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        explore_scores.append(_combined(perf))
        dm.update()
        agent.train(dm, validate=False)

    # Single-shot inference
    print(f"\n  Inference (single-shot, kappa=0):")
    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dims({**prev, **proposed})
    exp = run_experiment(dataset, agent, fab, params, "infer_01")
    perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
    score = _combined(perf)

    print(f"  Proposed params:")
    for k, v in proposed.items():
        print(f"    {k:<22s} = {v:.4f}")
    print(f"  Inference score: {score:.3f}")
    print(f"  Exploration best: {max(explore_scores):.3f}")
    print(f"  {'✓ Inference improved over exploration' if score > max(explore_scores) else '! Exploration was better'}")


if __name__ == "__main__":
    main()
