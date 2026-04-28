"""07 — Inference Validation (Single-Shot).

After exploration, test whether a single inference proposal (kappa=0)
lands near the physics optimum — first-time-right manufacturing.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.orchestration import Optimizer
from sensors.physics import DELTA, THETA, SAG, COMPLEXITY, W_OPTIMAL, N_LAYERS, N_SEGMENTS
from visualization import plot_inference_convergence
from shared import make_env, run_baseline, train_models, with_dims, run_experiment, ensure_plot_dir
from pred_fab import combined_score
from utils import params_from_spec

N_BASELINE = 15
N_EXPLORE = 10
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
KAPPA = 0.7
EXPLORATION_RADIUS = 0.5


def _combined(perf):
    return combined_score(perf, PERF_WEIGHTS)


def main():
    warnings.filterwarnings("ignore")
    plot_dir = ensure_plot_dir()

    agent, fab, dataset = make_env("07_inference", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(radius=EXPLORATION_RADIUS)
    agent.configure_optimizer(backend=Optimizer.DE)
    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    # Exploration phase
    prev = baseline_params[-1]
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, kappa=KAPPA)
        prev = with_dims({**prev, **params_from_spec(spec)})
        run_experiment(dataset, agent, fab, prev, f"explore_{i+1:02d}")
        dm.update()
        agent.train(dm, validate=False)

    # Single-shot inference
    spd_opt = float(np.clip(np.sqrt(THETA * SAG / (DELTA * COMPLEXITY)), 20.0, 60.0))
    w_opt = W_OPTIMAL

    print(f"\n  Inference (single-shot, kappa=0):")
    print(f"  Physics optimum: speed={spd_opt:.1f}, water={w_opt:.2f}")

    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dims({**prev, **proposed})
    exp = run_experiment(dataset, agent, fab, params, "infer_01")
    perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
    score = _combined(perf)

    dist_w = abs(params["water_ratio"] - w_opt)
    dist_s = abs(params["print_speed"] - spd_opt)
    print(f"  Proposed: w={params['water_ratio']:.3f}  spd={params['print_speed']:.1f}  "
          f"score={score:.3f}  dist_w={dist_w:.3f}  dist_s={dist_s:.1f}")

    infer_log = [{"water": params["water_ratio"], "speed": params["print_speed"], "score": score}]
    out = os.path.join(plot_dir, "07_inference.png")
    plot_inference_convergence(out, infer_log, spd_opt, w_opt, perf_weights=PERF_WEIGHTS)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
