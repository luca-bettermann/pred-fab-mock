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

from pred_fab.orchestration import Optimizer
from visualization import plot_trajectory_comparison
from shared import make_env, run_baseline, train_models, with_dims, run_experiment, ensure_plot_dir
from pred_fab import combined_score
from utils import params_from_spec

N_BASELINE = 10
N_EXPLORE_FIXED = 3
N_EXPLORE_TRAJ = 3
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
KAPPA = 0.5
EXPLORATION_RADIUS = 0.15
ADAPTATION_DELTA = {"print_speed": 5.0}
MPC_LOOKAHEAD = 2    # look 2 layers ahead for trajectory optimization
MPC_DISCOUNT = 0.9   # discount factor for future layers
TRAJECTORY_SMOOTHING = 0.25  # penalize speed changes between layers (0=off, 0.3=strong)


def _combined(perf):
    return combined_score(perf, PERF_WEIGHTS)


def _extract_schedules(spec):
    """Extract per-layer speed values from an ExperimentSpec.

    ExperimentSpec.schedules is {dim_code: ParameterSchedule}. Each schedule
    has .entries: list[tuple[step_index, ParameterProposal]]. We reconstruct
    the full per-layer sequence, filling layer 0 from initial_params.
    """
    result = {}
    initial = dict(spec.initial_params.to_dict())
    for dim_code, sched in spec.schedules.items():
        # Collect which params this schedule covers
        for _, proposal in sched.entries:
            for param_code in proposal.to_dict():
                # Build full sequence: layer 0 = initial, rest from schedule
                values = [initial.get(param_code, 0.0)]
                for step_idx, prop in sched.entries:
                    values.append(prop.to_dict().get(param_code, values[-1]))
                result[param_code] = values
    return result


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    plot_dir = ensure_plot_dir()

    agent, fab, dataset = make_env("06_trajectory", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(radius=EXPLORATION_RADIUS)
    agent.configure_optimizer(backend=Optimizer.DE)
    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    # Phase 1: Fixed-parameter exploration
    print(f"\n  Fixed-parameter exploration ({N_EXPLORE_FIXED} rounds):")
    fixed_scores = []
    prev = baseline_params[-1]
    for i in range(N_EXPLORE_FIXED):
        spec = agent.exploration_step(dm, kappa=KAPPA)
        proposed = params_from_spec(spec)
        params = with_dims({**prev, **proposed})
        exp = run_experiment(dataset, agent, fab, params, f"fixed_{i+1:02d}")
        perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        score = _combined(perf)
        fixed_scores.append(score)
        dm.update()
        agent.train(dm, validate=False)
        prev = params

    # Phase 2: Trajectory exploration (OFAT on print_speed per layer)
    print(f"\n  Trajectory exploration ({N_EXPLORE_TRAJ} rounds, step_param=print_speed@n_layers):")
    agent.configure_trajectory(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta=ADAPTATION_DELTA,
        mpc_lookahead=MPC_LOOKAHEAD,
        mpc_discount=MPC_DISCOUNT,
        smoothing=TRAJECTORY_SMOOTHING,
    )

    traj_scores = []
    traj_schedules = []
    for i in range(N_EXPLORE_TRAJ):
        # Pass current_params so _build_step_grid constructs the multi-step grid
        spec = agent.exploration_step(dm, kappa=KAPPA, current_params=prev)
        proposed = params_from_spec(spec)
        params = with_dims({**prev, **proposed})

        schedules = _extract_schedules(spec)
        traj_schedules.append(schedules)

        exp = run_experiment(dataset, agent, fab, params, f"traj_{i+1:02d}")
        perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        score = _combined(perf)
        traj_scores.append(score)
        dm.update()
        agent.train(dm, validate=False)
        prev = params

        # Schedule is printed by pred-fab's exploration_step console output

    out = os.path.join(plot_dir, "06_trajectory.png")
    plot_trajectory_comparison(out, fixed_scores, traj_scores, traj_schedules)
    print(f"\n  Saved: {out}")

    print(f"\n  Summary:")
    print(f"    Fixed avg score:      {np.mean(fixed_scores):.3f}")
    print(f"    Trajectory avg score: {np.mean(traj_scores):.3f}")


if __name__ == "__main__":
    main()
