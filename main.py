"""Full PFAB journey: Baseline → Training → Exploration → Inference → Adaptation.

Configure the parameters below, then run: python main.py
"""

import numpy as np

from pred_fab.orchestration import Optimizer
from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from sensors.physics import path_deviation as physics_deviation
from utils import params_from_spec
from workflow import (
    JourneyState, clean_artifacts, with_dimensions,
    run_and_evaluate,
)
from reporting import PLOT_DIRS, report_baseline, report_training, \
    report_exploration_round, report_exploration_summary, \
    report_inference_round, report_inference_summary, \
    report_adaptation, report_summary

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — tweak these values to experiment with different settings
# ═══════════════════════════════════════════════════════════════════════════════

QUICK_TEST = False                           # True: minimal runs for fast iteration

# Experiment counts
N_BASELINE   = 2  if QUICK_TEST else 20      # LHS space-filling experiments
N_EXPLORE    = 2  if QUICK_TEST else 10      # exploration rounds
N_INFER      = 1  if QUICK_TEST else 3       # inference rounds

# Agent-level configuration (persists across all phases)
BOUNDS              = {"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)}
PERFORMANCE_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
EXPLORATION_RADIUS  = 0.5                    # KDE bubble size c → h = c·√d/√N, γ = max(1, c·√N)
MPC_LOOKAHEAD       = 0                      # 0 = greedy, N = N-step discounted lookahead
MPC_DISCOUNT        = 0.9                    # discount factor γ for MPC
OPTIMIZER           = Optimizer.DE           # LBFGSB (gradient, fast) or DE (global, slower)
BOUNDARY_BUFFER     = (0.10, 0.8, 2.0)      # (extent, strength, exponent) — penalise edge proposals

# Step-level parameters (per-call, overridable)
W_EXPLORE             = 0.7                  # exploration weight κ ∈ (0, 1]
N_OPTIMIZATION_ROUNDS = 5                    # L-BFGS-B random restarts (ignored by DE)

# Design intent for inference phase
MATERIAL      = "clay"                                     # fixed material for mock
DESIGN_INTENT = {"design": "A", "material": MATERIAL}

# Adaptation
ADAPTATION_START_SPEED = 40.0
ADAPTATION_DELTA       = {"print_speed": 5.0}


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    clean_artifacts(list(PLOT_DIRS.values()))
    state = JourneyState()
    fab   = FabricationSystem(CameraSystem(), EnergySensor())

    # ── Phase 0: Setup ───────────────────────────────────────────────────────
    schema  = build_schema()
    agent   = build_agent(schema, fab.camera, fab.energy)
    dataset = Dataset(schema=schema)

    agent.console.print_phase_header(0, "Setup", "Configure agent, sensors, schema, and calibration bounds")

    agent.configure(
        bounds=BOUNDS,
        performance_weights=PERFORMANCE_WEIGHTS,
        exploration_radius=EXPLORATION_RADIUS,
        mpc_lookahead=MPC_LOOKAHEAD,
        mpc_discount=MPC_DISCOUNT,
        optimizer=OPTIMIZER,
        boundary_buffer=BOUNDARY_BUFFER,
        fixed_params={"material": MATERIAL},
    )

    # ── Phase 1: Baseline ────────────────────────────────────────────────────
    agent.console.print_phase_header(1, "Baseline Sampling",
                       f"{N_BASELINE} Sobol-sequence experiments — no model yet, space-filling only")

    specs = agent.baseline_step(n=N_BASELINE)

    baseline_log = []
    for i, spec in enumerate(specs):
        params = with_dimensions(params_from_spec(spec), fab)
        exp_code = f"baseline_{i+1:02d}"
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = exp_data.performance.get_values_dict()
        baseline_log.append((exp_code, params, {k: float(v) for k, v in perf.items() if v is not None}))
        state.record("baseline", exp_code, params_from_spec(spec), baseline_log[-1][2])

    state.prev_params = with_dimensions(params_from_spec(specs[-1]), fab)
    report_baseline(agent, baseline_log)

    # ── Phase 2: Initial Training ────────────────────────────────────────────
    agent.console.print_phase_header(2, "Initial Training",
                       "Fit prediction models (deviation + energy) on baseline data")

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    results = agent.train(datamodule, validate=True)

    report_training(agent, datamodule, results)

    # ── Phase 3: Exploration ─────────────────────────────────────────────────
    agent.console.print_phase_header(3, "Exploration",
                       f"{N_EXPLORE} rounds  (w_explore={W_EXPLORE}) — model guides search toward uncertain regions")

    explore_log = []
    for i in range(N_EXPLORE):
        # Agent proposes next experiment
        spec = agent.exploration_step(
            datamodule, w_explore=W_EXPLORE, n_optimization_rounds=N_OPTIMIZATION_ROUNDS,
        )

        # Execute experiment
        proposed = params_from_spec(spec)
        params = with_dimensions({**state.prev_params, **proposed}, fab)
        exp_code = f"explore_{i+1:02d}"

        # Plot BEFORE retraining so the topology reflects the landscape the optimizer saw
        proposed_full = {**state.prev_params, **proposed, "n_layers": 5, "n_segments": 4}
        u = agent.predict_uncertainty(proposed_full, datamodule)

        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)

        # Record results
        perf = {k: float(v) for k, v in exp_data.performance.get_values_dict().items() if v is not None}
        explore_log.append((exp_code, params, perf))
        state.record("exploration", exp_code, params, perf)

        report_exploration_round(agent, state, exp_code, params, perf, u, proposed, W_EXPLORE)

        # Retrain on expanded dataset (after plotting)
        datamodule.update()
        agent.train(datamodule, validate=False)

    report_exploration_summary(agent, explore_log, state, N_EXPLORE)

    # ── Phase 4: Inference ───────────────────────────────────────────────────
    agent.console.print_phase_header(4, "Inference",
                       f"{N_INFER} rounds  ·  intent: {DESIGN_INTENT}  ·  w_explore=0")

    agent.configure(fixed_params=DESIGN_INTENT)
    state.prev_params = with_dimensions({**state.prev_params, **DESIGN_INTENT}, fab)

    infer_log = []
    last_exp_data = None
    for i in range(N_INFER):
        params = state.prev_params
        exp_code = f"infer_{i+1:02d}"

        # Execute current parameters
        exp_data = dataset.create_experiment(exp_code, parameters=params)
        fab.run_experiment(params)

        # Agent evaluates and proposes improved parameters
        agent.evaluate(exp_data)
        spec = agent.inference_step(
            exp_data, datamodule, w_explore=0.0,
            n_optimization_rounds=N_OPTIMIZATION_ROUNDS, current_params=params,
        )
        next_params = with_dimensions({**params, **params_from_spec(spec)}, fab)

        # Persist and retrain
        dataset.save_experiment(exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)

        perf = {k: float(v) for k, v in exp_data.performance.get_values_dict().items() if v is not None}
        infer_log.append((exp_code, params, perf))
        last_exp_data = exp_data
        state.record("inference", exp_code, params, perf)
        state.prev_params = next_params

        report_inference_round(agent, state, exp_code, params, perf, DESIGN_INTENT)

    report_inference_summary(agent, infer_log, state, last_exp_data, fab.camera, DESIGN_INTENT, N_INFER)

    # ── Phase 5: Online Adaptation ───────────────────────────────────────────
    agent.console.print_phase_header(5, "Online Adaptation",
                       "print_speed adjusted after each layer based on live deviation feedback")

    agent.configure(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta=ADAPTATION_DELTA,
    )

    # Set up adaptation experiment
    adapt_params = with_dimensions({**state.prev_params, "print_speed": ADAPTATION_START_SPEED}, fab)
    adapt_exp = dataset.create_experiment("adapt_01", parameters=adapt_params)
    agent.set_active_experiment(adapt_exp)

    speeds = []
    deviations = []
    for layer_idx in range(int(adapt_params["n_layers"])):
        # Fabricate one layer
        fab.run_layer(adapt_params, layer_idx)
        start, end = adapt_exp.parameters.get_start_and_end_indices("n_layers", layer_idx)
        agent.feature_system.run_feature_extraction(  # type: ignore[union-attr]
            adapt_exp, evaluate_from=start, evaluate_to=end,
        )

        speed = float(adapt_params["print_speed"])
        speeds.append(speed)
        feat = adapt_exp.features.get_value("path_deviation")
        deviations.append(float(np.mean(feat[layer_idx, :])))

        # Agent adapts speed for next layer
        if layer_idx < int(adapt_params["n_layers"]) - 1:
            spec = agent.adaptation_step(
                dimension="n_layers", step_index=layer_idx,
                exp_data=adapt_exp, record=True,
            )
            adapt_params["print_speed"] = float(spec.initial_params.get("print_speed", speed))

    dataset.save_experiment("adapt_01")

    # Counterfactual: what would happen without adaptation
    n_segs = int(adapt_params["n_segments"])
    counterfactual = [
        float(sum(
            physics_deviation(ADAPTATION_START_SPEED, str(adapt_params["design"]), s,
                              float(adapt_params["water_ratio"]), str(adapt_params["material"]), li)
            for s in range(n_segs)
        ) / n_segs)
        for li in range(len(speeds))
    ]

    report_adaptation(agent, speeds, deviations, counterfactual)

    # ── Summary ──────────────────────────────────────────────────────────────
    report_summary(agent, state, DESIGN_INTENT)


if __name__ == "__main__":
    main()
