"""Full PFAB journey: Baseline → Training → Exploration → Inference → Adaptation.

Configure the parameters below, then run: python main.py
"""

from pred_fab.orchestration import Optimizer

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from pred_fab.core import Dataset
from utils import params_from_spec
from workflow import (
    JourneyState, clean_artifacts, with_dimensions,
    run_baseline_phase, run_exploration_round, run_inference_round,
    run_adaptation_phase, get_physics_optimum,
)
from visualization import (
    plot_path_comparison_3d, plot_filament_volume, plot_prediction_accuracy,
    plot_parameter_space, plot_performance_trajectory, plot_adaptation,
    plot_inference_convergence, plot_acquisition_topology, plot_physics_topology,
    plot_baseline_scatter,
    print_phase_header, print_section, print_experiment_row, print_phase_summary,
    print_training_summary, print_adaptation_row, print_run_summary, print_done,
    print_explore_row, print_infer_row, print_optimizer_row,
)

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
OPTIMIZER           = Optimizer.LBFGSB       # LBFGSB (gradient, fast) or DE (global, slower)

# Step-level parameters (per-call, overridable)
W_EXPLORE             = 0.7                  # exploration weight κ ∈ (0, 1]
N_OPTIMIZATION_ROUNDS = 5                    # L-BFGS-B random restarts (ignored by DE)

# Design intent for inference phase
DESIGN_INTENT = {"design": "A", "material": "concrete"}

# Adaptation
ADAPTATION_START_SPEED = 40.0
ADAPTATION_DELTA       = {"print_speed": 5.0}

# Plot directories
PLOT_DIRS = {
    "baseline":    "./plots/phase_1_baseline",
    "training":    "./plots/phase_2_training",
    "exploration": "./plots/phase_3_exploration",
    "inference":   "./plots/phase_4_inference",
    "adaptation":  "./plots/phase_5_adaptation",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    clean_artifacts(list(PLOT_DIRS.values()))
    state = JourneyState()
    d = PLOT_DIRS

    # ── Phase 0: Setup ────────────────────────────────────────────────────────
    print_phase_header(0, "Setup", "Configure agent, sensors, schema, and calibration bounds")
    schema = build_schema()
    fab    = FabricationSystem(CameraSystem(), EnergySensor())
    agent  = build_agent(schema, fab.camera, fab.energy)

    agent.configure(
        bounds=BOUNDS,
        performance_weights=PERFORMANCE_WEIGHTS,
        exploration_radius=EXPLORATION_RADIUS,
        mpc_lookahead=MPC_LOOKAHEAD,
        mpc_discount=MPC_DISCOUNT,
        optimizer=OPTIMIZER,
    )
    dataset = Dataset(schema=schema)

    # ── Phase 1: Baseline ─────────────────────────────────────────────────────
    print_phase_header(1, "Baseline Sampling",
                       f"{N_BASELINE} Latin-hypercube experiments — no model yet, space-filling only")

    baseline_log, baseline_exps, baseline_specs = run_baseline_phase(
        agent, dataset, fab, N_BASELINE, state,
    )
    for code, params, perf in baseline_log:
        print_experiment_row(code, params, perf)

    print_phase_summary(baseline_log)
    plot_physics_topology(agent, save_dir=d["baseline"])
    plot_baseline_scatter(baseline_log, save_dir=d["baseline"])
    print_section(f"2 plots saved to {d['baseline']}/")

    # ── Phase 2: Initial Training ─────────────────────────────────────────────
    print_phase_header(2, "Initial Training",
                       "Fit prediction models (deviation + energy) on baseline data")
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    agent.logger.set_console_output(False)
    agent.train(datamodule, validate=True)
    agent.logger.set_console_output(True)

    r2_scores = plot_prediction_accuracy(agent, datamodule, save_dir=d["training"])
    print_training_summary(r2_scores)
    print_section(f"1 plot saved to {d['training']}/")

    # ── Phase 3: Exploration ──────────────────────────────────────────────────
    print_phase_header(3, "Exploration",
                       f"{N_EXPLORE} rounds  (w_explore={W_EXPLORE}) — model guides search toward uncertain regions")

    explore_log = []
    for i in range(N_EXPLORE):
        code, params, perf, u, proposed = run_exploration_round(
            agent, dataset, fab, datamodule, state, i, W_EXPLORE, N_OPTIMIZATION_ROUNDS,
        )
        explore_log.append((code, params, perf))

        # Topology plot (shows what the model saw before this proposal)
        plot_acquisition_topology(
            agent, W_EXPLORE, proposed,
            state.all_params, proposed, code, save_dir=d["exploration"],
        )
        print_explore_row(code, params, perf, u, agent.last_opt_score, W_EXPLORE)
        print_optimizer_row(agent.last_opt_n_starts, agent.last_opt_nfev)

    print_phase_summary(explore_log)
    plot_parameter_space(state.all_params, state.all_phases, state.perf_history,
                         save_dir=d["exploration"])
    print_section(f"{N_EXPLORE + 1} plots saved to {d['exploration']}/")

    # ── Phase 4: Inference ────────────────────────────────────────────────────
    print_phase_header(4, "Inference",
                       f"{N_INFER} rounds  ·  intent: {DESIGN_INTENT}  ·  w_explore=0")
    agent.configure(fixed_params=DESIGN_INTENT)
    state.prev_params = with_dimensions({**state.prev_params, **DESIGN_INTENT}, fab)

    infer_log = []
    last_exp_data = None
    for i in range(N_INFER):
        code, params, perf, exp_data = run_inference_round(
            agent, dataset, fab, datamodule, state, i, N_OPTIMIZATION_ROUNDS,
        )
        infer_log.append((code, params, perf))
        last_exp_data = exp_data
        print_infer_row(code, params, perf, agent.last_opt_score)
        print_optimizer_row(agent.last_opt_n_starts, agent.last_opt_nfev)

        plot_acquisition_topology(
            agent, 0.0, params, state.all_params, DESIGN_INTENT, code, save_dir=d["inference"],
        )

    print_phase_summary(infer_log)
    if last_exp_data is not None:
        plot_path_comparison_3d(last_exp_data, fab.camera, state.prev_params, save_dir=d["inference"])
        plot_filament_volume(last_exp_data, fab.camera, state.prev_params, save_dir=d["inference"])
    plot_performance_trajectory(state.perf_history, state.all_phases, exp_codes=state.all_codes,
                                save_dir=d["inference"])
    plot_inference_convergence(infer_log, DESIGN_INTENT, save_dir=d["inference"])
    print_section(f"{N_INFER + 4} plots saved to {d['inference']}/")

    # ── Phase 5: Online Adaptation ────────────────────────────────────────────
    print_phase_header(5, "Online Adaptation",
                       "print_speed adjusted after each layer based on live deviation feedback")
    agent.configure(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta=ADAPTATION_DELTA,
    )

    speeds, deviations, counterfactual = run_adaptation_phase(
        agent, dataset, fab, state, start_speed=ADAPTATION_START_SPEED,
    )
    for li in range(len(speeds)):
        if li < len(speeds) - 1:
            print_adaptation_row(li, speeds[li], deviations[li],
                                 speed_after=speeds[li + 1] if li + 1 < len(speeds) else None,
                                 n_evals=agent.last_opt_nfev)
        else:
            print_adaptation_row(li, speeds[li], deviations[li])

    plot_adaptation(speeds, deviations, no_adapt_deviations=counterfactual,
                    save_dir=d["adaptation"])
    print_section(f"1 plot saved to {d['adaptation']}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    spd_opt, w_opt = get_physics_optimum(DESIGN_INTENT)
    print_run_summary(state.perf_history, state.all_phases, state.all_codes, DESIGN_INTENT,
                      phys_opt_speed=spd_opt, phys_opt_water=w_opt)
    print_done()


if __name__ == "__main__":
    main()
