"""Phase-level reporting: console output + plot generation.

Bundles the per-phase printing and plotting so that main.py
stays focused on agent operations.
"""

from typing import Any

from pred_fab import PfabAgent
from pred_fab.core import DataModule, ExperimentData

from sensors import CameraSystem, FabricationSystem
from visualization import (
    plot_path_comparison_3d, plot_filament_volume, plot_prediction_accuracy,
    plot_parameter_space, plot_performance_trajectory, plot_adaptation,
    plot_inference_convergence, plot_acquisition_topology, plot_physics_topology,
    plot_baseline_scatter,
)
from workflow import ExperimentLog, JourneyState, get_physics_optimum


# ── Plot directory layout ────────────────────────────────────────────────────

PLOT_DIRS = {
    "baseline":    "./plots/phase_1_baseline",
    "training":    "./plots/phase_2_training",
    "exploration": "./plots/phase_3_exploration",
    "inference":   "./plots/phase_4_inference",
    "adaptation":  "./plots/phase_5_adaptation",
}


# ── Phase reporters ──────────────────────────────────────────────────────────

def report_baseline(
    agent: PfabAgent,
    log: ExperimentLog,
) -> None:
    """Print baseline experiment rows, summary, and phase plots."""
    for code, params, perf in log:
        agent.console.print_experiment_row(code, params, perf)
    agent.console.print_phase_summary(log)
    plot_physics_topology(agent, save_dir=PLOT_DIRS["baseline"])
    plot_baseline_scatter(log, save_dir=PLOT_DIRS["baseline"])
    agent.console.print_section(f"2 plots saved to {PLOT_DIRS['baseline']}/")


def report_training(
    agent: PfabAgent,
    datamodule: DataModule,
    results: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Plot prediction accuracy and print R² summary. Returns R² scores."""
    r2_scores = plot_prediction_accuracy(agent, datamodule, save_dir=PLOT_DIRS["training"])
    if results:
        agent.console.print_training_summary(results)
    else:
        # Fallback: build per-feature dict from plot's R² scores
        agent.console.print_training_summary({name: {"r2": r2} for name, r2 in r2_scores.items()})
    agent.console.print_section(f"1 plot saved to {PLOT_DIRS['training']}/")
    return r2_scores


def report_exploration_round(
    agent: PfabAgent,
    state: JourneyState,
    code: str,
    params: dict[str, Any],
    perf: dict[str, float],
    uncertainty: float,
    proposed: dict[str, Any],
    w_explore: float,
) -> None:
    """Print one exploration row and save the topology plot."""
    plot_acquisition_topology(
        agent, w_explore, proposed,
        state.all_params, proposed, code,
        save_dir=PLOT_DIRS["exploration"],
    )
    agent.console.print_exploration_row(code, params, perf, uncertainty, agent.last_opt_score)
    agent.console.print_optimizer_stats(agent.last_opt_n_starts, agent.last_opt_nfev)


def report_exploration_summary(
    agent: PfabAgent,
    log: ExperimentLog,
    state: JourneyState,
    n_explore: int,
) -> None:
    """Print exploration phase summary and parameter-space plot."""
    agent.console.print_phase_summary(log)
    plot_parameter_space(
        state.all_params, state.all_phases, state.perf_history,
        save_dir=PLOT_DIRS["exploration"],
    )
    agent.console.print_section(f"{n_explore + 1} plots saved to {PLOT_DIRS['exploration']}/")


def report_inference_round(
    agent: PfabAgent,
    state: JourneyState,
    code: str,
    params: dict[str, Any],
    perf: dict[str, float],
    design_intent: dict[str, Any],
) -> None:
    """Print one inference row and save the topology plot."""
    agent.console.print_inference_row(code, params, perf, agent.last_opt_score)
    agent.console.print_optimizer_stats(agent.last_opt_n_starts, agent.last_opt_nfev)
    plot_acquisition_topology(
        agent, 0.0, params,
        state.all_params, design_intent, code,
        save_dir=PLOT_DIRS["inference"],
    )


def report_inference_summary(
    agent: PfabAgent,
    log: ExperimentLog,
    state: JourneyState,
    last_exp_data: ExperimentData | None,
    camera: CameraSystem,
    design_intent: dict[str, Any],
    n_infer: int,
) -> None:
    """Print inference summary and save all inference-phase plots."""
    agent.console.print_phase_summary(log)
    if last_exp_data is not None:
        plot_path_comparison_3d(last_exp_data, camera, state.prev_params,
                                save_dir=PLOT_DIRS["inference"])
        plot_filament_volume(last_exp_data, camera, state.prev_params,
                             save_dir=PLOT_DIRS["inference"])
    plot_performance_trajectory(
        state.perf_history, state.all_phases, exp_codes=state.all_codes,
        save_dir=PLOT_DIRS["inference"],
    )
    plot_inference_convergence(log, design_intent, save_dir=PLOT_DIRS["inference"])
    agent.console.print_section(f"{n_infer + 4} plots saved to {PLOT_DIRS['inference']}/")


def report_adaptation(
    agent: PfabAgent,
    speeds: list[float],
    deviations: list[float],
    counterfactual: list[float],
) -> None:
    """Print adaptation rows and save the comparison plot."""
    for li in range(len(speeds)):
        if li < len(speeds) - 1:
            agent.console.print_adaptation_row(
                li, speeds[li], deviations[li],
                speed_after=speeds[li + 1],
                n_evals=agent.last_opt_nfev,
            )
        else:
            agent.console.print_adaptation_row(li, speeds[li], deviations[li])

    plot_adaptation(speeds, deviations, no_adapt_deviations=counterfactual,
                    save_dir=PLOT_DIRS["adaptation"])
    agent.console.print_section(f"1 plot saved to {PLOT_DIRS['adaptation']}/")


def report_summary(
    agent: PfabAgent,
    state: JourneyState,
    design_intent: dict[str, Any],
) -> None:
    """Print the final run summary table with physics optimum comparison."""
    agent.console.print_run_summary(
        state.perf_history, state.all_phases, state.all_codes,
    )
    # Mock-specific: show physics optimum for reference
    spd_opt, w_opt = get_physics_optimum(design_intent)
    des = str(design_intent.get("design", "?"))
    mat = str(design_intent.get("material", "?"))
    print(
        f"  \033[1m{'Physics optimum':<16}\033[0m"
        f"{des:<8}{mat:<11}"
        f"{w_opt:>5.2f} {spd_opt:>6.1f}  "
        f"\033[2m(theoretical minimum deviation)\033[0m"
    )
    print(f"  \033[2m{'─' * 58}\033[0m")
    agent.console.print_done()
