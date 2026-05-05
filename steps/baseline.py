"""Run baseline (Sobol space-filling) experiments — no model required."""
import argparse
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session,
)
from utils import get_performance
from visualization import (
    print_phase_header, print_experiment_row, print_phase_summary, print_section,
    plot_baseline_scatter, plot_feature_heatmap, plot_layer_profiles, plot_physics_topology,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = config.get("performance_weights")

    print_phase_header("1", "Baseline Sampling",
                       f"{args.n} Sobol experiments — no model yet, space-filling only")
    specs = agent.baseline_step(n=args.n)

    log: list[tuple[str, dict, dict]] = []
    for spec in specs:
        code = next_code(state, "baseline")
        exp_data, params, _sched = run_and_record(
            dataset, agent, fab, spec, code, dataset_code="baseline",
        )
        perf = get_performance(exp_data)
        state.record("baseline", code, params, perf)
        log.append((code, params, perf))
        print_experiment_row(code, params, perf, perf_weights)

    print_phase_summary(log, perf_weights)

    # Plots
    if log:
        plot_baseline_scatter(log, save_dir=plot_dir)
        plot_feature_heatmap(log[-1][1], exp_code=log[-1][0], save_dir=plot_dir)
        plot_layer_profiles(log[-1][1], exp_code=log[-1][0], save_dir=plot_dir)
        plot_physics_topology(perf_weights, save_dir=plot_dir)
        print_section(f"4 plots saved to {plot_dir}/")

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (Sobol)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--schedule", action="append", default=[],
                        help="Per-trajectory schedule, repeatable: PARAM:DIM")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
