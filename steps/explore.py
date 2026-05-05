"""Run model-guided exploration rounds (incremental)."""
import argparse
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session,
)
from utils import get_performance
from workflow import with_dimensions
from visualization import (
    print_phase_header, print_explore_row, print_phase_summary, print_section,
    plot_parameter_space, plot_acquisition_topology,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = config.get("performance_weights")

    if not state.all_codes:
        raise RuntimeError("No baseline experiments yet — run `cli.py baseline --n N` first.")

    print_phase_header("3", "Exploration",
                       f"{args.n} rounds (κ={args.kappa}) — model guides search toward uncertain regions")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.25)

    log: list[tuple[str, dict, dict]] = []
    for _ in range(args.n):
        agent.train(dm, validate=False)
        current = with_dimensions(state.prev_params) if state.prev_params else None
        spec = agent.exploration_step(dm, kappa=args.kappa, current_params=current)
        code = next_code(state, "explore")

        exp_data, params, _sched = run_and_record(
            dataset, agent, fab, spec, code,
            extra_params=state.prev_params, dataset_code="exploration",
        )
        dm.update()

        perf = get_performance(exp_data)
        state.record("exploration", code, params, perf)
        log.append((code, params, perf))
        print_explore_row(code, params, perf, perf_weights, kappa=args.kappa)

        plot_acquisition_topology(
            agent, args.kappa, params, state.all_params,
            label=code, save_dir=plot_dir,
        )

    print_phase_summary(log, perf_weights)
    plot_parameter_space(
        state.all_params, state.all_phases, state.perf_history,
        perf_weights, save_dir=plot_dir,
    )
    n_plots = len(log) + 1  # topology per round + parameter_space
    print_section(f"{n_plots} plots saved to {plot_dir}/")

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-guided exploration rounds")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=0.5,
                        help="Exploration weight: 1=evidence-only, 0=performance-only")
    parser.add_argument("--schedule", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
