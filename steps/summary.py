"""Show run summary across all phases."""
import argparse

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import load_session, ensure_plot_dir
from visualization import (
    print_run_summary, print_done,
    plot_performance_trajectory,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    perf_weights = config.get("performance_weights")
    plot_dir = ensure_plot_dir()

    print_run_summary(
        state.perf_history, state.all_phases, state.all_codes, perf_weights,
    )

    if state.perf_history:
        plot_performance_trajectory(
            state.perf_history, state.all_phases, state.all_codes,
            perf_weights, save_dir=plot_dir,
        )

    print_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show run summary across all phases")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
