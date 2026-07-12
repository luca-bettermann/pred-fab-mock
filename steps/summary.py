"""Show run summary across all phases."""
import argparse
import os

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from visualization import plot_journey, physics_combined_at, get_physics_optimum
from steps._common import (
    load_session, combined_score, ensure_plot_dir, show_plot_with_header,
    load_physics_from_session,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    perf_weights = config.get("performance_weights") or {
        "path_accuracy": 1.0, "energy_efficiency": 1.0, "production_rate": 1.0,
    }

    print(f"\n  Run Summary:")
    print(f"  {'─' * 60}")
    print(f"  {'Phase':<15s}  {'Experiments':>11s}  {'Best Combined':>14s}")
    print(f"  {'─' * 60}")

    for phase in ["baseline", "exploration", "inference", "adaptation"]:
        indices = [i for i, p in enumerate(state.all_phases) if p == phase]
        if not indices:
            continue
        scores = [combined_score(state.perf_history[i][1], perf_weights)
                  for i in indices]
        best = max(scores)
        print(f"  {phase:<15s}  {len(indices):>11d}  {best:>14.3f}")

    print(f"  {'─' * 60}")
    total = len(state.all_params)
    test_n = config.get("test_set_n", 0)
    print(f"  Total: {total} training experiments + {test_n} test experiments")
    print()

    if state.perf_history:
        load_physics_from_session(config)
        all_scores = [combined_score(perf, perf_weights) for _, perf in state.perf_history]
        opt_w, opt_s = get_physics_optimum(perf_weights)
        opt_score = physics_combined_at(opt_w, opt_s, perf_weights)
        path = os.path.join(ensure_plot_dir(), "06_journey.png")
        plot_journey(path, state.all_phases, all_scores, optimum_score=opt_score)
        show_plot_with_header(path, "Journey: Score vs Experiments",
                              inline=getattr(args, "plot", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show run summary across all phases")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
