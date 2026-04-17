"""Show run summary across all phases."""
import argparse

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import load_session, combined_score


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    perf_weights = config.get("performance_weights") or {
        "path_accuracy": 1, "energy_efficiency": 1, "production_rate": 1,
    }

    print(f"\n  Run Summary:")
    print(f"  {'─' * 60}")
    print(f"  {'Phase':<15s}  {'Experiments':>11s}  {'Best Combined':>14s}")
    print(f"  {'─' * 60}")

    for phase in ["baseline", "exploration", "trajectory", "inference", "adaptation"]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show run summary across all phases")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
