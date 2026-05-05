"""Show run summary across all phases."""

import argparse
import sys as _sys
from pathlib import Path
from collections import Counter

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import load_session, effective_weights


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    perf_weights = effective_weights(config)

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"; _G = "\033[32m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  Summary{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

    if not state.perf_history:
        print("  No experiments recorded yet.")
        return

    # Per-phase breakdown
    phase_counts = Counter(state.all_phases)
    phase_scores: dict[str, list[float]] = {}
    for (params, perf), phase in zip(state.perf_history, state.all_phases):
        phase_scores.setdefault(phase, []).append(combined_score(perf, perf_weights or {}))

    print(f"  {'Phase':<16s} {'Count':>6s} {'Best':>8s} {'Mean':>8s}")
    print(f"  {'─' * 42}")
    for phase in dict.fromkeys(state.all_phases):
        scores = phase_scores[phase]
        count = phase_counts[phase]
        best = max(scores)
        mean = sum(scores) / len(scores)
        print(f"  {phase:<16s} {count:>6d} {best:>8.3f} {mean:>8.3f}")

    all_scores = [combined_score(p, perf_weights or {}) for _, p in state.perf_history]
    print(f"  {'─' * 42}")
    print(f"  {'Total':<16s} {len(all_scores):>6d} {max(all_scores):>8.3f} "
          f"{sum(all_scores)/len(all_scores):>8.3f}")

    # Best experiment
    best_idx = int(max(range(len(all_scores)), key=lambda i: all_scores[i]))
    best_code = state.all_codes[best_idx]
    best_score = all_scores[best_idx]
    print(f"\n  {_G}✓{_R} Best: {best_code} ({best_score:.3f})")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show run summary")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
