"""Run baseline (Sobol space-filling) experiments — no model required."""

import argparse
import os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session, show_plot_with_header,
    get_performance, effective_weights, SPEED_AXIS, CALIB_AXIS, DEFAULT_FIXED,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = effective_weights(config)

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 1{_R}{_B} ▸ Baseline Sampling{_R}")
    print(f"  {_D}{args.n} Sobol experiments — no model yet, space-filling only{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

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
        score = combined_score(perf, perf_weights or {})
        print(f"  {code:<14s}  combined={score:.3f}")

    # Phase summary
    scores = [combined_score(p, perf_weights or {}) for _, _, p in log]
    print(f"\n  {_D}{'─' * 40}{_R}")
    print(f"  {len(log)} experiments  "
          f"best={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")

    if getattr(args, "plot", False) and log:
        from visualization.helpers import evaluate_physics_grid
        from pred_fab.plotting import plot_metric_topology, plot_phase_proposals

        speeds, calibs, metrics = evaluate_physics_grid(25, perf_weights)
        individual = {k: v for k, v in metrics.items() if k != "combined"}

        path = os.path.join(plot_dir, "01_baseline_topology.png")
        plot_metric_topology(
            path, SPEED_AXIS, CALIB_AXIS, speeds, calibs,
            individual, metrics["combined"],
            combined_label="combined",
            weights=perf_weights,
            fixed_params={k: v for k, v in DEFAULT_FIXED.items()
                          if k not in ("print_speed", "calibration_factor")},
        )
        show_plot_with_header(path, "Baseline: Ground Truth Topology", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (Sobol)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--schedule", action="append", default=[],
                        help="Per-trajectory schedule, repeatable: PARAM:DIM")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
