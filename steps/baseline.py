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


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    ensure_plot_dir()

    print(f"\n  Running {args.n} baseline (Sobol) experiments...")
    specs = agent.baseline_step(n=args.n)

    for spec in specs:
        code = next_code(state, "baseline")
        exp_data, params, _sched = run_and_record(
            dataset, agent, fab, spec, code, dataset_code="baseline",
        )
        perf = get_performance(exp_data)
        state.record("baseline", code, params, perf)
        print(f"  {code}: " + ", ".join(f"{k}={v:.3f}" for k, v in perf.items()))

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (Sobol)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--schedule", action="append", default=[],
                        help="Per-trajectory schedule, repeatable: PARAM:DIM")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
