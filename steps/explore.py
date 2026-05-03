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


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    ensure_plot_dir()

    if not state.all_codes:
        raise RuntimeError("No baseline experiments yet — run `cli.py baseline --n N` first.")

    print(f"\n  Running {args.n} exploration rounds (kappa={args.kappa})...")
    for _ in range(args.n):
        agent.train_step(dataset)
        dm = agent.predict_system.datamodule
        current = with_dimensions(state.prev_params) if state.prev_params else None
        spec = agent.acquisition_step(dm, kappa=args.kappa, current_params=current)
        code = next_code(state, "explore")

        exp_data, params, _sched = run_and_record(
            dataset, agent, fab, spec, code,
            extra_params=state.prev_params, dataset_code="exploration",
        )
        perf = get_performance(exp_data)
        state.record("exploration", code, params, perf)
        print(f"  {code}: " + ", ".join(f"{k}={v:.3f}" for k, v in perf.items()))

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
