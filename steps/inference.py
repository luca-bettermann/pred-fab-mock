"""First-time-right inference call: kappa=0 acquisition → one fabrication."""
import argparse
import json
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

    design_intent = json.loads(args.design_intent) if args.design_intent else {}

    print("\n  Running inference (kappa=0; performance-only optimisation)...")
    agent.train_step(dataset)
    dm = agent.predict_system.datamodule
    current = with_dimensions(state.prev_params) if state.prev_params else None
    spec = agent.acquisition_step(dm, kappa=0.0, current_params=current)

    code = next_code(state, "infer")
    extra = dict(state.prev_params) if state.prev_params else {}
    extra.update(design_intent)
    exp_data, params, _sched = run_and_record(
        dataset, agent, fab, spec, code,
        extra_params=extra, dataset_code="inference",
    )
    params.update(design_intent)
    perf = get_performance(exp_data)
    state.record("inference", code, params, perf)
    print(f"  {code}: " + ", ".join(f"{k}={v:.3f}" for k, v in perf.items()))

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-time-right inference")
    parser.add_argument("--design-intent", type=str, default=None,
                        help='JSON-encoded fixed parameters, e.g. \'{"layer_height":2.5}\'')
    parser.add_argument("--schedule", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
