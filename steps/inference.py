"""First-time-right inference call: kappa=0 acquisition → one fabrication."""

import argparse
import json
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session, show_plot_with_header,
    get_performance, effective_weights,
)
from workflow import with_dimensions


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = effective_weights(config)

    design_intent = json.loads(args.design_intent) if args.design_intent else {}

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 4{_R}{_B} ▸ Inference{_R}")
    intent_str = f"  ·  intent: {design_intent}" if design_intent else ""
    print(f"  {_D}κ=0 (performance-only){intent_str}{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.25)
    agent.train(dm, validate=False)
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

    score = combined_score(perf, perf_weights or {})
    print(f"  {code:<14s}  combined={score:.3f}")
    for k, v in perf.items():
        print(f"    {k:<24s} {v:.3f}")

    if getattr(args, "plot", False):
        import os
        from pred_fab.plotting import plot_performance_radar

        all_perfs = [p for _, p in state.perf_history]
        dataset_scores = [combined_score(p, perf_weights) for _, p in state.perf_history]
        dataset_avg = float(sum(dataset_scores) / len(dataset_scores))

        path = os.path.join(plot_dir, "04_inference.png")
        plot_performance_radar(
            path,
            performance=perf,
            dataset_performances=all_perfs,
            weights=perf_weights,
            combined_score=score,
            dataset_combined=dataset_avg,
            exp_code=code,
        )
        show_plot_with_header(path, "Inference: Performance Radar", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-time-right inference")
    parser.add_argument("--design-intent", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--schedule", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
