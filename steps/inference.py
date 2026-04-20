"""Single-shot inference with design intent."""
import argparse
import json
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_inference_result
from visualization.helpers import physics_combined_at
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, combined_score, get_physics_optimum, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, FIXED_DIMS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    perf_weights = agent.calibration_system.performance_weights
    plot_dir = ensure_plot_dir()

    if getattr(args, 'schedule', False):
        agent.configure_schedule(
            "print_speed", "n_layers",
            delta=args.delta, smoothing=args.smoothing,
        )

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_layers = design_intent.get("n_layers", N_LAYERS)

    agent.console.print_phase_header(3, "Inference", "First-time-right")

    if design_intent:
        parts = [f"{k}={v}" for k, v in design_intent.items()]
        print(f"  Design intent: {', '.join(parts)}")
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    params.update(design_intent)
    exp_code = next_code(state, "infer")

    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
    perf = get_performance(exp_data)
    state.record("inference", exp_code, params, perf)

    score = combined_score(perf, perf_weights)
    print(f"\n  Proposed parameters:")
    print(f"    water_ratio  = {params['water_ratio']:.3f}")
    print(f"    print_speed  = {params['print_speed']:.1f} mm/s")
    for k, v in design_intent.items():
        print(f"    {k:<13s} = {v}  (design intent)")
    print(f"\n  Performance:")
    for k, v in perf.items():
        print(f"    {k:<20s} = {v:.3f}")
    print(f"    {'combined':<20s} = {score:.3f}")

    opt_w, opt_s = get_physics_optimum(perf_weights, n_layers=n_layers)
    opt_score = physics_combined_at(opt_w, opt_s, perf_weights, n_layers=n_layers)
    gap = opt_score - score
    print(f"\n  Physics optimum: combined={opt_score:.3f} (gap={gap:+.3f})")

    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    pred_grid = np.zeros((40, 40))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                p = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(p, perf_weights)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "05_inference.png")
    plot_inference_result(
        path, X_AXIS, Y_AXIS, waters, speeds, pred_grid,
        proposed={X_AXIS.key: params["water_ratio"], Y_AXIS.key: params["print_speed"]},
        proposed_score=score,
        optimum={X_AXIS.key: opt_w, Y_AXIS.key: opt_s},
        optimum_score=opt_score,
        points=state.all_params,
        fixed_params=FIXED_DIMS,
    )
    show_plot(path, inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-shot first-time-right manufacturing")
    parser.add_argument("--design-intent", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
