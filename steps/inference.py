"""Single-shot inference with design intent."""
import argparse
import json
import os

from pred_fab.plotting import plot_inference_result
from visualization.helpers import physics_combined_at
from visualization.helpers import get_physics_optimum
from steps._common import (
    run_step,
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot_with_header, get_performance, run_and_record, combined_score,
    predict_score_grid, N_LAYERS,
    X_AXIS, Y_AXIS, FIXED_DIMS, apply_schedule_args,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    perf_weights = agent.calibration_system.performance_weights
    plot_dir = ensure_plot_dir()

    apply_schedule_args(agent, args, config)

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

    spec = agent.acquisition_step(dm, kappa=0.0)
    exp_code = next_code(state, "infer")

    # Merge design_intent into extra_params so it overrides schedule proposals if needed.
    extra = dict(state.prev_params) if state.prev_params else {}
    extra.update(design_intent)
    exp_data, params, sched_data = run_and_record(
        dataset, agent, fab, spec, exp_code,
        extra_params=extra, dataset_code="inference",
    )
    # Re-apply design_intent on top of returned params (schedule may have overridden).
    params.update(design_intent)
    perf = get_performance(exp_data)
    state.record("inference", exp_code, params, perf, trajectory=sched_data)

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

    waters, speeds, pred_grid = predict_score_grid(agent, perf_weights, n_layers=n_layers)

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
    show_plot_with_header(path, "Inference: Predicted Topology", inline=args.plot)

    save_session(config, state)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--design-intent", type=str, default=None,
                        help="JSON: fix parameters for inference. Example: '{\"n_layers\":5}'")
    parser.add_argument("--plot", action="store_true", help="Show plots inline")
    parser.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                        help="Override the configured schedule. Repeatable.")


if __name__ == "__main__":
    run_step(__doc__, add_arguments, run)
