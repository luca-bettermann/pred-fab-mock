"""Online inference with layer-by-layer adaptation."""
import argparse
import json

import numpy as np

from pred_fab.utils import Mode

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import (
    load_session, save_session, rebuild, next_code,
    with_dimensions, params_from_spec, get_performance,
    combined_score, N_LAYERS, N_SEGMENTS, apply_schedule_args,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    perf_weights = agent.calibration_system.performance_weights

    apply_schedule_args(agent, args, config)

    # Derive scheduled params and dimension from what was just configured
    cal = agent.calibration_system
    sched_params = list(cal.schedule_configs.keys())
    sched_dims = list(cal.schedule_configs.values())
    if not sched_params:
        print("  Error: no schedule configured. Run 'configure --schedule PARAM:DIM' first.")
        return
    # Use the first dimension for the adaptation loop
    adapt_dim = sched_dims[0]

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_steps = design_intent.get(adapt_dim, N_LAYERS)
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(5, "Online Inference",
                                      f"Inference + step-by-step adaptation")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    print(f"\n  Step 1: Initial inference...")
    spec = agent.acquisition_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    params.update(design_intent)

    # Display starting values for scheduled params
    sched_summary = ", ".join(f"{p}={params.get(p, '?')}" for p in sched_params)
    print(f"    Starting params: {sched_summary}")

    exp_code = next_code(state, "adapt")
    exp_data = dataset.create_experiment(exp_code, parameters=params, dataset_code="adapt")

    # Build table header from scheduled param names
    hdr_params = "  ".join(f"{p:>10s}" for p in sched_params)
    hdr_adapted = "  ".join(f"{'adapted':>10s}" for _ in sched_params)
    print(f"\n  Step 2: Fabrication with online adaptation ({n_steps} steps, dim={adapt_dim}):")
    print(f"    {'Step':<6s}  {hdr_params}  {hdr_adapted}  {'Deviation':>10s}")
    print(f"    {'─' * (8 + 24 * len(sched_params) + 12)}")

    for step_idx in range(n_steps):
        fab.run_layer(params, step_idx)
        agent.evaluate(exp_data)

        vals_before = {p: params.get(p) for p in sched_params}

        if step_idx < n_steps - 1:
            agent.set_active_experiment(exp_data)
            adapt_spec = agent.adaptation_step(
                dimension=adapt_dim,
                step_index=step_idx + 1,
                exp_data=exp_data,
                mode=Mode.INFERENCE,
                kappa=0.0,
                record=True,
            )
            for p in sched_params:
                new_val = adapt_spec.initial_params.get(p, vals_before[p])
                params = {**params, p: float(new_val)}

        # Compute layer deviation (mock-specific sensor data)
        dev_vals = exp_data.features.get_value("path_deviation")
        if dev_vals is not None and hasattr(dev_vals, '__len__'):
            flat = np.array(dev_vals).flatten()
            n_segs = int(params.get("n_segments", N_SEGMENTS))
            start_idx = step_idx * n_segs
            end_idx = min(start_idx + n_segs, len(flat))
            if end_idx > start_idx:
                layer_dev = float(np.mean(flat[start_idx:end_idx]))
            else:
                layer_dev = 0.0
        else:
            layer_dev = 0.0

        # Format table row
        before_cols = "  ".join(f"{float(vals_before[p]):10.1f}" for p in sched_params)
        after_cols = "  ".join(
            f"{float(params.get(p, 0)):10.1f}" if params.get(p) != vals_before[p] else f"{'—':>10s}"
            for p in sched_params
        )
        print(f"    {step_idx+1:<6d}  {before_cols}  {after_cols}  {layer_dev:10.6f}")

    dataset.save_experiment(exp_code)
    perf = get_performance(exp_data)
    state.record("adaptation", exp_code, params, perf)

    score = combined_score(perf, perf_weights)
    print(f"\n  Result:")
    for k, v in perf.items():
        print(f"    {k:<20s} = {v:.3f}")
    print(f"    {'combined':<20s} = {score:.3f}")

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online inference with layer-by-layer adaptation")
    parser.add_argument("--design-intent", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
