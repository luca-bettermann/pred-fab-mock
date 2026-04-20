"""Online inference with layer-by-layer adaptation."""
import argparse
import json

import numpy as np

from pred_fab.utils import Mode

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import (
    load_session, save_session, rebuild, next_code,
    with_dimensions, params_from_spec, get_performance,
    combined_score, N_LAYERS, N_SEGMENTS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    perf_weights = agent.calibration_system.performance_weights

    agent.configure_schedule("print_speed", "n_layers", delta=args.delta)

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_layers = design_intent.get("n_layers", N_LAYERS)
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(5, "Online Inference",
                                      f"Inference + layer-by-layer adaptation")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    print(f"\n  Step 1: Initial inference...")
    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    params.update(design_intent)

    print(f"    Starting params: w={params['water_ratio']:.3f}, spd={params['print_speed']:.1f}")

    exp_code = next_code(state, "adapt")
    exp_data = dataset.create_experiment(exp_code, parameters=params)

    print(f"\n  Step 2: Fabrication with online adaptation ({n_layers} layers):")
    print(f"    {'Layer':<8s}  {'Speed':>8s}  {'Adapted':>8s}  {'Deviation':>10s}")
    print(f"    {'─' * 38}")

    for layer_idx in range(n_layers):
        fab.run_layer(params, layer_idx)
        agent.evaluate(exp_data)

        speed_before = params["print_speed"]

        if layer_idx < n_layers - 1:
            agent.set_active_experiment(exp_data)
            adapt_spec = agent.adaptation_step(
                dimension="n_layers",
                step_index=layer_idx + 1,
                exp_data=exp_data,
                mode=Mode.INFERENCE,
                kappa=0.0,
                record=True,
            )
            new_speed = adapt_spec.initial_params.get("print_speed", speed_before)
            params = {**params, "print_speed": float(new_speed)}
        else:
            new_speed = speed_before

        dev_vals = exp_data.features.get_value("path_deviation")
        if dev_vals is not None and hasattr(dev_vals, '__len__'):
            flat = np.array(dev_vals).flatten()
            n_segs = int(params.get("n_segments", N_SEGMENTS))
            start_idx = layer_idx * n_segs
            end_idx = min(start_idx + n_segs, len(flat))
            if end_idx > start_idx:
                layer_dev = float(np.mean(flat[start_idx:end_idx]))
            else:
                layer_dev = 0.0
        else:
            layer_dev = 0.0

        adapted_str = f"{new_speed:.1f}" if new_speed != speed_before else "\u2014"
        print(f"    {layer_idx+1:<8d}  {speed_before:8.1f}  {adapted_str:>8s}  {layer_dev:10.6f}")

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
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--design-intent", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
