"""Trajectory exploration: per-layer speed optimization with MPC lookahead."""
import argparse
import json

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import (
    load_session, save_session, rebuild, next_code,
    with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, N_LAYERS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)

    agent.configure_trajectory(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta={"print_speed": args.delta},
        smoothing=args.smoothing,
        mpc_lookahead=args.lookahead,
        mpc_discount=args.discount,
    )

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_layers = design_intent.get("n_layers", N_LAYERS)
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(4, "Trajectory Exploration",
                                      f"{args.n} rounds, \u0394speed=\u00b1{args.delta} mm/s")
    if design_intent:
        parts = [f"{k}={v}" for k, v in design_intent.items()]
        print(f"  Design intent: {', '.join(parts)}")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    prev = state.prev_params if state.prev_params else {}

    for i in range(args.n):
        spec = agent.exploration_step(dm, kappa=args.kappa, current_params=prev)
        proposed = params_from_spec(spec)
        params = with_dimensions({**prev, **proposed})
        params.update(design_intent)

        exp_code = next_code(state, "traj")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        if spec.schedules:
            spec.apply_schedules(exp_data)

        perf = get_performance(exp_data)
        state.record("trajectory", exp_code, params, perf)

        dm.update()
        agent.train(dm, validate=False)
        prev = params

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory exploration: per-layer speed optimization")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--design-intent", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
