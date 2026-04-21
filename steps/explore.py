"""Run exploration rounds (incremental -- can be called multiple times)."""
import argparse
import os

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_acquisition
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, compute_acquisition_grid,
    X_AXIS, Y_AXIS, FIXED_DIMS, apply_schedule_args,
    extract_schedule_steps,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    plot_dir = ensure_plot_dir()

    apply_schedule_args(agent, args)
    if getattr(args, 'schedule', None) and getattr(args, 'design_intent', None):
        import json
        design_intent = json.loads(args.design_intent)
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    n_existing = len([p for p in state.all_phases if p == "exploration"])
    total_after = n_existing + args.n

    agent.console.print_phase_header(2, "Exploration",
                                      f"rounds {n_existing+1}..{total_after} (\u03ba={args.kappa})")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=args.validate)

    for i in range(args.n):
        round_num = n_existing + i + 1
        # Pass current_params so schedule dimensions are resolved
        current = with_dimensions(state.prev_params) if state.prev_params else None
        spec = agent.exploration_step(dm, kappa=args.kappa, current_params=current)
        proposed = params_from_spec(spec)
        params = with_dimensions({**state.prev_params, **proposed})
        exp_code = next_code(state, "explore")

        if args.plot:
            acq_data = compute_acquisition_grid(agent, dm, args.kappa, res=30)

        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        if spec.schedules:
            spec.apply_schedules(exp_data)
        perf = get_performance(exp_data)
        sched_data = extract_schedule_steps(spec, params) if spec.schedules else None
        state.record("exploration", exp_code, params, perf, schedule=sched_data)

        if args.plot:
            w, s, p, u, c = acq_data
            path = os.path.join(plot_dir, f"03_explore_round_{round_num:02d}.png")
            plot_acquisition(path, X_AXIS, Y_AXIS, w, s, p, u, c,
                             points=state.all_params[:-1],
                             proposed=params,
                             title=f"Exploration \u2014 Round {round_num}",
                             fixed_params=FIXED_DIMS)
            show_plot(path, inline=True)

        dm.update()
        agent.train(dm, validate=False)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploration rounds (incremental)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
