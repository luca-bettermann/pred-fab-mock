"""Run exploration rounds (incremental -- can be called multiple times)."""
import argparse
import json
import os

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_acquisition, plot_convergence
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot_with_header, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, run_and_record, compute_acquisition_grid,
    X_AXIS, Y_AXIS, FIXED_DIMS, apply_schedule_args,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    plot_dir = ensure_plot_dir()

    if getattr(args, 'iterations', None) is not None:
        agent.calibration_system.de_maxiter = args.iterations

    apply_schedule_args(agent, args, config)

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    n_existing = len([p for p in state.all_phases if p == "exploration"])
    total_after = n_existing + args.n

    agent.console.print_phase_header(2, "Exploration",
                                      f"rounds {n_existing+1}..{total_after} (\u03ba={args.kappa})")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=args.validate)

    all_convergence: dict[str, list[float]] = {}

    for i in range(args.n):
        round_num = n_existing + i + 1
        # Pass current_params so schedule dimensions are resolved
        current = with_dimensions(state.prev_params) if state.prev_params else None
        spec = agent.acquisition_step(dm, kappa=args.kappa, current_params=current)
        exp_code = next_code(state, "explore")

        if args.plot:
            acq_data = compute_acquisition_grid(agent, dm, args.kappa, res=30)

        exp_data, params, sched_data = run_and_record(
            dataset, agent, fab, spec, exp_code, extra_params=state.prev_params,
        )
        perf = get_performance(exp_data)
        state.record("exploration", exp_code, params, perf, schedule=sched_data)

        if args.plot:
            w, s, p, u, c = acq_data
            path = os.path.join(plot_dir, f"03_explore_round_{round_num:02d}.png")
            plot_acquisition(path, X_AXIS, Y_AXIS, w, s, p, u, c,
                             points=state.all_params[:-1],
                             proposed=params,
                             schedules=state.schedules, codes=state.all_codes[:-1],
                             fixed_params=FIXED_DIMS)
            show_plot_with_header(
                path, f"Exploration: Round {round_num} (κ={args.kappa})", inline=True
            )

        # Collect convergence for this round
        conv = agent.calibration_system.convergence_history
        for label, hist in conv.items():
            all_convergence[f"Round {round_num} ({label})"] = hist

        dm.update()
        agent.train(dm, validate=False)

    # Convergence plot across all rounds
    if all_convergence:
        path_conv = os.path.join(plot_dir, "03_convergence.png")
        plot_convergence(path_conv, all_convergence)
        show_plot_with_header(path_conv, "Exploration: Convergence", inline=args.plot)

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
