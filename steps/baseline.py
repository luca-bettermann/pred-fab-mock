"""Run baseline experiments (space-filling, no model)."""
import argparse
import json
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_parameter_space, plot_dimensional_trajectories, plot_convergence
from visualization.helpers import physics_combined_at
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, combined_score, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, Z_AXIS, FIXED_DIMS, apply_schedule_args,
    extract_schedule_steps,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    plot_dir = ensure_plot_dir()

    if getattr(args, 'iterations', None) is not None:
        agent.calibration_system.de_maxiter = args.iterations

    apply_schedule_args(agent, args)

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(1, "Baseline", f"{args.n} experiments")
    specs = agent.baseline_step(n=args.n)

    for spec in specs:
        params = with_dimensions(params_from_spec(spec))
        exp_code = next_code(state, "baseline")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        # Extract per-step schedule if present
        sched_data = None
        if spec.schedules:
            sched_data = extract_schedule_steps(spec, params)
        state.record("baseline", exp_code, params, perf, schedule=sched_data)
        agent.console.print_experiment_row(exp_code, params, perf)

        # Show schedule table if present
        if sched_data and len(sched_data) > 1:
            cal = agent.calibration_system
            tunable_codes = set(cal.schedule_configs.keys())
            agent.console.print_schedule_table(
                sched_data, tunable_codes, cal.schedule_configs,
            )

    state.prev_params = with_dimensions(params_from_spec(specs[-1]))

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    pw = agent.calibration_system.performance_weights
    true_grid = np.array([[physics_combined_at(w, spd, pw) for w in waters] for spd in speeds])
    pred_grid = np.zeros_like(true_grid)
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(perf, pw)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "01_baseline.png")
    plot_parameter_space(path, X_AXIS, Y_AXIS, waters, speeds,
                         state.all_params, true_grid, pred_grid,
                         fixed_params=FIXED_DIMS)
    show_plot(path, inline=args.plot)

    path_3d_params = os.path.join(plot_dir, "01_baseline_3d.png")
    plot_dimensional_trajectories(
        path_3d_params, X_AXIS, Y_AXIS, "n_layers",
        state.all_params,
        schedules=state.schedules, codes=state.all_codes,
        title="Baseline Parameter Space",
    )
    show_plot(path_3d_params, inline=args.plot)

    # Convergence plot
    conv_history = agent.calibration_system.convergence_history
    if conv_history:
        path_conv = os.path.join(plot_dir, "01_convergence.png")
        plot_convergence(path_conv, conv_history, title="Baseline Convergence")
        show_plot(path_conv, inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (space-filling)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
