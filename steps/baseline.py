"""Run baseline experiments (space-filling, no model)."""
import argparse
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_parameter_space, plot_parameter_space_3d
from visualization import plot_path_comparison_3d
from visualization.helpers import physics_combined_at
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, combined_score, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, Z_AXIS, FIXED_DIMS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    plot_dir = ensure_plot_dir()

    if getattr(args, 'schedule', False):
        agent.configure_schedule(
            "print_speed", "n_layers",
            delta=args.delta, smoothing=args.smoothing,
        )

    agent.console.print_phase_header(1, "Baseline", f"{args.n} experiments")
    specs = agent.baseline_step(n=args.n)

    for spec in specs:
        params = with_dimensions(params_from_spec(spec))
        exp_code = next_code(state, "baseline")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("baseline", exp_code, params, perf)
        agent.console.print_experiment_row(exp_code, params, perf)

    state.prev_params = with_dimensions(params_from_spec(specs[-1]))

    last_params = state.all_params[-1]
    path_3d = os.path.join(plot_dir, "01_path_deviation_3d.png")
    plot_path_comparison_3d(path_3d, fab.camera, last_params, exp_code=state.all_codes[-1])
    show_plot(path_3d, inline=args.plot)

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
    plot_parameter_space_3d(path_3d_params, X_AXIS, Y_AXIS, Z_AXIS,
                             state.all_params, title="Baseline Parameter Space")
    show_plot(path_3d_params, inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (space-filling)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
