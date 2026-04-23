"""Run baseline experiments (space-filling, no model)."""
import argparse
import json
import os
from typing import Any

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_parameter_space, plot_dimensional_trajectories, plot_convergence, plot_phase_validation, AxisSpec
from visualization.helpers import physics_combined_at
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, run_and_record, combined_score, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, Z_AXIS, FIXED_DIMS, apply_schedule_args,
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

    exp_results: list[tuple[str, dict[str, float], float]] = []
    pw = agent.calibration_system.performance_weights

    for spec in specs:
        exp_code = next_code(state, "baseline")
        exp_data, params, sched_data = run_and_record(dataset, agent, fab, spec, exp_code)
        perf = get_performance(exp_data)
        state.record("baseline", exp_code, params, perf, schedule=sched_data)
        exp_results.append((exp_code, perf, combined_score(perf, pw)))

    # Performance summary: attributes in grey, sys= in green spectrum (capped for readability)
    _D = "\033[2m"
    _R = "\033[0m"
    _N = "\033[38;2;39;39;42m"  # Zinc-800
    scores = [s for _, _, s in exp_results]
    s_min, s_max = min(scores), max(scores)
    s_range = max(s_max - s_min, 1e-6)
    for exp_code, perf, sys_score in exp_results:
        perf_parts = "  ".join(f"{k[:3]}={v:.3f}" for k, v in perf.items())
        # Green spectrum: Emerald-100 (#D1FAE5) → Emerald-500 (#10B981), capped for readability
        t = (sys_score - s_min) / s_range
        r = int(209 - t * 193)  # 209 → 16
        g = int(250 - t * 65)   # 250 → 185
        b = int(229 - t * 100)  # 229 → 129
        _C = f"\033[38;2;{r};{g};{b}m"
        print(f"  {_N}{exp_code}{_R}  {_D}{perf_parts}{_R}  {_C}sys={sys_score:.3f}{_R}")

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

    _D = "\033[2m"
    _R = "\033[0m"

    path = os.path.join(plot_dir, "01_baseline.png")
    print(f"  {_D}Baseline: Ground Truth vs Initial Model{_R}")
    plot_parameter_space(path, X_AXIS, Y_AXIS, waters, speeds,
                         state.all_params, true_grid, pred_grid,
                         schedules=state.schedules, codes=state.all_codes,
                         fixed_params=FIXED_DIMS)
    show_plot(path, inline=args.plot)

    path_3d_params = os.path.join(plot_dir, "01_baseline_3d.png")
    print(f"  {_D}Baseline: Dimensional Trajectories{_R}")
    plot_dimensional_trajectories(
        path_3d_params, X_AXIS, Y_AXIS, "n_layers",
        state.all_params,
        schedules=state.schedules, codes=state.all_codes,
    )
    show_plot(path_3d_params, inline=args.plot)

    # Phase validation plot — show uncertainty topology behind scatter points
    cal = agent.calibration_system
    path_val = os.path.join(plot_dir, "01_phase_validation.png")
    validation_panels: list[tuple] = []

    # Compute uncertainty grid for process/schedule panels (what the optimizer sees)
    unc_grid_data = None
    if cal.last_process_points is not None:
        unc_res = 30
        unc_waters = np.linspace(0.30, 0.50, unc_res)
        unc_speeds = np.linspace(20.0, 60.0, unc_res)
        unc_grid = np.zeros((unc_res, unc_res))
        for i_w, w in enumerate(unc_waters):
            for j_s, spd in enumerate(unc_speeds):
                p = {"water_ratio": w, "print_speed": spd, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
                unc_grid[j_s, i_w] = agent.predict_uncertainty(p, dm)
        unc_grid_data = (unc_waters, unc_speeds, unc_grid, "Blues")

    # Domain axes are now optimized jointly in Process — no separate Domain panel
    if cal.last_process_points is not None:
        validation_panels.append(("Process", X_AXIS, Y_AXIS, cal.last_process_points, None, unc_grid_data))
    if cal.last_schedule_points is not None and cal.last_schedule_exp_ids is not None and cal.last_process_points is not None:
        sched_pts_raw = cal.last_schedule_points
        sched_ids = cal.last_schedule_exp_ids
        sched_dicts: list[dict[str, Any]] = []
        for j, eid in enumerate(sched_ids):
            water = cal.last_process_points[eid].get("water_ratio", 0.4)
            speed_norm = float(sched_pts_raw[j, 0])
            speed = Y_AXIS.bounds[0] + speed_norm * (Y_AXIS.bounds[1] - Y_AXIS.bounds[0])  # type: ignore[index]
            sched_dicts.append({"water_ratio": water, "print_speed": speed})
        validation_panels.append(("Schedule", X_AXIS, Y_AXIS, sched_dicts, sched_ids, unc_grid_data))
    if validation_panels:
        print(f"  {_D}Phase Validation{_R}")
        plot_phase_validation(path_val, validation_panels)
        show_plot(path_val, inline=args.plot)

    # Convergence plot
    conv_history = cal.convergence_history
    if conv_history:
        path_conv = os.path.join(plot_dir, "01_convergence.png")
        print(f"  {_D}Baseline Convergence{_R}")
        plot_convergence(path_conv, conv_history)
        show_plot(path_conv, inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (space-filling)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
