"""Run baseline experiments (space-filling, no model)."""
import argparse
import json
import os
from typing import Any

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import (
    plot_parameter_space, plot_parameter_space_per_cell, plot_mean_error_topology,
    plot_dimensional_trajectories, plot_convergence, plot_phase_proposals, AxisSpec,
)
from visualization.helpers import physics_combined_at
from sensors.physics import path_deviation as physics_path_deviation
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot_with_header, with_dimensions, params_from_spec, get_performance,
    run_and_evaluate, run_and_record, combined_score, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, Z_AXIS, LAYER_AXIS, SEGMENT_AXIS, FIXED_DIMS, apply_schedule_args,
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

    agent.console.print_phase_header(1, "Baseline", f"{args.n} experiments")
    specs = agent.baseline_step(n=args.n)

    exp_results: list[tuple[str, dict[str, Any], list[dict[str, Any]] | None, dict[str, float], float]] = []
    pw = agent.calibration_system.performance_weights

    for spec in specs:
        exp_code = next_code(state, "baseline")
        exp_data, params, sched_data = run_and_record(
            dataset, agent, fab, spec, exp_code, dataset_code="baseline",
        )
        perf = get_performance(exp_data)
        state.record("baseline", exp_code, params, perf, trajectory=sched_data)
        exp_results.append((exp_code, params, sched_data, perf, combined_score(perf, pw)))

    # Per-experiment summary: process params (with scheduled ranges) + perf scores.
    # Domain axes are skipped — they were already shown in the Domain console block.
    _D = "\033[2m"
    _R = "\033[0m"
    _N = "\033[38;2;39;39;42m"  # Zinc-800
    scores = [s for *_, s in exp_results]
    s_min, s_max = min(scores), max(scores)
    s_range = max(s_max - s_min, 1e-6)
    skip_codes = {"n_layers", "n_segments"}
    for exp_code, params, sched_data, perf, sys_score in exp_results:
        sched_codes: set[str] = set()
        if sched_data:
            for step in sched_data:
                sched_codes.update(step.keys())

        param_parts: list[str] = []
        for code, val in params.items():
            if code in skip_codes:
                continue
            short = code[:3]
            if code in sched_codes and sched_data:
                vals = [float(val)] + [float(step.get(code, val)) for step in sched_data]
                param_parts.append(f"{short}=[{min(vals):.1f}, {max(vals):.1f}]")
            elif isinstance(val, float):
                param_parts.append(f"{short}={val:.3f}")
            else:
                param_parts.append(f"{short}={val}")
        param_str = "  ".join(param_parts)
        perf_parts = "  ".join(f"{k[:3]}={v:.3f}" for k, v in perf.items())
        # Green spectrum: Emerald-100 (#D1FAE5) → Emerald-500 (#10B981), capped for readability
        t = (sys_score - s_min) / s_range
        r = int(209 - t * 193)  # 209 → 16
        g = int(250 - t * 65)   # 250 → 185
        b = int(229 - t * 100)  # 229 → 129
        _C = f"\033[38;2;{r};{g};{b}m"
        print(f"  {_N}{exp_code}{_R}  {_D}{param_str}  {perf_parts}{_R}  {_C}sys={sys_score:.3f}{_R}")

    state.prev_params = with_dimensions(params_from_spec(specs[-1]))

    print()  # blank line before training output
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
                         trajectories=state.trajectories, codes=state.all_codes,
                         fixed_params=FIXED_DIMS)
    show_plot_with_header(path, "Baseline: Ground Truth vs Initial Model", inline=args.plot)

    # Per-cell variant: bypasses eval-aggregation so model bias at one
    # specific (layer, segment) cell is directly visible. Default to the
    # middle cell. The fourth panel shows mean |truth - pred| averaged
    # across all cells, surfacing parameter regions where the model is
    # systematically weak regardless of position.
    mid_layer = N_LAYERS // 2
    mid_seg = N_SEGMENTS // 2
    true_cell_grid = np.array([
        [physics_path_deviation(spd, mid_seg, w, mid_layer) for w in waters]
        for spd in speeds
    ])
    pred_cell_grid = np.zeros_like(true_cell_grid)
    mean_diff_grid = np.zeros_like(true_cell_grid)
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                tensor = agent.pred_system._predict_from_params(  # type: ignore[attr-defined]
                    params={"water_ratio": w, "print_speed": spd,
                            "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
                )
                pred_dev = tensor["path_deviation"]
                pred_cell_grid[j, i] = float(pred_dev[mid_layer, mid_seg])
                # Truth tensor at this (w, spd) for all cells
                true_dev = np.array([
                    [physics_path_deviation(spd, s, w, k) for s in range(N_SEGMENTS)]
                    for k in range(N_LAYERS)
                ])
                mean_diff_grid[j, i] = float(np.mean(np.abs(true_dev - pred_dev)))
            except Exception:
                pred_cell_grid[j, i] = 0.0
                mean_diff_grid[j, i] = 0.0

    cell_path = os.path.join(plot_dir, "01_baseline_per_cell.png")
    cell_label = f"layer={mid_layer}, segment={mid_seg}  ·  path_deviation"
    plot_parameter_space_per_cell(
        cell_path, X_AXIS, Y_AXIS, waters, speeds,
        state.all_params, true_cell_grid, pred_cell_grid,
        cell_label=cell_label,
        trajectories=state.trajectories, codes=state.all_codes,
        fixed_params=FIXED_DIMS,
    )
    show_plot_with_header(cell_path, "Baseline: Per-Cell Comparison", inline=args.plot)

    mean_path = os.path.join(plot_dir, "01_baseline_mean_error.png")
    plot_mean_error_topology(
        mean_path, X_AXIS, Y_AXIS, waters, speeds,
        state.all_params, mean_diff_grid,
        label="Mean |Truth − Pred|  ·  path_deviation (all cells)",
        trajectories=state.trajectories, codes=state.all_codes,
        fixed_params=FIXED_DIMS,
    )
    show_plot_with_header(mean_path, "Baseline: Mean Error Across Cells", inline=args.plot)

    path_3d_params = os.path.join(plot_dir, "01_baseline_3d.png")
    plot_dimensional_trajectories(
        path_3d_params, X_AXIS, Y_AXIS, "n_layers",
        state.all_params,
        trajectories=state.trajectories, codes=state.all_codes,
    )
    show_plot_with_header(path_3d_params, "Baseline: Dimensional Trajectories", inline=args.plot)

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

    if cal.last_domain_values is not None:
        validation_panels.append(("Domain", LAYER_AXIS, SEGMENT_AXIS, cal.last_domain_values, None))

    # Second panel: the final post-Schedule trajectory if scheduling ran,
    # otherwise the static Process points. The pre-Schedule Process panel was
    # removed because its points are stale once Schedule refines them.
    has_schedule = (
        cal.last_trajectory_points is not None
        and cal.last_trajectory_exp_ids is not None
        and cal.last_process_points is not None
    )
    if has_schedule:
        sched_pts_raw = cal.last_trajectory_points
        sched_ids = cal.last_trajectory_exp_ids
        sched_dicts: list[dict[str, Any]] = []
        for j, eid in enumerate(sched_ids):
            water = cal.last_process_points[eid].get("water_ratio", 0.4)  # type: ignore[index]
            speed_norm = float(sched_pts_raw[j, 0])  # type: ignore[index]
            speed = Y_AXIS.bounds[0] + speed_norm * (Y_AXIS.bounds[1] - Y_AXIS.bounds[0])  # type: ignore[index]
            sched_dicts.append({"water_ratio": water, "print_speed": speed})
        validation_panels.append(("Process + Schedule", X_AXIS, Y_AXIS, sched_dicts, sched_ids, unc_grid_data))
    elif cal.last_process_points is not None:
        validation_panels.append(("Process", X_AXIS, Y_AXIS, cal.last_process_points, None, unc_grid_data))

    if validation_panels:
        plot_phase_proposals(path_val, validation_panels)
        show_plot_with_header(path_val, "Phase Validation", inline=args.plot)

    # Convergence plot
    conv_history = cal.convergence_history
    if conv_history:
        path_conv = os.path.join(plot_dir, "01_convergence.png")
        plot_convergence(path_conv, conv_history)
        show_plot_with_header(path_conv, "Baseline: Convergence", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (space-filling)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
