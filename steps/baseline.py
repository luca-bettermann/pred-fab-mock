"""Run baseline experiments (space-filling, no model)."""
import argparse
import json
import os
from typing import Any

import numpy as np

from pred_fab.plotting import (
    plot_parameter_space, plot_parameter_space_per_cell, plot_mean_error_topology,
    plot_dimensional_trajectories, plot_convergence, plot_phase_proposals,
)
from visualization.helpers import physics_combined_at
from sensors.physics import path_deviation as physics_path_deviation
from steps._common import (
    run_step,
    load_session, save_session, rebuild, ensure_plot_dir, next_code,
    show_plot_with_header, with_dimensions, params_from_spec, get_performance,
    run_and_record, combined_score, predict_score_grid, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, LAYER_AXIS, SEGMENT_AXIS, FIXED_DIMS, apply_schedule_args,
    _D, _R,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    plot_dir = ensure_plot_dir()

    apply_schedule_args(agent, args, config)

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(1, "Baseline", f"{args.n} experiments")
    specs = agent.discovery_step(n=args.n)

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

    waters, speeds, pred_grid = predict_score_grid(agent, pw)
    true_grid = np.array([[physics_combined_at(w, spd, pw) for w in waters] for spd in speeds])

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
    grid_params = [
        {"water_ratio": float(w), "print_speed": float(spd),
         "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
        for spd in speeds for w in waters
    ]
    grid_preds = agent.pred_system._predict_from_params_tensor(grid_params)  # type: ignore[attr-defined]
    for j, spd in enumerate(speeds):
        for i, w in enumerate(waters):
            t = grid_preds[j * len(waters) + i]["path_deviation"]
            pred_dev = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            pred_cell_grid[j, i] = float(pred_dev[mid_layer, mid_seg])
            true_dev = np.array([
                [physics_path_deviation(spd, s, w, k) for s in range(N_SEGMENTS)]
                for k in range(N_LAYERS)
            ])
            mean_diff_grid[j, i] = float(np.mean(np.abs(true_dev - pred_dev)))

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

    # Compute evidence grid for process/schedule panels (what the optimizer sees)
    unc_grid_data = None
    if cal.last_process_points is not None:
        unc_waters, unc_speeds, _density, evidence_grid = cal.compute_evidence_grids(
            X_AXIS.key, Y_AXIS.key, X_AXIS.bounds, Y_AXIS.bounds,
            fixed_params=dict(FIXED_DIMS), resolution=30,
        )
        unc_grid_data = (unc_waters, unc_speeds, evidence_grid, "Blues")

    if cal.last_domain_values is not None:
        validation_panels.append(("Domain", LAYER_AXIS, SEGMENT_AXIS, cal.last_domain_values, None))

    # Second panel: the post-Schedule trajectory when scheduling ran, else the static Process points.
    has_schedule = (
        cal.last_trajectory_points is not None
        and cal.last_trajectory_exp_ids is not None
        and cal.last_process_points is not None
    )
    if has_schedule:
        sched_pts_raw = cal.last_trajectory_points
        sched_ids: list[Any] = []
        sched_dicts: list[dict[str, Any]] = []
        for j, eid in enumerate(cal.last_trajectory_exp_ids or []):
            water = cal.last_process_points[eid].get("water_ratio")  # type: ignore[index]
            if water is None:
                print(f"  Warning: schedule point {eid} missing water_ratio; skipping in validation plot")
                continue
            speed_norm = float(sched_pts_raw[j, 0])  # type: ignore[index]
            speed = Y_AXIS.bounds[0] + speed_norm * (Y_AXIS.bounds[1] - Y_AXIS.bounds[0])  # type: ignore[index]
            sched_ids.append(eid)
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=5, help="Number of experiments")
    parser.add_argument("--plot", action="store_true", help="Show plots inline")
    parser.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                        help="Override the configured schedule (e.g. print_speed:n_layers). Repeatable.")
    parser.add_argument("--design-intent", type=str, default=None,
                        help="JSON: fix parameters (required for schedule). Example: '{\"n_layers\":5}'")


if __name__ == "__main__":
    run_step(__doc__, add_arguments, run)
