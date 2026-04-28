"""Generate a visual report for a specific experiment."""
import argparse
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_performance_radar, plot_dimensional_trajectories, AxisSpec
from visualization import plot_path_comparison_3d
from steps._common import (
    load_session, rebuild, ensure_plot_dir, show_plot_with_header, combined_score,
    N_LAYERS, N_SEGMENTS, X_AXIS, Y_AXIS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    exp_code = args.exp_code

    # Find experiment by code
    if exp_code not in state.all_codes:
        print(f"  Error: experiment '{exp_code}' not found.")
        print(f"  Available: {', '.join(state.all_codes)}")
        return

    idx = state.all_codes.index(exp_code)
    params = state.all_params[idx]
    phase = state.all_phases[idx]
    _, perf = state.perf_history[idx]
    perf_weights: dict[str, float] = config.get("performance_weights") or {
        "path_accuracy": 1.0, "energy_efficiency": 1.0, "production_rate": 1.0,
    }

    agent, dataset, fab = rebuild(config)
    plot_dir = ensure_plot_dir()
    report_dir = os.path.join(plot_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    print(f"\n  Report: {exp_code}  (phase: {phase})")
    print(f"  {'─' * 50}")

    # ── 1. As-printed vs as-designed 3D ──
    path_3d = os.path.join(report_dir, f"{exp_code}_path_3d.png")
    plot_path_comparison_3d(path_3d, fab.camera, params, exp_code=exp_code)
    show_plot_with_header(path_3d, f"Report ({exp_code}): Path Comparison (3D)", inline=args.plot)

    # ── 2. Performance radar ──
    # Collect dataset performances for comparison
    all_perfs = [p for _, p in state.perf_history]
    score = combined_score(perf, perf_weights)
    dataset_scores = [combined_score(p, perf_weights) for _, p in state.perf_history]
    dataset_avg = float(np.mean(dataset_scores))

    path_radar = os.path.join(report_dir, f"{exp_code}_performance.png")
    plot_performance_radar(
        path_radar,
        performance=perf,
        dataset_performances=all_perfs,
        weights=perf_weights,
        combined_score=score,
        dataset_combined=dataset_avg,
        exp_code=exp_code,
    )
    show_plot_with_header(path_radar, f"Report ({exp_code}): Performance Radar", inline=args.plot)

    # ── 3. Dimensional trajectories (highlighted) ──
    path_traj = os.path.join(report_dir, f"{exp_code}_trajectories.png")
    plot_dimensional_trajectories(
        path_traj, X_AXIS, Y_AXIS, "n_layers",
        state.all_params,
        schedules=state.schedules, codes=state.all_codes,
        highlight=exp_code,
    )
    show_plot_with_header(path_traj, f"Report ({exp_code}): Dimensional Trajectories", inline=args.plot)

    print(f"  {'─' * 50}")
    print(f"  Combined score: {score:.3f}  (dataset avg: {dataset_avg:.3f})")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual report for an experiment")
    parser.add_argument("exp_code", type=str, help="Experiment code (e.g. base_01, explore_03)")
    parser.add_argument("--plot", action="store_true", help="Show plots inline in terminal")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
