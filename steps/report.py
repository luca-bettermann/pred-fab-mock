"""Generate a visual report for a specific experiment."""
import argparse
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_performance_radar, plot_schedule_detail
from visualization import plot_path_comparison_3d
from steps._common import (
    load_session, rebuild, ensure_plot_dir, show_plot, combined_score,
    N_LAYERS, N_SEGMENTS,
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
    show_plot(path_3d, inline=args.plot)
    print(f"  ✓ Path comparison: {path_3d}")

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
    show_plot(path_radar, inline=args.plot)
    print(f"  ✓ Performance radar: {path_radar}")

    # ── 3. Schedule detail (if applicable) ──
    # Check if this experiment was part of a schedule phase
    if phase in ("schedule", "adaptation"):
        # Collect all schedule experiments' params as pseudo-schedules
        sched_indices = [i for i, p in enumerate(state.all_phases) if p == phase]
        schedules = []
        for si in sched_indices:
            p = state.all_params[si]
            # Build a single-point "schedule" from params (for schedule_detail)
            schedules.append(p)

        # Find which param was likely scheduled (varying across schedule experiments)
        if len(sched_indices) > 1:
            # Detect which parameters vary across the scheduled experiments
            first_p = state.all_params[sched_indices[0]]
            varying = []
            for key in first_p:
                if key in ("n_layers", "n_segments"):
                    continue
                vals = [state.all_params[i].get(key) for i in sched_indices]
                if len(set(float(v) for v in vals if v is not None)) > 1:
                    varying.append(key)

            for param_key in varying:
                # Build per-experiment value list for the schedule detail plot
                sched_data = [{param_key: [state.all_params[i][param_key]]}
                              for i in sched_indices]
                path_sched = os.path.join(report_dir, f"{exp_code}_schedule_{param_key}.png")
                plot_schedule_detail(
                    path_sched,
                    schedules=sched_data,
                    param_key=param_key,
                    title=f"Schedule: {param_key}  ·  {exp_code}",
                )
                show_plot(path_sched, inline=args.plot)
                print(f"  ✓ Schedule ({param_key}): {path_sched}")

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
