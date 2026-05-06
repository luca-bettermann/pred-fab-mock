"""Run baseline (Sobol space-filling) experiments — no model required."""

import argparse
import os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session, show_plot_with_header,
    get_performance, effective_weights, SPEED_AXIS, CALIB_AXIS, SLOWDOWN_AXIS, DEFAULT_FIXED,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = effective_weights(config)

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 1{_R}{_B} ▸ Baseline Sampling{_R}")
    print(f"  {_D}{args.n} Sobol experiments — no model yet, space-filling only{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

    specs = agent.baseline_step(n=args.n)

    log: list[tuple[str, dict, dict]] = []
    for spec in specs:
        code = next_code(state, "baseline")
        exp_data, params, _sched = run_and_record(
            dataset, agent, fab, spec, code, dataset_code="baseline",
        )
        perf = get_performance(exp_data)
        state.record("baseline", code, params, perf)
        log.append((code, params, perf))
        score = combined_score(perf, perf_weights or {})
        print(f"  {code:<14s}  combined={score:.3f}")

    # Phase summary
    scores = [combined_score(p, perf_weights or {}) for _, _, p in log]
    print(f"\n  {_D}{'─' * 40}{_R}")
    print(f"  {len(log)} experiments  "
          f"best={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")

    if getattr(args, "plot", False) and log:
        import numpy as np
        from visualization.helpers import evaluate_physics_grid, physics_combined_at
        from pred_fab.plotting import (
            plot_metric_topology, plot_phase_proposals, plot_convergence,
            plot_dimensional_trajectories, plot_parameter_space,
        )

        speeds, calibs, metrics = evaluate_physics_grid(25, perf_weights)
        individual = {k: v for k, v in metrics.items() if k != "combined"}

        path = os.path.join(plot_dir, "01_baseline_topology.png")
        plot_metric_topology(
            path, SPEED_AXIS, CALIB_AXIS, speeds, calibs,
            individual, metrics["combined"],
            combined_label="combined",
            weights=perf_weights,
            fixed_params={k: v for k, v in DEFAULT_FIXED.items()
                          if k not in ("print_speed", "calibration_factor")},
        )
        show_plot_with_header(path, "Baseline: Ground Truth Topology", inline=args.plot)

        # True vs model topology
        dm = agent.create_datamodule(dataset)
        dm.prepare(val_size=0.0)
        agent.train(dm, validate=False)

        topo_res = 40
        topo_speeds = np.linspace(*SPEED_AXIS.bounds, topo_res)  # type: ignore[arg-type]
        topo_calibs = np.linspace(*CALIB_AXIS.bounds, topo_res)  # type: ignore[arg-type]
        true_grid = np.array([
            [physics_combined_at(spd, cal, perf_weights) for cal in topo_calibs]
            for spd in topo_speeds
        ])
        pred_grid = np.zeros_like(true_grid)
        for i, cal in enumerate(topo_calibs):
            for j, spd in enumerate(topo_speeds):
                try:
                    p = agent.predict_performance(
                        {**DEFAULT_FIXED, "print_speed": spd, "calibration_factor": cal}
                    )
                    pred_grid[j, i] = combined_score(p, perf_weights or {})
                except Exception:
                    pred_grid[j, i] = 0.0

        path_ps = os.path.join(plot_dir, "01_baseline_parameter_space.png")
        plot_parameter_space(
            path_ps, SPEED_AXIS, CALIB_AXIS, topo_calibs, topo_speeds,
            state.all_params, true_grid, pred_grid,
            trajectories=state.trajectories, codes=state.all_codes,
            fixed_params={k: v for k, v in DEFAULT_FIXED.items()
                          if k not in ("print_speed", "calibration_factor")},
        )
        show_plot_with_header(path_ps, "Baseline: True vs Model Topology", inline=args.plot)

        # Dimensional trajectories
        path_3d = os.path.join(plot_dir, "01_baseline_trajectories.png")
        plot_dimensional_trajectories(
            path_3d, SPEED_AXIS, SLOWDOWN_AXIS, "n_layers",
            state.all_params,
            trajectories=state.trajectories, codes=state.all_codes,
        )
        show_plot_with_header(path_3d, "Baseline: Dimensional Trajectories", inline=args.plot)

        # Phase validation scatter
        cal = agent.calibration_system
        path_val = os.path.join(plot_dir, "01_phase_validation.png")
        validation_panels: list[tuple] = []
        if cal.last_process_points is not None:
            validation_panels.append(("Process", SPEED_AXIS, CALIB_AXIS, cal.last_process_points, None))
        if validation_panels:
            plot_phase_proposals(path_val, validation_panels)
            show_plot_with_header(path_val, "Phase Validation", inline=args.plot)

        # Convergence
        conv_history = cal.convergence_history
        if conv_history:
            path_conv = os.path.join(plot_dir, "01_convergence.png")
            plot_convergence(path_conv, conv_history)
            show_plot_with_header(path_conv, "Baseline: Convergence", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments (Sobol)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--schedule", action="append", default=[],
                        help="Per-trajectory schedule, repeatable: PARAM:DIM")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
