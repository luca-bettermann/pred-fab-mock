"""First-time-right inference call: kappa=0 acquisition → one fabrication."""

import argparse
import json
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session, show_plot_with_header,
    get_performance, effective_weights,
    SPEED_AXIS, CALIB_AXIS, DEFAULT_FIXED,
)
from workflow import with_dimensions


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = effective_weights(config)

    design_intent = json.loads(args.design_intent) if args.design_intent else {}

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 4{_R}{_B} ▸ Inference{_R}")
    intent_str = f"  ·  intent: {design_intent}" if design_intent else ""
    print(f"  {_D}κ=0 (performance-only){intent_str}{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.25)
    agent.train(dm, validate=False)
    current = with_dimensions(state.prev_params) if state.prev_params else None
    spec = agent.acquisition_step(dm, kappa=0.0, current_params=current)

    code = next_code(state, "infer")
    extra = dict(state.prev_params) if state.prev_params else {}
    extra.update(design_intent)
    exp_data, params, _sched = run_and_record(
        dataset, agent, fab, spec, code,
        extra_params=extra, dataset_code="inference",
    )
    params.update(design_intent)
    perf = get_performance(exp_data)
    state.record("inference", code, params, perf)

    score = combined_score(perf, perf_weights or {})
    print(f"  {code:<14s}  combined={score:.3f}")
    for k, v in perf.items():
        print(f"    {k:<24s} {v:.3f}")

    if getattr(args, "plot", False):
        import os
        import numpy as np
        from pred_fab.plotting import plot_performance_radar, plot_inference_result
        from visualization.helpers import physics_combined_at

        all_perfs = [p for _, p in state.perf_history]
        dataset_scores = [combined_score(p, perf_weights) for _, p in state.perf_history]
        dataset_avg = float(sum(dataset_scores) / len(dataset_scores))

        path = os.path.join(plot_dir, "04_inference_radar.png")
        plot_performance_radar(
            path,
            performance=perf,
            dataset_performances=all_perfs,
            weights=perf_weights,
            combined_score=score,
            dataset_combined=dataset_avg,
            exp_code=code,
        )
        show_plot_with_header(path, "Inference: Performance Radar", inline=args.plot)

        # Predicted topology with proposed optimum
        topo_res = 40
        topo_calibs = np.linspace(*CALIB_AXIS.bounds, topo_res)  # type: ignore[arg-type]
        topo_speeds = np.linspace(*SPEED_AXIS.bounds, topo_res)  # type: ignore[arg-type]
        pred_grid = np.zeros((topo_res, topo_res))
        for i, cal in enumerate(topo_calibs):
            for j, spd in enumerate(topo_speeds):
                try:
                    p = agent.predict_performance(
                        {**DEFAULT_FIXED, "print_speed": spd, "calibration_factor": cal}
                    )
                    pred_grid[j, i] = combined_score(p, perf_weights or {})
                except Exception:
                    pred_grid[j, i] = 0.0

        # Physics optimum for reference
        opt_grid_res = 20
        opt_calibs = np.linspace(*CALIB_AXIS.bounds, opt_grid_res)  # type: ignore[arg-type]
        opt_speeds = np.linspace(*SPEED_AXIS.bounds, opt_grid_res)  # type: ignore[arg-type]
        best_opt_score = -1.0
        opt_spd, opt_cal = SPEED_AXIS.bounds[0], CALIB_AXIS.bounds[0]  # type: ignore[index]
        for s in opt_speeds:
            for c in opt_calibs:
                sc = physics_combined_at(s, c, perf_weights)
                if sc > best_opt_score:
                    best_opt_score, opt_spd, opt_cal = sc, s, c

        path_topo = os.path.join(plot_dir, "04_inference_topology.png")
        plot_inference_result(
            path_topo, SPEED_AXIS, CALIB_AXIS, topo_calibs, topo_speeds, pred_grid,
            proposed={"print_speed": params["print_speed"],
                      "calibration_factor": params["calibration_factor"]},
            proposed_score=score,
            optimum={"print_speed": opt_spd, "calibration_factor": opt_cal},
            optimum_score=best_opt_score,
            points=state.all_params,
            fixed_params={k: v for k, v in DEFAULT_FIXED.items()
                          if k not in ("print_speed", "calibration_factor")},
        )
        show_plot_with_header(path_topo, "Inference: Predicted Topology", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-time-right inference")
    parser.add_argument("--design-intent", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--schedule", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
