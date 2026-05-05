"""Compare ground-truth performance vs. prediction-model performance.

Visual comparison is the headline output. With ``--test-set N``, generates
N held-out experiments inline and reports MAE / max-error on them.
"""

import argparse
import os

import numpy as np

from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, show_plot_with_header,
    combined_score, get_performance, effective_weights,
    SPEED_AXIS, CALIB_AXIS, DEFAULT_FIXED,
)
from workflow import with_dimensions, run_and_evaluate


def _evaluate_test_set(
    agent, dataset, fab, n: int, perf_weights: dict[str, float] | None,
) -> None:
    """Generate N held-out test experiments and report MAE/max error."""
    rng = np.random.default_rng(seed=99)
    from schema import PARAM_BOUNDS
    bounds = {code: (lo, hi) for code, lo, hi in PARAM_BOUNDS}

    print(f"\n  Generating {n} test experiments...")
    errors: list[float] = []
    for i in range(n):
        params = with_dimensions({
            code: float(rng.uniform(lo, hi))
            for code, (lo, hi) in bounds.items()
        })
        code = f"analyse_test_{i + 1:02d}"
        if dataset.has_experiment(code):
            exp = dataset.get_experiment(code)
        else:
            exp = run_and_evaluate(dataset, agent, fab, params, code, dataset_code="analyse_test")

        true_perf = get_performance(exp)
        true_score = combined_score(true_perf, perf_weights or {})
        try:
            pred_perf = agent.predict_performance(params)
            pred_score = combined_score(pred_perf, perf_weights or {})
        except Exception:
            pred_score = 0.0
        errors.append(abs(true_score - pred_score))

    mae = float(np.mean(errors))
    print(f"\n  Test-set evaluation ({n} experiments):")
    print(f"    MAE (combined score): {mae:.4f}")
    print(f"    Max error:            {max(errors):.4f}")


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    perf_weights = effective_weights(config)
    plot_dir = ensure_plot_dir()

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    if args.plot:
        from visualization.helpers import evaluate_physics_grid
        from pred_fab.plotting import plot_topology_comparison

        speeds, calibs, phys_metrics = evaluate_physics_grid(30, perf_weights)
        true_grid = phys_metrics["combined"]

        pred_grid = np.zeros_like(true_grid)
        for i, spd in enumerate(speeds):
            for j, cal in enumerate(calibs):
                params = with_dimensions({
                    **DEFAULT_FIXED,
                    "print_speed": float(spd),
                    "calibration_factor": float(cal),
                })
                try:
                    perf = agent.predict_performance(params)
                    pred_grid[i, j] = combined_score(perf, perf_weights or {})
                except Exception:
                    pred_grid[i, j] = 0.0

        path = os.path.join(plot_dir, "05_analysis_topology.png")
        plot_topology_comparison(
            path, SPEED_AXIS, CALIB_AXIS, speeds, calibs,
            {"Ground Truth": true_grid, "Model Prediction": pred_grid},
            fixed_params={k: v for k, v in DEFAULT_FIXED.items()
                          if k not in ("print_speed", "calibration_factor")},
        )
        show_plot_with_header(path, "Analysis: Ground Truth vs Model Prediction", inline=args.plot)

    if getattr(args, "test_set", 0) and args.test_set > 0:
        _evaluate_test_set(agent, dataset, fab, args.test_set, perf_weights)
        config["test_set_n"] = int(args.test_set)
        save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse model: visual + optional test-set numerics")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--test-set", type=int, default=0, dest="test_set")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
