"""Evaluate the prediction model on the test set and show sensitivity analysis."""
import argparse
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import (
    load_session, rebuild, ensure_plot_dir, show_plot, get_physics_optimum,
    combined_score, compute_local_sensitivity, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, FIXED_DIMS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    perf_weights = agent.calibration_system.performance_weights
    plot_dir = ensure_plot_dir()

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    test_codes = [c for c in dataset.get_populated_experiment_codes() if c.startswith("test_")]
    if not test_codes:
        print("  No test set found. Run 'uv run cli.py test-set --n 20' first.")
        return

    print(f"\n  Evaluating model on {len(test_codes)} test experiments:")
    errors = []
    for code in sorted(test_codes):
        exp = dataset.get_experiment(code)
        params = exp.parameters.get_values_dict()
        true_perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        true_score = combined_score(true_perf, perf_weights)

        try:
            pred_perf = agent.predict_performance(params)
            pred_score = combined_score(pred_perf, perf_weights)
        except Exception:
            pred_score = 0.0

        errors.append(abs(true_score - pred_score))

    mae = np.mean(errors)
    print(f"    MAE (combined score): {mae:.4f}")
    print(f"    Max error:            {max(errors):.4f}")

    from pred_fab.plotting import plot_topology_comparison
    from visualization.helpers import physics_combined_at
    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    true_grid = np.array([[physics_combined_at(w, spd, perf_weights) for w in waters] for spd in speeds])
    pred_grid = np.zeros_like(true_grid)
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(perf, perf_weights)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "05_analysis_topology.png")
    plot_topology_comparison(path, X_AXIS, Y_AXIS, waters, speeds,
                              {"Ground Truth": true_grid, "Model Prediction": pred_grid},
                              title="Model Analysis on Test Set",
                              fixed_params=FIXED_DIMS)
    show_plot(path, inline=args.plot)

    opt_w, opt_s = get_physics_optimum(perf_weights)
    opt_params = {"water_ratio": opt_w, "print_speed": opt_s,
                  "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}

    tunable = agent.calibration_system.get_tunable_params(dm)
    sensitivities = compute_local_sensitivity(agent, opt_params, tunable, perf_weights)

    print(f"\n  Local sensitivity at optimum:")
    for code, val in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        print(f"    {code:<20s}  {val:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on test set + sensitivity analysis")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
