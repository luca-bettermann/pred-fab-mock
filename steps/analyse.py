"""Compare ground-truth performance vs. prediction-model performance.

Visual comparison is the headline output (no test set needed). With
`--test-set N`, the step also generates N held-out experiments inline
and reports MAE / max-error of the model on them.
"""
import argparse
import os

import numpy as np

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pred_fab.plotting import plot_topology_comparison, plot_parameter_space_3d
from visualization.helpers import physics_combined_at
from steps._common import (
    load_session, save_session, rebuild, ensure_plot_dir, show_plot_with_header, get_physics_optimum,
    combined_score, compute_local_sensitivity, N_LAYERS, N_SEGMENTS,
    X_AXIS, Y_AXIS, Z_AXIS, FIXED_DIMS,
    generate_test_params, with_dimensions, run_and_evaluate, load_physics_from_session,
)


def _evaluate_test_set(agent, dataset, fab, n: int, perf_weights) -> None:
    """Generate N held-out test experiments inline and report MAE/max error."""
    test_params = generate_test_params(n)
    print(f"\n  Generating {len(test_params)} test experiments...")
    test_codes: list[str] = []
    for i, params in enumerate(test_params):
        code = f"test_{i + 1:02d}"
        test_codes.append(code)
        if dataset.has_experiment(code):
            continue
        params = with_dimensions(params)
        run_and_evaluate(dataset, agent, fab, params, code)

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

    mae = float(np.mean(errors))
    print(f"  Test-set evaluation ({len(test_codes)} experiments):")
    print(f"    MAE (combined score): {mae:.4f}")
    print(f"    Max error:            {max(errors):.4f}")


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    load_physics_from_session(config)
    perf_weights = agent.calibration_system.performance_weights
    plot_dir = ensure_plot_dir()

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    # Visual comparison: ground truth vs. model prediction
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
                              fixed_params=FIXED_DIMS)
    show_plot_with_header(path, "Analysis: Ground Truth vs Model Prediction", inline=args.plot)

    # 3D parameter-space scatter: where the experiments landed across all dims
    path_3d = os.path.join(plot_dir, "05_analysis_parameter_space_3d.png")
    plot_parameter_space_3d(
        path_3d, X_AXIS, Y_AXIS, Z_AXIS,
        state.all_params, codes=state.all_codes,
    )
    show_plot_with_header(path_3d, "Analysis: Experiment Coverage (3D)", inline=args.plot)

    # Optional numerical evaluation on a held-out test set
    if getattr(args, "test_set", 0) and args.test_set > 0:
        _evaluate_test_set(agent, dataset, fab, args.test_set, perf_weights)
        config["test_set_n"] = int(args.test_set)
        save_session(config, state)

    # Sensitivity around the physics optimum
    opt_w, opt_s = get_physics_optimum(perf_weights)
    opt_params = {"water_ratio": opt_w, "print_speed": opt_s,
                  "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
    tunable = agent.calibration_system.get_tunable_params(dm)
    sensitivities = compute_local_sensitivity(agent, opt_params, tunable, perf_weights)

    print(f"\n  Local sensitivity at optimum:")
    for code, val in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        print(f"    {code:<20s}  {val:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse model: visual + optional test-set numerics")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--test-set", type=int, default=0, dest="test_set",
                        help="Generate N test experiments inline and report MAE / max error")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
