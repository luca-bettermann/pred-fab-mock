"""03 — Prediction Model Validation.

Train on baseline data and verify prediction quality for both MLP and RF:
  - R² and MAE per feature on validation set
  - Predicted topology vs ground truth
  - MLP vs RF comparison
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.utils import SplitType
from sensors.physics import N_LAYERS, N_SEGMENTS
from visualization import plot_prediction_scatter, plot_topology_comparison
from visualization.helpers import physics_combined_at, PERF_WEIGHTS_DEFAULT
from shared import make_env, run_baseline, train_models, ensure_plot_dir

N_BASELINE = 20
VAL_SIZE = 0.25
RESOLUTION = 40
PERF_WEIGHTS = PERF_WEIGHTS_DEFAULT


def _evaluate_model(agent, dm):
    """Compute per-feature R² and MAE on validation set."""
    val_batches = dm.get_batches(SplitType.VAL)
    if not val_batches:
        return {}
    X_val = np.concatenate([b[0] for b in val_batches])
    y_val = np.concatenate([b[1] for b in val_batches])

    results = {}
    for model in agent.pred_system.models:
        input_idx = dm.get_input_indices(model.input_parameters + model.input_features)
        out_idx = [dm.output_columns.index(f) for f in model.outputs]
        y_true = dm.denormalize_values(y_val[:, out_idx], model.outputs)
        y_pred = dm.denormalize_values(model.forward_pass(X_val[:, input_idx]), model.outputs)
        for i, feat in enumerate(model.outputs):
            ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            results[feat] = {"r2": r2, "mae": mae, "y_true": y_true[:, i], "y_pred": y_pred[:, i]}
    return results


def _predict_combined_grid(agent, waters, speeds):
    grid = np.zeros((len(speeds), len(waters)))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                total_w = sum(PERF_WEIGHTS.values())
                grid[j, i] = sum(PERF_WEIGHTS.get(k, 0) * float(v)
                                  for k, v in perf.items() if v is not None) / total_w
            except Exception:
                grid[j, i] = 0.0
    return grid


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    plot_dir = ensure_plot_dir()

    waters = np.linspace(0.30, 0.50, RESOLUTION)
    speeds = np.linspace(20.0, 60.0, RESOLUTION)

    # Ground truth
    true_grid = np.array([[physics_combined_at(w, spd) for w in waters] for spd in speeds])

    model_results = {}
    pred_grids = {}

    for model_type in ["mlp", "rf"]:
        agent, fab, dataset = make_env(f"03_{model_type}", model_type=model_type, verbose=False)
        agent.configure(performance_weights=PERF_WEIGHTS)
        run_baseline(agent, fab, dataset, N_BASELINE)
        dm, _ = train_models(agent, dataset, val_size=VAL_SIZE)
        model_results[model_type] = _evaluate_model(agent, dm)
        pred_grids[model_type] = _predict_combined_grid(agent, waters, speeds)

    # Console
    print(f"\n  Prediction quality ({N_BASELINE} baseline, {VAL_SIZE:.0%} held out):")
    print(f"  {'Feature':25s}  {'MLP R²':>8s}  {'MLP MAE':>10s}  {'RF R²':>8s}  {'RF MAE':>10s}")
    print(f"  {'─' * 65}")
    for feat in model_results["mlp"]:
        mlp = model_results["mlp"][feat]
        rf = model_results["rf"].get(feat, {"r2": 0, "mae": 0})
        print(f"  {feat:25s}  {mlp['r2']:8.3f}  {mlp['mae']:10.6f}  {rf['r2']:8.3f}  {rf['mae']:10.6f}")

    # MLP scatter
    out = os.path.join(plot_dir, "03_prediction_accuracy.png")
    plot_prediction_scatter(out, model_results["mlp"], title="MLP Prediction Accuracy")
    print(f"\n  Saved: {out}")

    # Topology: truth vs MLP vs RF
    out = os.path.join(plot_dir, "03_topology_comparison.png")
    plot_topology_comparison(out, waters, speeds,
                              {"Ground Truth": true_grid, "MLP": pred_grids["mlp"], "RF": pred_grids["rf"]},
                              title="Topology: Ground Truth vs MLP vs RF")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
