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
from pred_fab import combined_score
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
                grid[j, i] = combined_score(perf, PERF_WEIGHTS)
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

    # Importance weighting validation: show how weights map to performance
    _plot_importance_weights(plot_dir, model_results["mlp"])


def _plot_importance_weights(plot_dir: str, model_results: dict):
    """Visualize the R²_adj sigmoid importance weighting scheme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.helpers import save_fig

    floor = 0.1
    steepness = 0.8
    perf_range = np.linspace(0.0, 1.0, 200)

    # Show for a representative mean and std
    mean, std = 0.5, 0.15
    k = steepness / std
    sigmoid = 1.0 / (1.0 + np.exp(-k * (perf_range - mean)))
    weights = floor + (1.0 - floor) * sigmoid
    midpoint = (1.0 + floor) / 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("R\u00b2_adj Importance Weighting", fontsize=13, fontweight="bold")

    # Left: sigmoid weight function
    ax1.plot(perf_range, weights, "b-", lw=2)
    ax1.axhline(midpoint, color="gray", ls="--", lw=0.8, alpha=0.5, label=f"midpoint = {midpoint:.2f}")
    ax1.axhline(floor, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.axvline(mean, color="red", ls="--", lw=1, alpha=0.7, label=f"mean = {mean}")
    ax1.axvspan(mean - std, mean + std, alpha=0.06, color="red")
    ax1.fill_between(perf_range, floor, weights, alpha=0.08, color="blue")
    ax1.set_xlabel("Combined Performance Score")
    ax1.set_ylabel("Importance Weight")
    ax1.set_title(f"sigmoid(k\u00b7(perf \u2212 mean)), k = {steepness}/std")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Right: interpretation
    scenarios = {
        "Uniform quality\n(R\u00b2_adj \u2248 R\u00b2)": (0.0, "#888888"),
        "Better above avg\n(R\u00b2_adj > R\u00b2)": (0.05, "#2ca02c"),
        "Worse above avg\n(R\u00b2_adj < R\u00b2)": (-0.05, "#d62728"),
    }
    y_pos = 0
    for label, (gap, color) in scenarios.items():
        ax2.barh(y_pos, gap, height=0.6, color=color, alpha=0.7)
        ax2.text(gap + 0.002 * np.sign(gap), y_pos, label, va="center",
                 ha="left" if gap >= 0 else "right", fontsize=9)
        y_pos += 1
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("Gap (R²_adj - R²)")
    ax2.set_title("Interpretation")
    ax2.set_xlim(-0.08, 0.08)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.2, axis="x")

    out = os.path.join(plot_dir, "03_importance_weights.png")
    save_fig(out)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
