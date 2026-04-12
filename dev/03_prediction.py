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

from pred_fab.core import DataModule

N_BASELINE = 20
VAL_SIZE = 0.25
RESOLUTION = 40
PERF_WEIGHTS = PERF_WEIGHTS_DEFAULT

# Feature → performance weight mapping (features predict performance metrics)
FEATURE_WEIGHTS = {
    "path_deviation": PERF_WEIGHTS.get("path_accuracy", 1.0),
    "energy_per_segment": PERF_WEIGHTS.get("energy_efficiency", 1.0),
    "production_rate": PERF_WEIGHTS.get("production_rate", 1.0),
}


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
    val_metrics = {}
    pred_grids = {}
    mlp_dm = None

    for model_type in ["mlp", "rf"]:
        agent, fab, dataset = make_env(f"03_{model_type}", model_type=model_type, verbose=False)
        agent.configure_performance(weights=PERF_WEIGHTS)
        run_baseline(agent, fab, dataset, N_BASELINE)
        dm, val_results = train_models(agent, dataset, val_size=VAL_SIZE)
        model_results[model_type] = _evaluate_model(agent, dm)
        val_metrics[model_type] = val_results or {}
        pred_grids[model_type] = _predict_combined_grid(agent, waters, speeds)
        if model_type == "mlp":
            mlp_dm = dm

    # Console — use val_metrics (R² + R²_adj from framework validation)
    print(f"\n  Prediction quality ({N_BASELINE} baseline, {VAL_SIZE:.0%} held out):")
    print(f"  {'Feature':25s}  {'MLP R²':>8s}  {'MLP R²_adj':>10s}  {'RF R²':>8s}  {'RF R²_adj':>10s}")
    print(f"  {'─' * 67}")
    for feat in val_metrics["mlp"]:
        mlp = val_metrics["mlp"][feat]
        rf = val_metrics["rf"].get(feat, {"r2": 0, "r2_adj": 0})
        print(f"  {feat:25s}  {mlp['r2']:8.3f}  {mlp.get('r2_adj', 0):10.3f}"
              f"  {rf['r2']:8.3f}  {rf.get('r2_adj', 0):10.3f}")

    # Weighted combined R² (feature weights derived from performance weights)
    total_w = sum(FEATURE_WEIGHTS.values())
    for tag in ["mlp", "rf"]:
        r2_comb = sum(FEATURE_WEIGHTS.get(f, 0) * val_metrics[tag][f]["r2"]
                      for f in val_metrics[tag]) / total_w
        r2a_comb = sum(FEATURE_WEIGHTS.get(f, 0) * val_metrics[tag][f].get("r2_adj", val_metrics[tag][f]["r2"])
                       for f in val_metrics[tag]) / total_w
        val_metrics[tag]["_combined"] = {"r2": r2_comb, "r2_adj": r2a_comb}
    mlp_c = val_metrics["mlp"]["_combined"]
    rf_c = val_metrics["rf"]["_combined"]
    print(f"  {'─' * 67}")
    print(f"  {'Combined (weighted)':25s}  {mlp_c['r2']:8.3f}  {mlp_c['r2_adj']:10.3f}"
          f"  {rf_c['r2']:8.3f}  {rf_c['r2_adj']:10.3f}")

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
    _plot_importance_weights(plot_dir, mlp_dm, val_metrics["mlp"])


def _plot_importance_weights(plot_dir: str, dm: DataModule, val_results: dict):
    """Visualize the R²_adj sigmoid importance weighting with real training data.

    Left panel:  sigmoid weight curve with actual experiment scores plotted.
    Right panel: actual per-feature R²_adj − R² gap from validation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.helpers import save_fig

    floor = 0.1
    steepness = 0.8

    # --- Collect real per-experiment combined scores from training split ---
    train_codes = dm.get_split_codes(SplitType.TRAIN)
    exp_scores = []
    for code in train_codes:
        exp = dm.dataset.get_experiment(code)
        perf = exp.performance.get_values_dict()
        exp_scores.append(combined_score(perf, PERF_WEIGHTS))
    exp_scores = np.array(exp_scores)

    s_mean = float(exp_scores.mean())
    s_std = float(exp_scores.std())
    k = steepness / s_std if s_std > 1e-10 else 0.0

    # Sigmoid curve from real mean/std
    perf_range = np.linspace(0.0, 1.0, 200)
    sigmoid_curve = 1.0 / (1.0 + np.exp(-k * (perf_range - s_mean)))
    weights_curve = floor + (1.0 - floor) * sigmoid_curve
    midpoint = (1.0 + floor) / 2.0

    # Per-experiment weights (dots on the curve)
    exp_sigmoid = 1.0 / (1.0 + np.exp(-k * (exp_scores - s_mean)))
    exp_weights = floor + (1.0 - floor) * exp_sigmoid

    # ---- Left panel: sigmoid + real data points ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("R\u00b2_adj Importance Weighting", fontsize=13, fontweight="bold")

    ax1.plot(perf_range, weights_curve, "b-", lw=2, zorder=1)
    ax1.scatter(exp_scores, exp_weights, c="red", s=40, zorder=3,
                edgecolors="darkred", linewidths=0.5, label=f"experiments (n={len(exp_scores)})")
    ax1.axhline(midpoint, color="gray", ls="--", lw=0.8, alpha=0.5, label=f"midpoint = {midpoint:.2f}")
    ax1.axhline(floor, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.axvline(s_mean, color="red", ls="--", lw=1, alpha=0.7, label=f"mean = {s_mean:.3f}")
    ax1.axvspan(s_mean - s_std, s_mean + s_std, alpha=0.06, color="red")
    ax1.fill_between(perf_range, floor, weights_curve, alpha=0.08, color="blue")
    ax1.set_xlabel("Combined Performance Score")
    ax1.set_ylabel("Importance Weight")
    ax1.set_title(f"sigmoid(k\u00b7(perf \u2212 mean)),  k = {steepness}/\u03c3 = {k:.1f}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ---- Right panel: actual per-feature R²_adj gaps ----
    if val_results:
        features = []
        gaps = []
        for feat, m in val_results.items():
            if "r2_adj" in m and "r2" in m:
                features.append(feat)
                gaps.append(m["r2_adj"] - m["r2"])

        if features:
            y_pos = np.arange(len(features))
            colors = ["#2ca02c" if g >= 0 else "#d62728" for g in gaps]
            ax2.barh(y_pos, gaps, height=0.6, color=colors, alpha=0.7)
            for i, (feat, g) in enumerate(zip(features, gaps)):
                sign = 1 if g >= 0 else -1
                ax2.text(g + 0.002 * sign, i, f"{g:+.4f}", va="center",
                         ha="left" if g >= 0 else "right", fontsize=8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features, fontsize=8)
            ax2.invert_yaxis()
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("Gap (R\u00b2_adj \u2212 R\u00b2)")
    ax2.set_title("Actual Validation Gaps")
    max_gap = max(abs(g) for g in gaps) if gaps else 0.05
    margin = max(max_gap * 1.5, 0.02)
    ax2.set_xlim(-margin, margin)
    ax2.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    out = os.path.join(plot_dir, "03_importance_weights.png")
    save_fig(out)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
