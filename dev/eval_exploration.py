"""Prediction model quality benchmark: baseline-only vs. exploration-guided sampling.

Research question (from ADVEI paper):
  How does the data efficiency of exploration-guided sampling compare to
  grid-search/random sampling as a function of dataset size?

Approach:
  1. Generate a fixed test dataset by uniform grid over the parameter space.
  2. Train on baseline-only data — record R² and R²_adj on test set.
  3. Add exploration experiments one-by-one (per optimizer) — record metrics.
  4. Train on random LHS (n=20) — record metrics for comparison.
  5. Plot learning curves (per-feature R² and R²_adj) and forward passes.

Two metrics reported:
  - R²: standard coefficient of determination on a uniform test set.
  - R²_adj: importance-weighted R² that up-weights high-importance samples.
    weight_i = alpha + (1 − alpha) · importance_i
    Gap interpretation (R²_adj − R²):
      gap > 0 → high-importance samples predicted better (exploration working)
      gap < 0 → high-importance samples predicted worse
    importance_i = max(perf_true_i, perf_pred_i) by default (symmetric),
    or perf_true_i only for cross-method comparison (symmetric=False).

Uses the same schema, agent, sensors, and models as main.py.
"""

import os
import sys
import shutil
import warnings
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── make local modules importable when running from dev/ ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab import Optimizer
from pred_fab.core import Dataset
from pred_fab.utils import Metrics, SplitType  # type: ignore[attr-defined]

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec

# ── Configuration ─────────────────────────────────────────────────────────────
N_BASELINE    = 5      # baseline experiments (LHS)
N_EXPLORE     = 10     # exploration rounds
N_TEST        = 32     # test experiments (uniform grid, never used for training)
N_RANDOM      = 15     # random-LHS baseline (same total as baseline + explore)
W_EXPLORE     = 0.7    # exploration weight for UCB acquisition (not used in evaluation)
ALPHA         = 0.0    # importance weight floor for R²_adj (0=pure importance, 1=standard R²)
SYMMETRIC     = True   # importance = max(true, pred); False = true only
MATERIAL      = "clay"  # fixed material for mock
MODEL_TYPE    = "mlp"   # "mlp" or "rf" — prediction model architecture
PERF_WEIGHTS: dict[str, float] = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
CAL_BOUNDS    = {"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)}
BOUNDARY_BUFFER = (0.10, 0.8, 2.0)  # (extent, strength, exponent) — penalise edge proposals

# Maps each predicted feature to its corresponding performance attribute.
FEATURE_PERF_MAP: dict[str, str] = {
    "path_deviation":    "path_accuracy",
    "energy_per_segment": "energy_efficiency",
    "production_rate":   "production_rate",
}

DATA_ROOT     = "./dev/data"
PLOT_DIR      = "./dev/plots"


# ── Types ────────────────────────────────────────────────────────────────────

# Per-step metrics: {feature_name: {"r2": float, "r2_adj": float}}
StepMetrics = dict[str, dict[str, float]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _weighted_summary(metrics: StepMetrics, key: str) -> float:
    """Compute performance-weighted summary of a metric across features."""
    total_w, weighted_sum = 0.0, 0.0
    for feat, perf_name in FEATURE_PERF_MAP.items():
        w = PERF_WEIGHTS.get(perf_name, 0.0)
        feat_metrics = metrics.get(feat, {})
        total_w += w
        weighted_sum += w * feat_metrics.get(key, 0.0)
    return weighted_sum / total_w if total_w > 0 else 0.0


def _make_fresh_env(
    data_root: str,
    boundary_buffer: tuple[float, float, float] | None = None,
) -> tuple[Any, Any, Any, Dataset]:
    """Build a fresh agent+dataset for one workflow run."""
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    schema = build_schema(root_folder=data_root)
    fab    = FabricationSystem(CameraSystem(), EnergySensor())
    agent  = build_agent(schema, fab.camera, fab.energy, model_type=MODEL_TYPE)
    configure_kwargs: dict[str, Any] = dict(
        bounds=CAL_BOUNDS, performance_weights=PERF_WEIGHTS,
        fixed_params={"material": MATERIAL},
    )
    if boundary_buffer is not None:
        configure_kwargs["boundary_buffer"] = boundary_buffer
    agent.configure(**configure_kwargs)
    dataset = Dataset(schema=schema)
    agent.logger.set_console_output(False)
    return agent, fab, schema, dataset


def _run_and_evaluate(
    dataset: Dataset,
    agent: Any,
    fab: "FabricationSystem",
    params: dict[str, Any],
    exp_code: str,
) -> Any:
    """Run one fabrication experiment and evaluate features + performance."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data


def _with_dimensions(params: dict[str, Any], fab: "FabricationSystem") -> dict[str, Any]:
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}


def _build_test_dataset(fab: "FabricationSystem") -> list[dict[str, Any]]:
    """Generate N_TEST parameter combinations on a jittered uniform grid."""
    designs   = ["A", "B"]
    materials = [MATERIAL]
    n_per_combo = N_TEST // (len(designs) * len(materials))

    test_params: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed=999)

    for design in designs:
        for material in materials:
            waters = np.linspace(0.31, 0.49, n_per_combo)
            speeds = np.linspace(21.0, 59.0, n_per_combo)
            waters += rng.uniform(-0.005, 0.005, n_per_combo)
            speeds += rng.uniform(-0.5,   0.5,   n_per_combo)
            for w, spd in zip(waters, speeds):
                p = {"design": design, "material": material,
                     "water_ratio": float(np.clip(w, 0.30, 0.50)),
                     "print_speed": float(np.clip(spd, 20.0, 60.0))}
                test_params.append(_with_dimensions(p, fab))

    print(f"  Test set: {len(test_params)} experiments across "
          f"{len(set((p['design'],p['material']) for p in test_params))} combos")
    return test_params


def _run_test_dataset(
    fab: "FabricationSystem",
    agent: Any,
    test_root: str,
    test_params: list[dict[str, Any]],
) -> Dataset:
    """Populate and evaluate all test experiments into a SEPARATE dataset."""
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
    test_schema = build_schema(root_folder=test_root)
    test_dataset = Dataset(schema=test_schema)
    for i, params in enumerate(test_params):
        code = f"test_{i+1:03d}"
        exp_data = test_dataset.create_experiment(code, parameters=params)
        fab.run_experiment(params)
        agent.evaluate(exp_data)
        test_dataset.save_experiment(code)
    return test_dataset


def _compute_combined_performance(perf: dict[str, Any]) -> float:
    """Combined performance = weighted average using PERF_WEIGHTS, normalized to [0,1]."""
    total_w = sum(PERF_WEIGHTS.values())
    score = sum(PERF_WEIGHTS.get(k, 0.0) * float(v) for k, v in perf.items() if v is not None)
    return score / total_w if total_w > 0 else 0.0


def _compute_metrics_on_test(
    agent: Any,
    train_dm: Any,
    test_dataset: Dataset,
    test_params: list[dict[str, Any]],
) -> StepMetrics:
    """Evaluate R² and R²_adj per feature on the fixed test set.

    Returns {feature_name: {"r2": float, "r2_adj": float}}.
    """
    from pred_fab.core import DataModule

    # Build a test DataModule with the training normalization
    test_dm = DataModule(dataset=test_dataset)
    test_dm.initialize(
        input_parameters=agent.pred_system.get_system_input_parameters(),
        input_features=agent.pred_system.get_system_input_features(),
        output_columns=agent.pred_system.get_system_outputs(),
    )
    test_codes = list(test_dataset._experiments.keys())
    test_dm.set_split_codes(train_codes=[], val_codes=[], test_codes=test_codes)
    test_dm.set_normalization_state(train_dm.get_normalization_state())

    batches = test_dm.get_batches(SplitType.TEST)
    if not batches:
        return {}
    X_list, y_list = zip(*batches)
    X_test = np.concatenate(X_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)

    # Compute importance per sample
    importance = np.zeros(len(X_test), dtype=float)
    for idx, (code, params) in enumerate(zip(test_codes, test_params)):
        exp = test_dataset.get_experiment(code)
        true_perf = exp.performance.get_values_dict()
        true_combined = _compute_combined_performance(true_perf)

        if SYMMETRIC:
            try:
                pred_perf = agent.predict_performance(params)
                pred_combined = _compute_combined_performance(pred_perf)
            except Exception:
                pred_combined = 0.0
            importance[idx] = max(true_combined, pred_combined)
        else:
            importance[idx] = true_combined

    # Per-feature R² and R²_adj
    results: StepMetrics = {}
    for model in agent.pred_system.models:
        input_indices = train_dm.get_input_indices(
            model.input_parameters + model.input_features
        )
        out_indices = [train_dm.output_columns.index(f) for f in model.outputs]

        y_true_norm = y_test[:, out_indices]
        y_true = test_dm.denormalize_values(y_true_norm, model.outputs)

        y_pred_norm = model.forward_pass(X_test[:, input_indices])
        y_pred = test_dm.denormalize_values(y_pred_norm, model.outputs)

        for i, feat in enumerate(model.outputs):
            adj = Metrics.calculate_adjusted_r2(
                y_true[:, i], y_pred[:, i], importance,
                alpha=ALPHA, symmetric=SYMMETRIC,
            )
            results[feat] = {"r2": adj["r2"], "r2_adj": adj["r2_adj"]}

    return results


# ── Workflow A: Baseline only ─────────────────────────────────────────────────

def run_baseline_workflow(
    fab: "FabricationSystem",
    test_dataset: Dataset,
    test_params: list[dict[str, Any]],
    n_baseline: int,
) -> StepMetrics:
    """Train on n_baseline LHS experiments, return per-feature metrics on test set."""
    tag = f"baseline{n_baseline}"
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + f"_{tag}")
    specs = agent.baseline_step(n=n_baseline)
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"baseline_{i+1:02d}")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    metrics = _compute_metrics_on_test(agent, dm, test_dataset, test_params)
    shutil.rmtree(DATA_ROOT + f"_{tag}", ignore_errors=True)
    return metrics


# ── Workflow B: Exploration-guided ────────────────────────────────────────────

def run_exploration_workflow(
    fab: "FabricationSystem",
    test_dataset: Dataset,
    test_params: list[dict[str, Any]],
    optimizer: Optimizer = Optimizer.LBFGSB,
    boundary_buffer: tuple[float, float, float] | None = None,
    tag_suffix: str = "",
) -> tuple[list[int], list[StepMetrics], list[int]]:
    """Baseline + incremental exploration.

    Returns (training_sizes, metrics_per_step, cumulative_nfev_per_step).
    """
    tag = f"_explore_{optimizer.value}{tag_suffix}"
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + tag, boundary_buffer=boundary_buffer)
    agent.configure(optimizer=optimizer)

    # Phase 1: Baseline
    specs = agent.baseline_step(n=N_BASELINE)
    prev_params: dict[str, Any] = {}
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"baseline_{i+1:02d}")
        prev_params = params

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    m0 = _compute_metrics_on_test(agent, dm, test_dataset, test_params)
    training_sizes = [N_BASELINE]
    metrics_history = [m0]
    nfev_cumulative = [0]
    r2_sum = _weighted_summary(m0, "r2")
    adj_sum = _weighted_summary(m0, "r2_adj")
    print(f"    n={N_BASELINE}: R²={r2_sum:.3f}  R²_adj={adj_sum:.3f}  |  nfev=0")

    # Phase 2: Exploration rounds
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, w_explore=W_EXPLORE)
        nfev_step = agent.last_opt_nfev
        params = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"explore_{i+1:02d}")
        dm.update()
        agent.train(dm, validate=False)
        prev_params = params

        n_total = N_BASELINE + i + 1
        cumulative = nfev_cumulative[-1] + nfev_step
        m = _compute_metrics_on_test(agent, dm, test_dataset, test_params)
        training_sizes.append(n_total)
        metrics_history.append(m)
        nfev_cumulative.append(cumulative)
        r2_s = _weighted_summary(m, "r2")
        adj_s = _weighted_summary(m, "r2_adj")
        print(f"    n={n_total}: R²={r2_s:.3f}  R²_adj={adj_s:.3f}  |  nfev_step={nfev_step}, cumulative={cumulative}")

    shutil.rmtree(DATA_ROOT + tag, ignore_errors=True)
    return training_sizes, metrics_history, nfev_cumulative


# ── Workflow C: Random sampling baseline ─────────────────────────────────────

def run_random_workflow(
    fab: "FabricationSystem",
    test_dataset: Dataset,
    test_params: list[dict[str, Any]],
    n_total: int,
    eval_from: int = 4,
) -> tuple[list[int], list[StepMetrics]]:
    """Train on 1..n_total random LHS experiments, evaluating incrementally.

    Returns (training_sizes, metrics_per_step) starting from *eval_from*.
    """
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + "_random")
    specs = agent.baseline_step(n=n_total)

    training_sizes: list[int] = []
    metrics_history: list[StepMetrics] = []

    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"random_{i+1:02d}")

        n = i + 1
        if n < eval_from:
            continue

        dm = agent.create_datamodule(dataset)
        dm.prepare(val_size=0.0)
        agent.train(dm, validate=False)

        m = _compute_metrics_on_test(agent, dm, test_dataset, test_params)
        training_sizes.append(n)
        metrics_history.append(m)
        r2_s = _weighted_summary(m, "r2")
        adj_s = _weighted_summary(m, "r2_adj")
        print(f"    n={n}: R²={r2_s:.3f}  R²_adj={adj_s:.3f}")

    shutil.rmtree(DATA_ROOT + "_random", ignore_errors=True)
    return training_sizes, metrics_history


# ── Plotting ──────────────────────────────────────────────────────────────────

# (label, sizes, metrics, fmt, color)
SeriesData = tuple[str, list[int], list[StepMetrics], str, str]


def plot_learning_curves(
    series: list[SeriesData],
    random_sizes: list[int],
    random_metrics: list[StepMetrics],
    filename: str = "learning_curves.png",
    title: str = "Learning Curves: Exploration vs. Random Sampling",
) -> None:
    """Plot R² and R²_adj learning curves for all methods side by side."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _summaries(metrics: list[StepMetrics], key: str) -> list[float]:
        return [_weighted_summary(m, key) for m in metrics]

    # Collect all y-values
    all_r2: list[float] = _summaries(random_metrics, "r2")
    all_adj: list[float] = _summaries(random_metrics, "r2_adj")
    all_sizes = list(random_sizes)
    for label, sizes, metrics, fmt, color in series:
        all_r2 += _summaries(metrics, "r2")
        all_adj += _summaries(metrics, "r2_adj")
        all_sizes += sizes

    x_min = min(all_sizes, default=5) - 0.5
    x_max = max(all_sizes, default=10) + 0.5

    fig, (ax_r2, ax_adj) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax, key, ax_title, y_vals in [
        (ax_r2, "r2", "R² (uniform weighting)", all_r2),
        (ax_adj, "r2_adj", "R²_adj (importance weighting)", all_adj),
    ]:
        # Random baseline
        ax.plot(random_sizes, _summaries(random_metrics, key), "^-",
                color="#6ACC65", lw=2, ms=6, label="Random LHS",
                markeredgecolor="white", markeredgewidth=0.8)
        # Exploration series
        for label, sizes, metrics, fmt, color in series:
            ax.plot(sizes, _summaries(metrics, key), fmt,
                    color=color, lw=2, ms=6, label=label,
                    markeredgecolor="white", markeredgewidth=0.8)

        ax.axvspan(x_min, N_BASELINE + 0.5, color="#f0f0f0", zorder=0)
        ax.set_title(ax_title, fontsize=11, pad=8)
        ax.set_xlabel("Training experiments", fontsize=10)
        ax.set_xlim(x_min, x_max)
        y_lo = max(min(y_vals) - 0.15, -3.0)
        ax.set_ylim(y_lo, 1.12)
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.axhline(1, color="#cccccc", lw=0.5, ls="--")
        ax.grid(True, alpha=0.2, ls="--")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.set_xticks(range(int(x_min) + 1, int(x_max) + 1))

    ax_r2.set_ylabel("Perf-weighted score", fontsize=10)

    fig.text(0.5, -0.04,
             f"test set: {N_TEST} uniform experiments  ·  "
             f"alpha={ALPHA}  ·  symmetric={SYMMETRIC}  ·  "
             f"gap > 0 → better on important samples",
             ha="center", fontsize=8, color="#888888")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


def plot_forward_passes(
    lbfgsb_sizes: list[int],
    lbfgsb_nfev: list[int],
    de_sizes: list[int],
    de_nfev: list[int],
) -> None:
    """Plot cumulative forward passes per optimizer over exploration rounds."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lbfgsb_sizes, lbfgsb_nfev, "o-", color="#DD8452", lw=2, ms=6, label="L-BFGS-B")
    ax.plot(de_sizes,     de_nfev,     "s--", color="#C44E52", lw=2, ms=6, label="DE")
    ax.set_xlabel("Training set size", fontsize=10)
    ax.set_ylabel("Cumulative forward passes", fontsize=10)
    ax.set_title("Computational Cost: Cumulative Forward Passes per Optimizer", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "forward_passes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _print_summary(
    results: dict[str, list[StepMetrics]],
    nfev: dict[str, list[int]] | None = None,
) -> None:
    """Print final R²/R²_adj metrics for all methods."""
    feats = sorted(FEATURE_PERF_MAP.keys())
    methods = list(results.keys())
    col_w = 8

    # Header
    r2_cols = "  ".join(f"{m[:8]:>{col_w}}" for m in methods)
    adj_cols = "  ".join(f"{m[:8]:>{col_w}}" for m in methods)
    print(f"\n══ Results Summary (at final training size) ══")
    print(f"  {'Feature':25s}  {r2_cols}  │  {adj_cols}")
    print("  " + "─" * (28 + (col_w + 2) * len(methods) * 2 + 5))
    for feat in feats:
        r2_vals = "  ".join(
            f"{results[m][-1].get(feat, {}).get('r2', float('nan')):{col_w}.3f}"
            for m in methods
        )
        adj_vals = "  ".join(
            f"{results[m][-1].get(feat, {}).get('r2_adj', float('nan')):{col_w}.3f}"
            for m in methods
        )
        print(f"  {feat:25s}  {r2_vals}  │  {adj_vals}")

    print(f"\n  [perf-weighted summary]")
    r2_line = "  ".join(f"{_weighted_summary(results[m][-1], 'r2'):{col_w}.3f}" for m in methods)
    adj_line = "  ".join(f"{_weighted_summary(results[m][-1], 'r2_adj'):{col_w}.3f}" for m in methods)
    gap_line = "  ".join(
        f"{_weighted_summary(results[m][-1], 'r2_adj') - _weighted_summary(results[m][-1], 'r2'):{col_w}.3f}"
        for m in methods
    )
    print(f"  {'R²':25s}  {r2_line}")
    print(f"  {'R²_adj':25s}  {adj_line}")
    print(f"  {'gap (adj−r²)':25s}  {gap_line}")

    if nfev:
        parts = ", ".join(f"{m}: {v[-1]}" for m, v in nfev.items())
        print(f"\n  Cumulative forward passes — {parts}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Boundary buffer conditions to compare: (label, (extent, strength, exponent), color, fmt)
BUFFER_CONDITIONS: list[tuple[str, tuple[float, float, float] | None, str, str]] = [
    ("No buffer",        None,               "#4878CF", "o-"),
    ("Default",          BOUNDARY_BUFFER,     "#D65F5F", "s-"),
]


def main() -> None:
    # Suppress sklearn numerical warnings from MLP training on tiny datasets
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Shared fab and test data
    _, fab, _, _ = _make_fresh_env(DATA_ROOT + "_setup")
    test_params = _build_test_dataset(fab)

    print("\n[1] Building test dataset...")
    agent_test, fab_test, _, _ = _make_fresh_env(DATA_ROOT + "_setup")
    test_dataset = _run_test_dataset(fab_test, agent_test, DATA_ROOT + "_test", test_params)
    shutil.rmtree(DATA_ROOT + "_setup", ignore_errors=True)

    # Random baseline (always needed)
    n_total = N_BASELINE + N_EXPLORE
    print(f"\n[2] Random LHS workflow (n={max(N_RANDOM, n_total)}, incremental)...")
    random_sizes, random_metrics = run_random_workflow(
        fab_test, test_dataset, test_params, max(N_RANDOM, n_total), eval_from=N_BASELINE,
    )

    # Run each buffer condition with DE optimizer
    all_results: dict[str, list[StepMetrics]] = {"random": random_metrics}
    all_sizes: dict[str, list[int]] = {"random": random_sizes}
    all_nfev: dict[str, list[int]] = {}
    plot_series: list[SeriesData] = []

    for idx, (label, buffer, color, fmt) in enumerate(BUFFER_CONDITIONS):
        tag = f"_buf{idx}"
        print(f"\n[{idx+3}] Exploration (DE, {label}): baseline={N_BASELINE}, explore={N_EXPLORE}...")
        sizes, metrics, nfev = run_exploration_workflow(
            fab_test, test_dataset, test_params,
            optimizer=Optimizer.DE,
            boundary_buffer=buffer,
            tag_suffix=tag,
        )
        all_results[label] = metrics
        all_sizes[label] = sizes
        all_nfev[label] = nfev
        plot_series.append((label, sizes, metrics, fmt, color))
        print(f"  Total forward passes ({label}): {nfev[-1]}")

    # Plots
    plot_learning_curves(
        plot_series, random_sizes, random_metrics,
        filename="learning_curves.png",
        title="Boundary Buffer Comparison (DE optimizer)",
    )

    # Cleanup
    shutil.rmtree(DATA_ROOT + "_test", ignore_errors=True)

    _print_summary(all_results, all_nfev)


if __name__ == "__main__":
    main()
