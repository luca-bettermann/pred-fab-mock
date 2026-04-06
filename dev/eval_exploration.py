"""Prediction model quality benchmark: baseline-only vs. exploration-guided sampling.

Research question (from ADVEI paper):
  How does the data efficiency of exploration-guided sampling compare to
  grid-search/random sampling as a function of dataset size?

Approach:
  1. Generate a fixed test dataset by uniform grid over the parameter space.
  2. Train on baseline-only data — record metrics on test set.
  3. Add exploration experiments one-by-one (per optimizer) — record metrics.
  4. Train on random LHS (n=20) — record metrics for comparison.
  5. Plot learning curves (per-feature R²) and cumulative forward passes.

Two metrics reported:
  - Per-feature R²: uniform R² for each predicted feature.
  - Weighted summary R²: Σ(perf_weight_i × R²_i) / Σ(perf_weight_i), mapping
    features to their corresponding performance attributes via FEATURE_PERF_MAP.
  - Exploration-adjusted Q: (1 − W_EXPLORE) × weighted_R² + W_EXPLORE.
    Models how much prediction accuracy matters for UCB guidance at this w_explore:
    the W_EXPLORE fraction of the acquisition is uncertainty-driven and is assumed
    perfect, so only the performance component needs good predictions.

Uses the same schema, agent, sensors, and models as main.py — no new code.
"""

import os
import sys
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ── make local modules importable when running from dev/ ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.core import Dataset
from pred_fab.utils import SplitType  # type: ignore[attr-defined]

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec

# ── Configuration ─────────────────────────────────────────────────────────────
N_BASELINE    = 4      # baseline experiments (LHS)
N_EXPLORE     = 10     # exploration rounds
N_TEST        = 32     # test experiments (uniform grid, never used for training)
N_RANDOM      = 20     # random-LHS baseline (larger pool for fairer comparison)
W_EXPLORE     = 0.7
PERF_WEIGHTS  = {"path_accuracy": 2, "energy_efficiency": 1, "production_rate": 1}
CAL_BOUNDS    = {"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)}

# Maps each predicted feature to its corresponding performance attribute.
FEATURE_PERF_MAP: Dict[str, str] = {
    "path_deviation":    "path_accuracy",
    "energy_per_segment": "energy_efficiency",
    "production_rate":   "production_rate",
}

DATA_ROOT     = "./dev_data"
PLOT_DIR      = "./dev_plots"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _weighted_summary_r2(r2_scores: Dict[str, float]) -> float:
    """Compute performance-weighted summary R² = Σ(perf_w_i × R²_i) / Σ(perf_w_i)."""
    total_w, weighted_sum = 0.0, 0.0
    for feat, perf_name in FEATURE_PERF_MAP.items():
        w = PERF_WEIGHTS.get(perf_name, 0.0)
        total_w += w
        weighted_sum += w * r2_scores.get(feat, 0.0)
    return weighted_sum / total_w if total_w > 0 else 0.0


def _exploration_adjusted_q(weighted_r2: float) -> float:
    """Q = (1-W_EXPLORE) × weighted_R² + W_EXPLORE.

    Reflects how much prediction accuracy matters for UCB guidance at this w_explore.
    """
    return (1.0 - W_EXPLORE) * weighted_r2 + W_EXPLORE


def _make_fresh_env(data_root: str) -> Tuple[Any, Any, Any, Dataset]:
    """Build a fresh agent+dataset for one workflow run."""
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    schema = build_schema(root_folder=data_root)
    fab    = FabricationSystem(CameraSystem(), EnergySensor())
    agent  = build_agent(schema, fab.camera, fab.energy)
    agent.configure_calibration(bounds=CAL_BOUNDS, performance_weights=PERF_WEIGHTS)
    dataset = Dataset(schema=schema)
    agent.logger.set_console_output(False)
    return agent, fab, schema, dataset


def _run_and_evaluate(
    dataset: Dataset,
    agent: Any,
    fab: "FabricationSystem",
    params: Dict[str, Any],
    exp_code: str,
) -> Any:
    """Run one fabrication experiment and evaluate features + performance."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data


def _with_dimensions(params: Dict[str, Any], fab: "FabricationSystem") -> Dict[str, Any]:
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}


def _build_test_dataset(fab: "FabricationSystem") -> List[Dict[str, Any]]:
    """Generate N_TEST parameter combinations on a jittered uniform grid.

    Covers the full (water_ratio, print_speed, design, material) space.
    Fixed seed=999 — never overlaps with training data.
    """
    designs   = ["A", "B"]
    materials = ["clay", "concrete"]
    n_per_combo = N_TEST // 4

    test_params: List[Dict[str, Any]] = []
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
    test_params: List[Dict[str, Any]],
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


def _compute_r2_on_test(
    agent: Any,
    train_dm: Any,
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate R² per feature on the fixed test set.

    Uses shared normalization from train_dm so predictions are comparable.
    Returns a dict mapping each feature name to its R² score.
    """
    from pred_fab.core import DataModule
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

    r2_scores: Dict[str, float] = {}
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
            r2_scores[feat] = float(r2_score(y_true[:, i], y_pred[:, i]))
    return r2_scores


# ── Workflow A: Baseline only ─────────────────────────────────────────────────

def run_baseline_workflow(
    fab: "FabricationSystem",
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
    n_baseline: int,
) -> Dict[str, float]:
    """Train on n_baseline LHS experiments, return per-feature R² on test set."""
    tag = f"baseline{n_baseline}"
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + f"_{tag}")
    specs = agent.baseline_step(n=n_baseline)
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"baseline_{i+1:02d}")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    r2 = _compute_r2_on_test(agent, dm, test_dataset, test_params)
    shutil.rmtree(DATA_ROOT + f"_{tag}", ignore_errors=True)
    return r2


# ── Workflow B: Exploration-guided ────────────────────────────────────────────

def run_exploration_workflow(
    fab: "FabricationSystem",
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
    optimizer: str = "lbfgsb",
) -> Tuple[List[int], List[Dict[str, float]], List[int]]:
    """Baseline + incremental exploration.

    After each new experiment, train and evaluate R² on the test set.
    Returns (training_sizes, r2_per_step, cumulative_nfev_per_step).
    """
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + f"_explore_{optimizer}")
    agent.calibration_system.optimizer = optimizer

    # Phase 1: Baseline
    specs = agent.baseline_step(n=N_BASELINE)
    prev_params: Dict[str, Any] = {}
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"baseline_{i+1:02d}")
        prev_params = params

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    r2_0 = _compute_r2_on_test(agent, dm, test_dataset, test_params)
    training_sizes = [N_BASELINE]
    r2_history = [r2_0]
    nfev_cumulative = [0]  # no forward passes before exploration
    print(f"    n={N_BASELINE}: {r2_0}  |  nfev=0")

    # Phase 2: Exploration rounds
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, w_explore=W_EXPLORE)
        nfev_step = agent.calibration_system.last_opt_nfev
        params = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"explore_{i+1:02d}")
        dm.update()
        agent.train(dm, validate=False)
        prev_params = params

        n_total = N_BASELINE + i + 1
        cumulative = nfev_cumulative[-1] + nfev_step
        r2 = _compute_r2_on_test(agent, dm, test_dataset, test_params)
        training_sizes.append(n_total)
        r2_history.append(r2)
        nfev_cumulative.append(cumulative)
        print(f"    n={n_total}: {r2}  |  nfev_step={nfev_step}, cumulative={cumulative}")

    shutil.rmtree(DATA_ROOT + f"_explore_{optimizer}", ignore_errors=True)
    return training_sizes, r2_history, nfev_cumulative


# ── Workflow C: Random sampling baseline ─────────────────────────────────────

def run_random_workflow(
    fab: "FabricationSystem",
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
    n_total: int,
) -> Dict[str, float]:
    """Train on n_total random LHS experiments, return per-feature R² on test set."""
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + "_random")
    specs = agent.baseline_step(n=n_total)
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"random_{i+1:02d}")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    r2 = _compute_r2_on_test(agent, dm, test_dataset, test_params)
    shutil.rmtree(DATA_ROOT + "_random", ignore_errors=True)
    return r2


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curves(
    lbfgsb_sizes: List[int],
    lbfgsb_r2s: List[Dict[str, float]],
    de_sizes: List[int],
    de_r2s: List[Dict[str, float]],
    baseline_r2: Dict[str, float],
    random_r2: Dict[str, float],
) -> None:
    """Plot per-feature R² learning curves + weighted summary R² for all workflows."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    features = list(lbfgsb_r2s[0].keys())
    n_feats = len(features)

    # 2-row layout: row 0 = per-feature R², row 1 = weighted summary R²
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(6 * n_feats, 10))
    gs = gridspec.GridSpec(2, n_feats, figure=fig, hspace=0.4)

    fig.suptitle(
        "Prediction Model Quality: Exploration vs. Baseline\n"
        f"(test set: {N_TEST} uniform experiments, seed=999)",
        fontsize=12, fontweight="bold",
    )

    colors = {
        "lbfgsb":   "#DD8452",
        "de":       "#C44E52",
        "baseline": "#4C72B0",
        "random":   "#55A868",
    }
    n_x = N_BASELINE + N_EXPLORE

    # Row 0: per-feature R²
    for col, feat in enumerate(features):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(lbfgsb_sizes, [d.get(feat, np.nan) for d in lbfgsb_r2s],
                "o-", color=colors["lbfgsb"], lw=2, ms=6, label="Explore (L-BFGS-B)")
        ax.plot(de_sizes, [d.get(feat, np.nan) for d in de_r2s],
                "s--", color=colors["de"], lw=2, ms=6, label="Explore (DE)")
        ax.axhline(baseline_r2.get(feat, np.nan), color=colors["baseline"],
                   lw=1.5, ls="--", label=f"Baseline (n={N_BASELINE})")
        ax.axhline(random_r2.get(feat, np.nan), color=colors["random"],
                   lw=1.5, ls=":", label=f"Random LHS (n={N_RANDOM})")
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Training set size", fontsize=9)
        ax.set_ylabel("R²", fontsize=9)
        ax.set_xlim(0, n_x + 1)
        ax.set_ylim(-0.5, 1.05)
        ax.axhline(0, color="grey", lw=0.8, alpha=0.4)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

    # Row 1: performance-weighted summary R² (single plot spanning all cols)
    ax_sum = fig.add_subplot(gs[1, :])
    lbfgsb_sum = [_weighted_summary_r2(d) for d in lbfgsb_r2s]
    de_sum     = [_weighted_summary_r2(d) for d in de_r2s]
    bl_sum     = _weighted_summary_r2(baseline_r2)
    rnd_sum    = _weighted_summary_r2(random_r2)

    ax_sum.plot(lbfgsb_sizes, lbfgsb_sum, "o-", color=colors["lbfgsb"], lw=2, ms=6,
                label="Explore (L-BFGS-B)")
    ax_sum.plot(de_sizes, de_sum, "s--", color=colors["de"], lw=2, ms=6,
                label="Explore (DE)")
    ax_sum.axhline(bl_sum,  color=colors["baseline"], lw=1.5, ls="--",
                   label=f"Baseline (n={N_BASELINE})")
    ax_sum.axhline(rnd_sum, color=colors["random"],   lw=1.5, ls=":",
                   label=f"Random LHS (n={N_RANDOM})")

    # Exploration-adjusted Q on secondary axis
    ax_q = ax_sum.twinx()
    lbfgsb_q = [_exploration_adjusted_q(v) for v in lbfgsb_sum]
    de_q     = [_exploration_adjusted_q(v) for v in de_sum]
    ax_q.plot(lbfgsb_sizes, lbfgsb_q, "o:", color=colors["lbfgsb"], lw=1, ms=4, alpha=0.5)
    ax_q.plot(de_sizes,     de_q,     "s:", color=colors["de"],     lw=1, ms=4, alpha=0.5)
    ax_q.set_ylabel(f"Exploration Q  (right, W={W_EXPLORE})", fontsize=8, color="grey")
    ax_q.set_ylim(0.5, 1.05)
    ax_q.tick_params(axis="y", labelcolor="grey", labelsize=7)

    ax_sum.set_title(
        f"Weighted Summary R²  (weights: {PERF_WEIGHTS})\n"
        f"(dotted = exploration-adjusted Q = (1−{W_EXPLORE})·R² + {W_EXPLORE})",
        fontsize=9,
    )
    ax_sum.set_xlabel("Training set size", fontsize=9)
    ax_sum.set_ylabel("Perf-weighted R²", fontsize=9)
    ax_sum.set_xlim(0, n_x + 1)
    ax_sum.set_ylim(-0.2, 1.05)
    ax_sum.axhline(0, color="grey", lw=0.8, alpha=0.4)
    ax_sum.grid(True, alpha=0.3)
    ax_sum.legend(fontsize=8)

    out = os.path.join(PLOT_DIR, "learning_curves.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


def plot_forward_passes(
    lbfgsb_sizes: List[int],
    lbfgsb_nfev: List[int],
    de_sizes: List[int],
    de_nfev: List[int],
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
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Shared fab and test data
    _, fab, _, _ = _make_fresh_env(DATA_ROOT + "_setup")
    test_params = _build_test_dataset(fab)

    print("\n[1/5] Building test dataset...")
    agent_test, fab_test, _, _ = _make_fresh_env(DATA_ROOT + "_setup")
    test_dataset = _run_test_dataset(fab_test, agent_test, DATA_ROOT + "_test", test_params)
    shutil.rmtree(DATA_ROOT + "_setup", ignore_errors=True)

    print(f"\n[2/5] Baseline-only workflow (n={N_BASELINE})...")
    baseline_r2 = run_baseline_workflow(fab_test, test_dataset, test_params, N_BASELINE)
    print(f"  R²: {baseline_r2}")

    print(f"\n[3/5] Exploration (L-BFGS-B): baseline={N_BASELINE}, explore={N_EXPLORE}...")
    lbfgsb_sizes, lbfgsb_r2s, lbfgsb_nfev = run_exploration_workflow(
        fab_test, test_dataset, test_params, optimizer="lbfgsb"
    )
    print(f"  Total forward passes (L-BFGS-B): {lbfgsb_nfev[-1]}")

    print(f"\n[4/5] Exploration (DE): baseline={N_BASELINE}, explore={N_EXPLORE}...")
    de_sizes, de_r2s, de_nfev = run_exploration_workflow(
        fab_test, test_dataset, test_params, optimizer="de"
    )
    print(f"  Total forward passes (DE): {de_nfev[-1]}")

    print(f"\n[5/5] Random LHS workflow (n={N_RANDOM})...")
    random_r2 = run_random_workflow(fab_test, test_dataset, test_params, N_RANDOM)
    print(f"  R²: {random_r2}")

    # Plots
    plot_learning_curves(
        lbfgsb_sizes, lbfgsb_r2s,
        de_sizes, de_r2s,
        baseline_r2, random_r2,
    )
    plot_forward_passes(lbfgsb_sizes, lbfgsb_nfev, de_sizes, de_nfev)

    # Cleanup
    shutil.rmtree(DATA_ROOT + "_test", ignore_errors=True)

    # Summary
    print("\n══ Results Summary ══")
    feats = list(lbfgsb_r2s[-1].keys())
    col_w = 8
    header = f"  {'Feature':25s}  {'baseline':>{col_w}}  {'lbfgsb':>{col_w}}  {'de':>{col_w}}  {'random':>{col_w}}"
    sep = "  " + "-" * (len(header) - 2)
    print(header); print(sep)
    for feat in feats:
        vals = [
            baseline_r2.get(feat, float("nan")),
            lbfgsb_r2s[-1].get(feat, float("nan")),
            de_r2s[-1].get(feat, float("nan")),
            random_r2.get(feat, float("nan")),
        ]
        print(f"  {feat:25s}" + "".join(f"  {v:{col_w}.3f}" for v in vals))

    print("\n  [perf-weighted summary R²]")
    bl_sum  = _weighted_summary_r2(baseline_r2)
    l_sum   = _weighted_summary_r2(lbfgsb_r2s[-1])
    d_sum   = _weighted_summary_r2(de_r2s[-1])
    rnd_sum = _weighted_summary_r2(random_r2)
    print(f"  {'weighted_R2':25s}  {bl_sum:{col_w}.3f}  {l_sum:{col_w}.3f}  {d_sum:{col_w}.3f}  {rnd_sum:{col_w}.3f}")
    print(f"  {'exploration_Q':25s}  {_exploration_adjusted_q(bl_sum):{col_w}.3f}  {_exploration_adjusted_q(l_sum):{col_w}.3f}  {_exploration_adjusted_q(d_sum):{col_w}.3f}  {_exploration_adjusted_q(rnd_sum):{col_w}.3f}")
    print(f"\n  Cumulative forward passes — L-BFGS-B: {lbfgsb_nfev[-1]}, DE: {de_nfev[-1]}")


if __name__ == "__main__":
    main()
