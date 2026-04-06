"""Prediction model quality benchmark: baseline-only vs. exploration-guided sampling.

Research question (from ADVEI paper):
  How does the data efficiency of exploration-guided sampling compare to
  grid-search/random sampling as a function of dataset size?

Approach:
  1. Generate a fixed test dataset by uniform grid over the parameter space.
  2. Train on baseline-only data — record R² on test set.
  3. Add exploration experiments one-by-one — record R² after each addition.
  4. Train on an equally-sized random sample — record R² for comparison.
  5. Plot learning curves: R² vs. training set size for each workflow.

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
from utils import params_from_spec, get_performance

# ── Configuration ─────────────────────────────────────────────────────────────
N_BASELINE    = 4      # baseline experiments (LHS)
N_EXPLORE     = 10     # exploration rounds
N_TEST        = 32     # test experiments (uniform grid, never used for training)
W_EXPLORE     = 0.7
PERF_WEIGHTS  = {"path_accuracy": 2, "energy_efficiency": 1, "production_rate": 1}
CAL_BOUNDS    = {"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)}

DATA_ROOT     = "./dev_data"
PLOT_DIR      = "./dev_plots"

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    fab: FabricationSystem,
    params: Dict[str, Any],
    exp_code: str,
) -> Any:
    """Run one fabrication experiment and evaluate features + performance."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data


def _with_dimensions(params: Dict[str, Any], fab: FabricationSystem) -> Dict[str, Any]:
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}


def _build_test_dataset(fab: FabricationSystem) -> List[Dict[str, Any]]:
    """Generate N_TEST parameter combinations on a uniform grid.

    Covers the full (water_ratio, print_speed, design, material) space.
    Offsets slightly from LHS grid to ensure independence from training data.
    """
    designs   = ["A", "B"]
    materials = ["clay", "concrete"]
    n_per_combo = N_TEST // 4  # equal per design×material

    test_params: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed=999)  # fixed seed for reproducibility

    for design in designs:
        for material in materials:
            # Jittered uniform grid
            waters = np.linspace(0.31, 0.49, n_per_combo)
            speeds = np.linspace(21.0, 59.0, n_per_combo)
            # Add small jitter so points don't land on LHS strata boundaries
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
    fab: FabricationSystem,
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
    """Evaluate prediction R² on the fixed test set.

    For each test experiment, predict features using the fitted model and
    compare to true physics values from the test dataset.
    """
    from pred_fab.core import DataModule
    # Build a test datamodule sharing train normalization
    test_dm = DataModule(dataset=test_dataset)
    test_dm.initialize(
        input_parameters=agent.pred_system.get_system_input_parameters(),
        input_features=agent.pred_system.get_system_input_features(),
        output_columns=agent.pred_system.get_system_outputs(),
    )
    test_codes = list(test_dataset._experiments.keys())
    test_dm.set_split_codes(train_codes=[], val_codes=[], test_codes=test_codes)

    # Copy normalization from training
    test_dm.set_normalization_state(train_dm.get_normalization_state())

    # Get ground-truth features
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
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            r2_scores[feat] = float(r2)
    return r2_scores


# ── Workflow A: Baseline only ─────────────────────────────────────────────────

def run_baseline_workflow(
    fab: FabricationSystem,
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
    n_baseline: int,
) -> Dict[str, float]:
    """Train on n_baseline LHS experiments, return R² on test set."""
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + "_baseline")
    # Baseline
    specs = agent.baseline_step(n=n_baseline)
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        _run_and_evaluate(dataset, agent, fab2, params, f"baseline_{i+1:02d}")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    r2 = _compute_r2_on_test(agent, dm, test_dataset, test_params)
    shutil.rmtree(DATA_ROOT + "_baseline", ignore_errors=True)
    return r2


# ── Workflow B: Exploration-guided ────────────────────────────────────────────

def run_exploration_workflow(
    fab: FabricationSystem,
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, float]]]:
    """
    Baseline + incremental exploration. After each new experiment, train and
    evaluate R² on test set. Returns (training_sizes, r2_per_step).
    """
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + "_explore")

    # Phase 1: Baseline
    specs = agent.baseline_step(n=N_BASELINE)
    baseline_exps = []
    prev_params: Dict[str, Any] = {}
    for i, spec in enumerate(specs):
        params = _with_dimensions(params_from_spec(spec), fab2)
        exp = _run_and_evaluate(dataset, agent, fab2, params, f"baseline_{i+1:02d}")
        baseline_exps.append(exp)
        prev_params = params

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    training_sizes = [N_BASELINE]
    r2_history = [_compute_r2_on_test(agent, dm, test_dataset, test_params)]
    print(f"    n={N_BASELINE}: {r2_history[-1]}")

    # Phase 2: Exploration rounds
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, w_explore=W_EXPLORE)
        params = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab2)
        exp_code = f"explore_{i+1:02d}"
        _run_and_evaluate(dataset, agent, fab2, params, exp_code)
        dm.update()
        agent.train(dm, validate=False)
        prev_params = params

        n_total = N_BASELINE + i + 1
        r2 = _compute_r2_on_test(agent, dm, test_dataset, test_params)
        training_sizes.append(n_total)
        r2_history.append(r2)
        print(f"    n={n_total}: {r2}")

    shutil.rmtree(DATA_ROOT + "_explore", ignore_errors=True)
    return training_sizes, r2_history


# ── Workflow C: Random sampling baseline ─────────────────────────────────────

def run_random_workflow(
    fab: FabricationSystem,
    test_dataset: Dataset,
    test_params: List[Dict[str, Any]],
    n_total: int,
) -> Dict[str, float]:
    """Train on n_total random experiments, return R² on test set.

    Samples uniformly at random (like a denser LHS), representing the
    strategy of 'just run more baseline experiments' without exploration.
    """
    agent, fab2, schema, dataset = _make_fresh_env(DATA_ROOT + "_random")
    # Use baseline_step with larger n to get random/LHS samples
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
    explore_sizes: List[int],
    explore_r2s: List[Dict[str, float]],
    baseline_r2: Dict[str, float],
    random_r2: Dict[str, float],
) -> None:
    """Plot R² learning curves for exploration vs. baselines."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Features to plot (exclude production_rate — deterministic, always R²~1)
    features = [f for f in explore_r2s[0].keys() if f != "production_rate"]

    fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 5))
    if len(features) == 1:
        axes = [axes]

    fig.suptitle(
        "Prediction Model Quality: Exploration vs. Baseline\n"
        f"(test set: {N_TEST} uniform experiments, never seen during training)",
        fontsize=12, fontweight="bold",
    )

    for ax, feat in zip(axes, features):
        # Exploration learning curve
        y_exp = [d.get(feat, np.nan) for d in explore_r2s]
        ax.plot(explore_sizes, y_exp, "o-", color="#DD8452", lw=2, ms=6,
                label="Exploration-guided")

        # Baseline-only (single point at N_BASELINE)
        ax.axhline(y=baseline_r2.get(feat, np.nan), color="#4C72B0", lw=1.5,
                   linestyle="--", label=f"Baseline-only (n={N_BASELINE})")

        # Random (single point at N_BASELINE + N_EXPLORE)
        n_random = N_BASELINE + N_EXPLORE
        ax.axhline(y=random_r2.get(feat, np.nan), color="#55A868", lw=1.5,
                   linestyle=":", label=f"Random (n={n_random})")

        ax.set_xlabel("Training set size", fontsize=10)
        ax.set_ylabel("R² on test set", fontsize=10)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_xlim(0, N_BASELINE + N_EXPLORE + 1)
        ax.set_ylim(-0.5, 1.05)
        ax.axhline(0, color="grey", lw=0.8, linestyle="-", alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "learning_curves.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Shared fab and test data
    _, fab, _, _ = _make_fresh_env(DATA_ROOT + "_setup")
    test_params = _build_test_dataset(fab)

    # Build test dataset using a shared agent for evaluation models
    print("\n[1/4] Building test dataset...")
    agent_test, fab_test, _, _ = _make_fresh_env(DATA_ROOT + "_setup")
    test_dataset = _run_test_dataset(fab_test, agent_test, DATA_ROOT + "_test", test_params)
    shutil.rmtree(DATA_ROOT + "_setup", ignore_errors=True)

    # Workflow A: baseline only
    print(f"\n[2/4] Baseline-only workflow (n={N_BASELINE})...")
    baseline_r2 = run_baseline_workflow(fab_test, test_dataset, test_params, N_BASELINE)
    print(f"  R²: {baseline_r2}")

    # Workflow B: exploration-guided
    print(f"\n[3/4] Exploration-guided workflow (baseline={N_BASELINE}, explore={N_EXPLORE})...")
    explore_sizes, explore_r2s = run_exploration_workflow(
        fab_test, test_dataset, test_params
    )

    # Workflow C: random (same total n as exploration)
    n_random = N_BASELINE + N_EXPLORE
    print(f"\n[4/4] Random sampling workflow (n={n_random})...")
    random_r2 = run_random_workflow(fab_test, test_dataset, test_params, n_random)
    print(f"  R²: {random_r2}")

    # Plot
    plot_learning_curves(explore_sizes, explore_r2s, baseline_r2, random_r2)

    # Cleanup
    shutil.rmtree(DATA_ROOT + "_test", ignore_errors=True)

    # Summary
    print("\n══ Results Summary ══")
    feats = list(explore_r2s[-1].keys())
    for feat in feats:
        b = baseline_r2.get(feat, np.nan)
        e_final = explore_r2s[-1].get(feat, np.nan)
        r = random_r2.get(feat, np.nan)
        print(f"  {feat:25s}  baseline={b:.3f}  explore_final={e_final:.3f}  random={r:.3f}")


if __name__ == "__main__":
    main()
