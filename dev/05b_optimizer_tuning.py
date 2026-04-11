"""05b — Optimizer Parameter Tuning.

Systematically test different optimizer configurations to find settings
that produce better exploration proposals (not stuck at boundaries).

Tests:
  1. DE: vary maxiter, popsize, mutation, recombination
  2. L-BFGS-B: vary maxfun, n_optimization_rounds, eps
  3. Brute force: evaluate the acquisition function on a dense grid to find
     the true optimum — compare what the optimizer finds vs the real answer.
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.orchestration import Optimizer
from sensors.physics import N_LAYERS, N_SEGMENTS

from shared import make_env, run_baseline, train_models, with_dims, ensure_plot_dir
from utils import params_from_spec

# ── Configuration ─────────────────────────────────────────────────────────────
N_BASELINE = 10
BOUNDS = {"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)}
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
W_EXPLORE = 0.7
EXPLORATION_RADIUS = 0.5
BOUNDARY_BUFFER = (0.10, 0.8, 2.0)


def _run_single_exploration(agent, dm, prev_params, optimizer, **extra_config):
    """Run one exploration step and return the proposal + metadata."""
    agent.configure(optimizer=optimizer, **extra_config)
    spec = agent.exploration_step(dm, w_explore=W_EXPLORE, n_optimization_rounds=10)
    proposed = params_from_spec(spec)
    params = with_dims({**prev_params, **proposed})
    return {
        "water": params["water_ratio"],
        "speed": params["print_speed"],
        "score": agent.calibration_system.last_opt_score,
        "nfev": agent.calibration_system.last_opt_nfev,
    }


def _brute_force_acquisition(agent, dm, res=50):
    """Evaluate the acquisition function on a dense grid to find the true maximum."""
    cal = agent.calibration_system
    cal._active_datamodule = dm
    bounds = cal._get_global_bounds(dm)
    objective = cal._build_objective(
        mode=__import__("pred_fab.utils", fromlist=["Mode"]).Mode.EXPLORATION,
        w_explore=W_EXPLORE,
        bounds=bounds,
    )

    waters = np.linspace(0.30, 0.50, res)
    speeds = np.linspace(20.0, 60.0, res)

    best_score = float("inf")
    best_w, best_s = 0.0, 0.0
    scores = np.zeros((res, res))

    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            params = {"water_ratio": w, "print_speed": spd,
                      "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
            X = dm.params_to_array(params)
            val = objective(X)
            scores[j, i] = -val  # negate back to positive
            if val < best_score:
                best_score = val
                best_w, best_s = w, spd

    return best_w, best_s, -best_score, waters, speeds, scores


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    plot_dir = ensure_plot_dir()

    # ── Setup: shared baseline ────────────────────────────────────────────────
    agent, fab, dataset = make_env("05b_tuning", verbose=False)
    agent.configure(
        bounds=BOUNDS, performance_weights=PERF_WEIGHTS,
        exploration_radius=EXPLORATION_RADIUS, boundary_buffer=BOUNDARY_BUFFER,
    )
    baseline_params = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)
    prev = baseline_params[-1]

    # ── Brute force: find the true acquisition maximum ────────────────────────
    print("\n  Brute force search (50×50 grid)...")
    bf_w, bf_s, bf_score, bf_waters, bf_speeds, bf_scores = _brute_force_acquisition(agent, dm, res=50)
    print(f"    True acquisition max: w={bf_w:.3f}  spd={bf_s:.1f}  score={bf_score:.4f}")

    # ── Test DE configurations ────────────────────────────────────────────────
    de_configs = [
        ("DE default",      {"maxiter": 50,  "popsize": 5,  "mutation": (0.5, 1.0), "recombination": 0.7}),
        ("DE more iters",   {"maxiter": 200, "popsize": 5,  "mutation": (0.5, 1.0), "recombination": 0.7}),
        ("DE larger pop",   {"maxiter": 50,  "popsize": 15, "mutation": (0.5, 1.0), "recombination": 0.7}),
        ("DE both",         {"maxiter": 200, "popsize": 15, "mutation": (0.5, 1.0), "recombination": 0.7}),
        ("DE high mutation", {"maxiter": 100, "popsize": 10, "mutation": (0.8, 1.5), "recombination": 0.9}),
    ]

    print(f"\n  DE configurations (w_explore={W_EXPLORE}):")
    print(f"  {'Config':25s}  {'water':>6s}  {'speed':>6s}  {'score':>7s}  {'nfev':>6s}  {'gap':>6s}")
    print(f"  {'─' * 65}")

    de_results = []
    for label, de_kwargs in de_configs:
        # We need to directly set DE params in the calibration system since
        # they're not exposed through agent.configure. Override them temporarily.
        cal = agent.calibration_system
        original_de = cal._optimize_de

        def _patched_de(bounds, objective_func, _kw=de_kwargs):
            from scipy.optimize import differential_evolution
            seed = int(cal.rng.randint(0, 2**31 - 1))
            result = differential_evolution(
                func=objective_func, bounds=bounds.tolist(),
                maxiter=_kw["maxiter"], popsize=_kw["popsize"], seed=seed,
                mutation=_kw["mutation"], recombination=_kw["recombination"],
                tol=1e-4, polish=True, init='latinhypercube',
            )
            from pred_fab.orchestration.calibration import _OptResult
            return _OptResult(best_x=result.x, nfev=result.nfev, n_starts=1,
                              score=float(-result.fun))

        cal._optimize_de = _patched_de
        r = _run_single_exploration(agent, dm, prev, Optimizer.DE)
        cal._optimize_de = original_de

        gap = bf_score - r["score"]
        print(f"  {label:25s}  {r['water']:6.3f}  {r['speed']:6.1f}  {r['score']:7.4f}  {r['nfev']:6d}  {gap:6.4f}")
        de_results.append((label, r))

    # ── Test L-BFGS-B configurations ──────────────────────────────────────────
    lbfgsb_configs = [
        ("LBFGSB 5 starts",   5),
        ("LBFGSB 10 starts", 10),
        ("LBFGSB 20 starts", 20),
        ("LBFGSB 50 starts", 50),
    ]

    print(f"\n  L-BFGS-B configurations:")
    print(f"  {'Config':25s}  {'water':>6s}  {'speed':>6s}  {'score':>7s}  {'nfev':>6s}  {'gap':>6s}")
    print(f"  {'─' * 65}")

    lbfgsb_results = []
    for label, n_rounds in lbfgsb_configs:
        r = _run_single_exploration(agent, dm, prev, Optimizer.LBFGSB,
                                     **{})  # n_optimization_rounds is passed in exploration_step
        # Retry with specific n_rounds
        spec = agent.exploration_step(dm, w_explore=W_EXPLORE, n_optimization_rounds=n_rounds)
        proposed = params_from_spec(spec)
        params = with_dims({**prev, **proposed})
        r = {
            "water": params["water_ratio"],
            "speed": params["print_speed"],
            "score": agent.calibration_system.last_opt_score,
            "nfev": agent.calibration_system.last_opt_nfev,
        }
        gap = bf_score - r["score"]
        print(f"  {label:25s}  {r['water']:6.3f}  {r['speed']:6.1f}  {r['score']:7.4f}  {r['nfev']:6d}  {gap:6.4f}")
        lbfgsb_results.append((label, r))

    # ── Plot: Acquisition landscape + optimizer proposals ─────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Optimizer Proposals vs True Acquisition Maximum", fontsize=13, fontweight="bold")

    im = ax.contourf(bf_waters, bf_speeds, bf_scores, levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Acquisition Score")

    # True maximum
    ax.plot(bf_w, bf_s, "w*", ms=18, markeredgecolor="black", markeredgewidth=1.2,
            label=f"True max ({bf_w:.2f}, {bf_s:.0f})", zorder=10)

    # DE proposals
    for label, r in de_results:
        ax.plot(r["water"], r["speed"], "o", ms=8, label=f"{label}", zorder=8)

    # LBFGSB proposals
    for label, r in lbfgsb_results:
        ax.plot(r["water"], r["speed"], "s", ms=8, label=f"{label}", zorder=8)

    ax.set_xlabel("Water Ratio")
    ax.set_ylabel("Print Speed [mm/s]")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = os.path.join(plot_dir, "05b_optimizer_tuning.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
