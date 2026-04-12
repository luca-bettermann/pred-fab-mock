"""05 — Exploration Objective + Optimizer Validation.

Test the combined acquisition function and compare L-BFGS-B vs DE.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.orchestration import Optimizer
from sensors.physics import N_LAYERS, N_SEGMENTS
from visualization import plot_optimizer_comparison, plot_acquisition_topology
from shared import make_env, run_baseline, train_models, with_dims, run_experiment, ensure_plot_dir
from utils import params_from_spec

N_BASELINE = 5
N_EXPLORE = 10
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
RESOLUTION = 30
W_EXPLORE = 0.7
EXPLORATION_RADIUS = 0.5
BOUNDARY_BUFFER = (0.10, 0.8, 2.0)


def _compute_acquisition_grid(agent, dm, w_explore, res):
    waters = np.linspace(0.30, 0.50, res)
    speeds = np.linspace(20.0, 60.0, res)
    perf_grid = np.zeros((res, res))
    unc_grid = np.zeros((res, res))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            p = {"water_ratio": w, "print_speed": spd, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
            try:
                perf = agent.predict_performance(p)
                total_w = sum(PERF_WEIGHTS.values())
                perf_grid[j, i] = sum(PERF_WEIGHTS.get(k, 0) * float(v)
                                       for k, v in perf.items() if v is not None) / total_w
            except Exception:
                perf_grid[j, i] = 0.0
            unc_grid[j, i] = agent.predict_uncertainty(p, dm)

    p_min, p_max = perf_grid.min(), perf_grid.max()
    u_min, u_max = unc_grid.min(), unc_grid.max()
    p_norm = np.clip((perf_grid - p_min) / max(p_max - p_min, 1e-10), 0, 1)
    u_norm = np.clip((unc_grid - u_min) / max(u_max - u_min, 1e-10), 0, 1)
    combined = (1 - w_explore) * p_norm + w_explore * u_norm
    return waters, speeds, perf_grid, unc_grid, combined


def _run_exploration(optimizer, tag):
    agent, fab, dataset = make_env(f"05_{tag}", verbose=False)
    agent.configure(performance_weights=PERF_WEIGHTS,
                    exploration_radius=EXPLORATION_RADIUS, boundary_buffer=BOUNDARY_BUFFER,
                    optimizer=optimizer)
    bp = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)
    prev = bp[-1]
    rounds = []
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, w_explore=W_EXPLORE)
        proposed = params_from_spec(spec)
        params = with_dims({**prev, **proposed})
        u = agent.predict_uncertainty(params, dm)
        run_experiment(dataset, agent, fab, params, f"explore_{i+1:02d}")
        dm.update()
        agent.train(dm, validate=False)
        prev = params
        rounds.append({"water": params["water_ratio"], "speed": params["print_speed"],
                        "u": u, "score": agent.calibration_system.last_opt_score,
                        "nfev": agent.calibration_system.last_opt_nfev})
    return rounds, bp


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    plot_dir = ensure_plot_dir()

    print(f"\n  Exploration: {N_BASELINE} baseline + {N_EXPLORE} rounds, w_explore={W_EXPLORE}")

    results = {}
    baseline_pts = {}
    for opt, tag in [(Optimizer.LBFGSB, "lbfgsb"), (Optimizer.DE, "de")]:
        print(f"\n  [{opt.value}]")
        rounds, bp = _run_exploration(opt, tag)
        results[tag] = rounds
        baseline_pts[tag] = bp
        for r in rounds:
            print(f"    w={r['water']:.3f}  spd={r['speed']:.1f}  u={r['u']:.3f}  "
                  f"score={r['score']:.3f}  nfev={r['nfev']}")

    out = os.path.join(plot_dir, "05_optimizer_comparison.png")
    plot_optimizer_comparison(out, results, baseline_pts,
                              title=f"L-BFGS-B vs DE (w_explore={W_EXPLORE})")
    print(f"\n  Saved: {out}")

    # Acquisition topology at round 1 and round N
    agent, fab, dataset = make_env("05_topo", verbose=False)
    agent.configure(performance_weights=PERF_WEIGHTS,
                    exploration_radius=EXPLORATION_RADIUS, boundary_buffer=BOUNDARY_BUFFER,
                    optimizer=Optimizer.DE)
    bp = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    w1, s1, p1, u1, c1 = _compute_acquisition_grid(agent, dm, W_EXPLORE, RESOLUTION)
    out = os.path.join(plot_dir, "05_acquisition_round1.png")
    plot_acquisition_topology(out, w1, s1, p1, u1, c1, title="Acquisition — Round 1")
    print(f"  Saved: {out}")

    prev = bp[-1]
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, w_explore=W_EXPLORE)
        p = params_from_spec(spec)
        prev = with_dims({**prev, **p})
        run_experiment(dataset, agent, fab, prev, f"explore_{i+1:02d}")
        dm.update()
        agent.train(dm, validate=False)

    wn, sn, pn, un, cn = _compute_acquisition_grid(agent, dm, W_EXPLORE, RESOLUTION)
    out = os.path.join(plot_dir, f"05_acquisition_round{N_EXPLORE}.png")
    plot_acquisition_topology(out, wn, sn, pn, un, cn, title=f"Acquisition — Round {N_EXPLORE}")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
