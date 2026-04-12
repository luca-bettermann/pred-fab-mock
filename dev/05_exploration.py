"""05 — Exploration Objective + Optimizer Validation.

Test the combined acquisition function and compare L-BFGS-B vs DE.
"""

import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.orchestration import Optimizer
from pred_fab import combined_score
from sensors.physics import N_LAYERS, N_SEGMENTS
from visualization import plot_optimizer_comparison, plot_acquisition_topology
from visualization.helpers import evaluate_physics_grid
from shared import make_env, run_baseline, train_models, with_dims, run_experiment, ensure_plot_dir, clean_plots
from utils import params_from_spec

N_BASELINE = 5
N_EXPLORE = 10
PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
RESOLUTION = 30
KAPPA = 0.5
EXPLORATION_RADIUS = 0.25



def _compute_acquisition_grid(agent, dm, kappa, res):
    """Compute performance, uncertainty (with buffer), and combined grids.

    Uses the calibration system's perf range for normalization — matching
    what the optimizer actually sees.
    """
    waters = np.linspace(0.30, 0.50, res)
    speeds = np.linspace(20.0, 60.0, res)
    perf_grid = np.zeros((res, res))
    unc_grid = np.zeros((res, res))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            p = {"water_ratio": w, "print_speed": spd, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
            try:
                perf = agent.predict_performance(p)
                perf_grid[j, i] = combined_score(perf, PERF_WEIGHTS)
            except Exception:
                perf_grid[j, i] = 0.0
            unc_grid[j, i] = agent.predict_uncertainty(p, dm)  # includes buffer

    # Normalize performance using training-data range (same as optimizer)
    cal = agent.calibration_system
    if cal._perf_range_min is not None and cal._perf_range_max is not None:
        p_min, p_max = cal._perf_range_min, cal._perf_range_max
    else:
        p_min, p_max = perf_grid.min(), perf_grid.max()
    span = max(p_max - p_min, 1e-10)
    p_norm = np.clip((perf_grid - p_min) / span, 0, 1)

    # Uncertainty is already [0,1] with buffer — no renormalization
    combined = (1 - kappa) * p_norm + kappa * unc_grid
    return waters, speeds, p_norm, unc_grid, combined


def _run_exploration(optimizer, tag):
    agent, fab, dataset = make_env(f"05_{tag}", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    agent.configure_exploration(radius=EXPLORATION_RADIUS)
    agent.configure_optimizer(backend=optimizer)
    bp = run_baseline(agent, fab, dataset, N_BASELINE)
    dm, _ = train_models(agent, dataset, val_size=0.0)
    prev = bp[-1]
    rounds = []
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(dm, kappa=KAPPA)
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

    print(f"\n  Exploration: {N_BASELINE} baseline + {N_EXPLORE} rounds, kappa={KAPPA}")

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
                              title=f"L-BFGS-B vs DE (kappa={KAPPA})")
    print(f"\n  Saved: {out}")

    # Physics optimum for reference
    _, _, phys_metrics = evaluate_physics_grid(50, PERF_WEIGHTS)
    combined_phys = list(phys_metrics.values())[-1]
    opt_idx = np.unravel_index(np.argmax(combined_phys), combined_phys.shape)
    phys_waters = np.linspace(0.30, 0.50, 50)
    phys_speeds = np.linspace(20.0, 60.0, 50)
    optimum = (phys_waters[opt_idx[1]], phys_speeds[opt_idx[0]])

    # Per-round acquisition topology plots (one subfolder per optimizer)
    for opt, tag in [(Optimizer.LBFGSB, "lbfgsb"), (Optimizer.DE, "de")]:
        subfolder = f"05_rounds_{tag}"
        clean_plots(subfolder)
        rounds_dir = os.path.join(plot_dir, subfolder)

        agent, fab, dataset = make_env(f"05_topo_{tag}", verbose=False)
        agent.configure_performance(weights=PERF_WEIGHTS)
        agent.configure_exploration(radius=EXPLORATION_RADIUS)
        agent.configure_optimizer(backend=opt)
        bp = run_baseline(agent, fab, dataset, N_BASELINE)
        dm, _ = train_models(agent, dataset, val_size=0.0)
        all_pts = list(bp)

        # Round 0 (before first exploration)
        w0, s0, p0, u0, c0 = _compute_acquisition_grid(agent, dm, KAPPA, RESOLUTION)
        spec = agent.exploration_step(dm, kappa=KAPPA)
        proposed = with_dims(params_from_spec(spec))
        out = os.path.join(rounds_dir, "round_00.png")
        plot_acquisition_topology(out, w0, s0, p0, u0, c0,
                                  experiment_pts=all_pts, proposed=proposed, optimum=optimum,
                                  title=f"{tag.upper()} — Round 0")

        prev = bp[-1]
        for i in range(N_EXPLORE):
            p = params_from_spec(spec)
            prev = with_dims({**prev, **p})
            run_experiment(dataset, agent, fab, prev, f"explore_{i+1:02d}")
            all_pts.append(prev)
            dm.update()
            agent.train(dm, validate=False)

            wi, si, pi, ui, ci = _compute_acquisition_grid(agent, dm, KAPPA, RESOLUTION)
            if i < N_EXPLORE - 1:
                spec = agent.exploration_step(dm, kappa=KAPPA)
                proposed = with_dims(params_from_spec(spec))
            else:
                proposed = None

            out = os.path.join(rounds_dir, f"round_{i+1:02d}.png")
            plot_acquisition_topology(out, wi, si, pi, ui, ci,
                                      experiment_pts=all_pts, proposed=proposed, optimum=optimum,
                                      title=f"{tag.upper()} — Round {i+1}")

        print(f"\n  Saved: {rounds_dir}/ ({N_EXPLORE + 1} plots)")


if __name__ == "__main__":
    main()
