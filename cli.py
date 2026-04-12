"""CLI for the PFAB mock — run each phase as a separate command.

Usage:
    python cli.py reset
    python cli.py configure --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'
    python cli.py baseline --n 10
    python cli.py explore --n 10 --kappa 0.5
    python cli.py infer --n-layers 5
    python cli.py summary
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from pred_fab.orchestration import Optimizer
from pred_fab.core import Dataset
from pred_fab import combined_score

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from sensors.physics import N_LAYERS, N_SEGMENTS
from utils import params_from_spec, get_performance
from workflow import JourneyState, with_dimensions, run_and_evaluate


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


SESSION_FILE = ".pfab_session.json"
PLOT_DIR = "./plots"


def _next_code(state: JourneyState, prefix: str) -> str:
    existing = [c for c in state.all_codes if c.startswith(prefix + "_")]
    return f"{prefix}_{len(existing) + 1:02d}"


# ── Session persistence ──────────────────────────────────────────────────────

def _save_session(config: dict[str, Any], journey: JourneyState) -> None:
    data = {
        "config": config,
        "journey": {
            "all_params": journey.all_params,
            "all_phases": journey.all_phases,
            "all_codes": journey.all_codes,
            "perf_history": [(p, pf) for p, pf in journey.perf_history],
            "prev_params": journey.prev_params,
        },
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(_to_native(data), f, indent=2)


def _load_session() -> tuple[dict[str, Any], JourneyState]:
    if not os.path.exists(SESSION_FILE):
        print("No session found. Run 'python cli.py configure' first.")
        sys.exit(1)
    with open(SESSION_FILE) as f:
        data = json.load(f)
    config = data["config"]
    j = data["journey"]
    state = JourneyState()
    state.all_params = j["all_params"]
    state.all_phases = j["all_phases"]
    state.all_codes = j["all_codes"]
    state.perf_history = [(p, pf) for p, pf in j["perf_history"]]
    state.prev_params = j["prev_params"]
    return config, state


def _rebuild(config: dict[str, Any], verbose: bool = False) -> tuple[Any, Dataset, FabricationSystem]:
    """Reconstruct agent + dataset + fab from session config."""
    if not verbose:
        _real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    try:
        schema = build_schema()
        fab = FabricationSystem(CameraSystem(), EnergySensor())
        agent = build_agent(schema, fab.camera, fab.energy, verbose=verbose)

        # Apply configuration using explicit configure_* methods
        if config.get("performance_weights"):
            agent.configure_performance(weights=config["performance_weights"])
        if config.get("exploration_radius") is not None:
            agent.configure_exploration(radius=config["exploration_radius"])

        opt_kwargs: dict[str, Any] = {}
        if config.get("optimizer"):
            opt_kwargs["backend"] = Optimizer(config["optimizer"])
        if config.get("de_maxiter") is not None:
            opt_kwargs["de_maxiter"] = config["de_maxiter"]
        if config.get("de_popsize") is not None:
            opt_kwargs["de_popsize"] = config["de_popsize"]
        if opt_kwargs:
            agent.configure_optimizer(**opt_kwargs)

        if config.get("bounds"):
            bounds = {k: tuple(v) for k, v in config["bounds"].items()}
            agent.calibration_system.configure_param_bounds(bounds)

        traj_kwargs: dict[str, Any] = {}
        if config.get("mpc_lookahead") is not None:
            traj_kwargs["mpc_lookahead"] = config["mpc_lookahead"]
        if config.get("mpc_discount") is not None:
            traj_kwargs["mpc_discount"] = config["mpc_discount"]
        if traj_kwargs:
            agent.configure_trajectory(**traj_kwargs)

        dataset = Dataset(schema=schema)
        dataset.populate()
    finally:
        if not verbose:
            sys.stdout.close()
            sys.stdout = _real_stdout

    return agent, dataset, fab


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _ensure_plot_dir() -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR


def _compute_acquisition_grid(agent, dm, kappa, res=30):
    """Compute normalized performance, uncertainty, and combined grids."""
    waters = np.linspace(0.30, 0.50, res)
    speeds = np.linspace(20.0, 60.0, res)
    perf_grid = np.zeros((res, res))
    unc_grid = np.zeros((res, res))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            p = {"water_ratio": w, "print_speed": spd, "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}
            try:
                perf = agent.predict_performance(p)
                pw = agent.calibration_system.performance_weights
                perf_grid[j, i] = combined_score(perf, pw)
            except Exception:
                perf_grid[j, i] = 0.0
            unc_grid[j, i] = agent.predict_uncertainty(p, dm)

    # Normalize perf using calibration range
    cal = agent.calibration_system
    if cal._perf_range_min is not None and cal._perf_range_max is not None:
        p_min, p_max = cal._perf_range_min, cal._perf_range_max
    else:
        p_min, p_max = perf_grid.min(), perf_grid.max()
    span = max(p_max - p_min, 1e-10)
    p_norm = np.clip((perf_grid - p_min) / span, 0, 1)

    combined = (1 - kappa) * p_norm + kappa * unc_grid
    return waters, speeds, p_norm, unc_grid, combined


def _get_physics_optimum(perf_weights=None, n_layers=N_LAYERS):
    """Find the physics optimum location."""
    from visualization.helpers import evaluate_physics_grid
    _, _, phys_metrics = evaluate_physics_grid(50, perf_weights, n_layers=n_layers)
    combined = list(phys_metrics.values())[-1]
    opt_idx = np.unravel_index(np.argmax(combined), combined.shape)
    phys_waters = np.linspace(0.30, 0.50, 50)
    phys_speeds = np.linspace(20.0, 60.0, 50)
    return (phys_waters[opt_idx[1]], phys_speeds[opt_idx[0]])


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_configure(args: argparse.Namespace) -> None:
    """Set up agent configuration and save session."""
    config: dict[str, Any] = {}

    if args.bounds:
        config["bounds"] = json.loads(args.bounds)
    if args.weights:
        config["performance_weights"] = json.loads(args.weights)
    if args.optimizer:
        config["optimizer"] = args.optimizer
    if args.radius is not None:
        config["exploration_radius"] = args.radius
    if args.de_maxiter is not None:
        config["de_maxiter"] = args.de_maxiter
    if args.de_popsize is not None:
        config["de_popsize"] = args.de_popsize

    agent, dataset, fab = _rebuild(config, verbose=True)
    state = JourneyState()
    _save_session(config, state)


def cmd_baseline(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = config.get("performance_weights")
    plot_dir = _ensure_plot_dir()

    agent.console.print_phase_header(1, "Baseline", f"{args.n} experiments")
    specs = agent.baseline_step(n=args.n)

    for spec in specs:
        params = with_dimensions(params_from_spec(spec))
        exp_code = _next_code(state, "baseline")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("baseline", exp_code, params, perf)
        agent.console.print_experiment_row(exp_code, params, perf)

    state.prev_params = with_dimensions(params_from_spec(specs[-1]))

    # ── Plots ──
    from visualization import plot_physics_topology, plot_baseline_scatter, plot_topology_comparison
    from visualization.helpers import physics_combined_at

    # 1. Physics topology (ground truth)
    path = os.path.join(plot_dir, "01_physics_topology.png")
    plot_physics_topology(path, perf_weights=perf_weights)
    print(f"\n  Plot: {path}")

    # 2. Baseline scatter
    path = os.path.join(plot_dir, "02_baseline_scatter.png")
    plot_baseline_scatter(path, state.all_params)
    print(f"  Plot: {path}")

    # 3. Initial model quality — train and compare topology
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    true_grid = np.array([[physics_combined_at(w, spd, perf_weights) for w in waters] for spd in speeds])
    pred_grid = np.zeros_like(true_grid)
    pw = perf_weights or {}
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(perf, pw)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "03_initial_topology.png")
    plot_topology_comparison(path, waters, speeds,
                              {"Ground Truth": true_grid, f"Model ({args.n} baseline)": pred_grid},
                              title="Initial Model vs Ground Truth")
    print(f"  Plot: {path}")

    _save_session(config, state)


def cmd_train(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=args.val_size)
    results = agent.train(dm, validate=args.val_size > 0)

    _save_session(config, state)


def cmd_explore(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = config.get("performance_weights")
    plot_dir = _ensure_plot_dir()

    agent.console.print_phase_header(2, "Exploration",
                                      f"{args.n} rounds (\u03ba={args.kappa})")
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    optimum = _get_physics_optimum(perf_weights)

    for i in range(args.n):
        spec = agent.exploration_step(dm, kappa=args.kappa)
        proposed = params_from_spec(spec)
        params = with_dimensions({**state.prev_params, **proposed})
        exp_code = _next_code(state, "explore")

        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("exploration", exp_code, params, perf)

        dm.update()
        agent.train(dm, validate=False)

        # Per-round acquisition plot
        if i < 5 or i == args.n - 1 or (i + 1) % 5 == 0:
            from visualization import plot_acquisition_topology
            w, s, p, u, c = _compute_acquisition_grid(agent, dm, args.kappa, res=30)
            path = os.path.join(plot_dir, f"04_explore_round_{i+1:02d}.png")
            plot_acquisition_topology(path, w, s, p, u, c,
                                      experiment_pts=state.all_params,
                                      optimum=optimum,
                                      title=f"Exploration — Round {i+1}")

    # Final topology comparison
    from visualization import plot_topology_comparison
    from visualization.helpers import physics_combined_at
    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    true_grid = np.array([[physics_combined_at(w, spd, perf_weights) for w in waters] for spd in speeds])
    pred_grid = np.zeros_like(true_grid)
    pw = perf_weights or {}
    for i_w, w in enumerate(waters):
        for j_s, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j_s, i_w] = combined_score(perf, pw)
            except Exception:
                pred_grid[j_s, i_w] = 0.0

    n_total = len(state.all_params)
    path = os.path.join(plot_dir, "05_final_topology.png")
    plot_topology_comparison(path, waters, speeds,
                              {"Ground Truth": true_grid, f"Model ({n_total} experiments)": pred_grid},
                              title="Model After Exploration vs Ground Truth")
    print(f"\n  Plot: {path}")
    print(f"  Round plots: {plot_dir}/04_explore_round_*.png")

    _save_session(config, state)


def cmd_infer(args: argparse.Namespace) -> None:
    """Single-shot inference — first-time-right manufacturing."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = config.get("performance_weights")
    plot_dir = _ensure_plot_dir()
    n_layers = args.n_layers or N_LAYERS

    agent.console.print_phase_header(3, "Inference",
                                      f"First-time-right (n_layers={n_layers})")

    # Fix n_layers to the design intent
    agent.calibration_system.configure_fixed_params({"n_layers": n_layers}, force=True)

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    params["n_layers"] = n_layers  # ensure design intent
    exp_code = _next_code(state, "infer")

    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
    perf = get_performance(exp_data)
    state.record("inference", exp_code, params, perf)

    # Show result
    pw = perf_weights or {}
    score = combined_score(perf, pw)
    print(f"\n  Inference result (n_layers={n_layers}):")
    print(f"    water_ratio  = {params['water_ratio']:.3f}")
    print(f"    print_speed  = {params['print_speed']:.1f} mm/s")
    for k, v in perf.items():
        print(f"    {k:<15s} = {v:.3f}")
    print(f"    combined     = {score:.3f}")

    # Physics optimum for comparison
    opt_w, opt_s = _get_physics_optimum(perf_weights, n_layers=n_layers)
    from visualization.helpers import physics_combined_at
    opt_score = physics_combined_at(opt_w, opt_s, perf_weights, n_layers=n_layers)
    print(f"\n  Physics optimum (n_layers={n_layers}):")
    print(f"    water_ratio  = {opt_w:.3f}")
    print(f"    print_speed  = {opt_s:.1f} mm/s")
    print(f"    combined     = {opt_score:.3f}")
    print(f"    gap          = {opt_score - score:.3f}")

    _save_session(config, state)


def cmd_summary(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = config.get("performance_weights", {})

    print("\n  Run Summary:")
    print(f"  {'─' * 60}")
    print(f"  {'Phase':<15s}  {'Experiments':>11s}  {'Best Combined':>14s}")
    print(f"  {'─' * 60}")

    for phase in ["baseline", "exploration", "inference"]:
        indices = [i for i, p in enumerate(state.all_phases) if p == phase]
        if not indices:
            continue
        scores = [combined_score(state.perf_history[i][1], perf_weights)
                  for i in indices]
        best = max(scores)
        print(f"  {phase:<15s}  {len(indices):>11d}  {best:>14.3f}")

    print(f"  {'─' * 60}")
    agent.console.print_done(PLOT_DIR)


def cmd_reset(args: argparse.Namespace) -> None:
    import shutil
    for path in [SESSION_FILE, "./local", "./plots", "./logs"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  Removed {path}")
    print("Session reset.")


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="PFAB mock CLI — run fabrication phases step by step",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # configure
    p = sub.add_parser("configure", help="Set up agent configuration")
    p.add_argument("--bounds", type=str, help='JSON: {"param": [lo, hi], ...}')
    p.add_argument("--weights", type=str, help='JSON: {"perf_attr": weight, ...}')
    p.add_argument("--optimizer", choices=["lbfgsb", "de"], default=None)
    p.add_argument("--radius", type=float, default=None, help="Exploration radius")
    p.add_argument("--de-maxiter", type=int, default=None, help="DE max generations")
    p.add_argument("--de-popsize", type=int, default=None, help="DE population size")
    p.set_defaults(func=cmd_configure)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline experiments")
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(func=cmd_baseline)

    # train
    p = sub.add_parser("train", help="Train prediction models")
    p.add_argument("--val-size", type=float, default=0.25)
    p.set_defaults(func=cmd_train)

    # explore
    p = sub.add_parser("explore", help="Run exploration rounds")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--kappa", type=float, default=0.5, help="Exploration weight (0=exploit, 1=explore)")
    p.set_defaults(func=cmd_explore)

    # infer
    p = sub.add_parser("infer", help="Single-shot inference")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Design intent: fix n_layers for inference (default: schema default)")
    p.set_defaults(func=cmd_infer)

    # summary
    p = sub.add_parser("summary", help="Print run summary")
    p.set_defaults(func=cmd_summary)

    # reset
    p = sub.add_parser("reset", help="Clear session state and data")
    p.set_defaults(func=cmd_reset)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
