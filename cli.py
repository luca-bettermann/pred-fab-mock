"""CLI for the PFAB mock — step-by-step predictive fabrication workflow.

Quick start:
    uv run cli.py reset
    uv run cli.py init-schema
    uv run cli.py init-agent
    uv run cli.py configure --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'
    uv run cli.py init-physics --seed 42 --plot
    uv run cli.py baseline --n 10 --plot
    uv run cli.py explore --n 5 --kappa 0.5 --plot
    uv run cli.py test-set --n 20
    uv run cli.py analyse --plot
    uv run cli.py inference --design-intent '{"n_layers":5}' --plot
    uv run cli.py summary
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
from cli_helpers import (
    show_plot, randomize_physics, apply_physics_config, load_physics_from_session,
    PHYSICS_CONFIG_KEY, generate_test_params, compute_local_sensitivity,
)


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
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
        print("No session found. Run 'uv run cli.py init-schema' first.")
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
    # Apply physics config before building anything
    load_physics_from_session(config)

    if not verbose:
        _real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    try:
        schema = build_schema()
        fab = FabricationSystem(CameraSystem(), EnergySensor())
        model_type = config.get("model_type", "mlp")
        agent = build_agent(schema, fab.camera, fab.energy, model_type=model_type, verbose=verbose)

        if config.get("performance_weights"):
            agent.configure_performance(weights=config["performance_weights"])

        explore_kwargs: dict[str, Any] = {}
        if config.get("exploration_radius") is not None:
            explore_kwargs["radius"] = config["exploration_radius"]
        if config.get("buffer") is not None:
            explore_kwargs["buffer"] = config["buffer"]
        if config.get("decay_exp") is not None:
            explore_kwargs["decay_exp"] = config["decay_exp"]
        if explore_kwargs:
            agent.configure_exploration(**explore_kwargs)

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

        dataset = Dataset(schema=schema)
        dataset.populate()
    finally:
        if not verbose:
            sys.stdout.close()
            sys.stdout = _real_stdout

    return agent, dataset, fab


def _ensure_plot_dir() -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR


def _get_physics_optimum(perf_weights=None, n_layers=N_LAYERS):
    """Find the physics optimum location."""
    from visualization.helpers import evaluate_physics_grid
    _, _, phys_metrics = evaluate_physics_grid(50, perf_weights, n_layers=n_layers)
    combined = list(phys_metrics.values())[-1]
    opt_idx = np.unravel_index(np.argmax(combined), combined.shape)
    phys_waters = np.linspace(0.30, 0.50, 50)
    phys_speeds = np.linspace(20.0, 60.0, 50)
    return (phys_waters[opt_idx[1]], phys_speeds[opt_idx[0]])


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

    cal = agent.calibration_system
    if cal._perf_range_min is not None and cal._perf_range_max is not None:
        p_min, p_max = cal._perf_range_min, cal._perf_range_max
    else:
        p_min, p_max = perf_grid.min(), perf_grid.max()
    span = max(p_max - p_min, 1e-10)
    p_norm = np.clip((perf_grid - p_min) / span, 0, 1)

    combined = (1 - kappa) * p_norm + kappa * unc_grid
    return waters, speeds, p_norm, unc_grid, combined


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_reset(args: argparse.Namespace) -> None:
    """Clear all session state, data, and plots."""
    import shutil
    for path in [SESSION_FILE, "./local", "./plots", "./logs"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  Removed {path}")
    print("  Session reset.")


def cmd_init_schema(args: argparse.Namespace) -> None:
    """Show the problem schema."""
    from schema import SCHEMA_TITLE
    config: dict[str, Any] = {}
    state = JourneyState()

    _B = "\033[1m"
    _C = "\033[36m"
    _R = "\033[0m"
    _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 0.1{_R}{_B} ▸ Schema{_R}")
    print(f"  {_D}{SCHEMA_TITLE}{_R}")
    print(f"{_B}{_C}{bar}{_R}")

    schema = build_schema()
    schema.state_report()

    _save_session(config, state)


def cmd_init_agent(args: argparse.Namespace) -> None:
    """Initialize the agent and show its state."""
    config, state = _load_session()

    model_type = args.model
    config["model_type"] = model_type

    _B = "\033[1m"
    _C = "\033[36m"
    _R = "\033[0m"
    _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 0.2{_R}{_B} ▸ Agent{_R}")
    print(f"  {_D}model={model_type}{_R}")
    print(f"{_B}{_C}{bar}{_R}")

    agent, _, _ = _rebuild(config)
    agent.state_report()

    _save_session(config, state)


def cmd_init_physics(args: argparse.Namespace) -> None:
    """Randomize physics constants and show the ground truth topology."""
    config, state = _load_session()
    plot_dir = _ensure_plot_dir()

    _B = "\033[1m"
    _C = "\033[36m"
    _R = "\033[0m"
    _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 0.3{_R}{_B} ▸ Physics{_R}")
    seed_str = f"seed={args.seed}" if args.seed is not None else "random"
    print(f"  {_D}Randomize ground truth ({seed_str}){_R}")
    print(f"{_B}{_C}{bar}{_R}")

    seed = args.seed
    physics = randomize_physics(seed)
    config[PHYSICS_CONFIG_KEY] = physics
    apply_physics_config(physics)

    print(f"\n  Physics constants:")
    for key, val in physics.items():
        if isinstance(val, list):
            print(f"    {key:<25s} = [{', '.join(f'{v:.3f}' for v in val)}]")
        else:
            print(f"    {key:<25s} = {val:.6f}")

    # Show topology
    from visualization import plot_physics_topology
    perf_weights = config.get("performance_weights")
    path = os.path.join(plot_dir, "00_physics_topology.png")
    plot_physics_topology(path, perf_weights=perf_weights)
    show_plot(path, inline=args.plot)

    _save_session(config, state)


def _print_config_set(label: str, value: Any) -> None:
    """Print a configuration confirmation line."""
    _G = "\033[32m"
    _D = "\033[2m"
    _R = "\033[0m"
    if isinstance(value, dict):
        print(f"\n  {_D}{label}{_R}")
        for k, v in value.items():
            print(f"  {_G}✓{_R} {k} = {v}")
    else:
        print(f"  {_G}✓{_R} {label} = {value}")


def cmd_configure(args: argparse.Namespace) -> None:
    """Set agent configuration."""
    config, state = _load_session()

    if args.weights:
        config["performance_weights"] = json.loads(args.weights)
        _print_config_set("Weights", config["performance_weights"])
    if args.bounds:
        config["bounds"] = json.loads(args.bounds)
        _print_config_set("Bounds", config["bounds"])
    if args.optimizer:
        config["optimizer"] = args.optimizer
        _print_config_set("Optimizer", args.optimizer)
    if args.radius is not None:
        config["exploration_radius"] = args.radius
        _print_config_set("Exploration radius", args.radius)
    if args.buffer is not None:
        config["buffer"] = args.buffer
        _print_config_set("Buffer", args.buffer)
    if args.decay_exp is not None:
        config["decay_exp"] = args.decay_exp
        _print_config_set("Decay exponent", args.decay_exp)
    if args.de_maxiter is not None:
        config["de_maxiter"] = args.de_maxiter
        _print_config_set("DE max iterations", args.de_maxiter)
    if args.de_popsize is not None:
        config["de_popsize"] = args.de_popsize
        _print_config_set("DE population size", args.de_popsize)

    print()
    _save_session(config, state)


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run baseline experiments (space-filling, no model)."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    plot_dir = _ensure_plot_dir()

    agent.console.print_phase_header(1, "Baseline", f"{args.n} experiments")
    specs = agent.baseline_step(n=args.n)
    nfev_suffix = f"nfev={agent.last_baseline_nfev}"

    for spec in specs:
        params = with_dimensions(params_from_spec(spec))
        exp_code = _next_code(state, "baseline")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("baseline", exp_code, params, perf)
        agent.console.print_experiment_row(exp_code, params, perf, suffix=nfev_suffix)

    state.prev_params = with_dimensions(params_from_spec(specs[-1]))

    # 3D filament view of a single experiment (last baseline)
    from visualization import plot_path_comparison_3d
    last_params = state.all_params[-1]
    path_3d = os.path.join(plot_dir, "01_path_deviation_3d.png")
    plot_path_comparison_3d(path_3d, fab.camera, last_params, exp_code=state.all_codes[-1])
    show_plot(path_3d, inline=args.plot)

    # Train initial model for topology comparison
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    # Combined 1x3 plot: parameter space + ground truth + initial model
    from visualization import plot_baseline_overview
    from visualization.helpers import physics_combined_at

    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    pw = agent.calibration_system.performance_weights
    true_grid = np.array([[physics_combined_at(w, spd, pw) for w in waters] for spd in speeds])
    pred_grid = np.zeros_like(true_grid)
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(perf, pw)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "01_baseline.png")
    plot_baseline_overview(path, state.all_params, waters, speeds,
                           true_grid, pred_grid, n_baseline=args.n)
    show_plot(path, inline=args.plot)

    _save_session(config, state)


def cmd_explore(args: argparse.Namespace) -> None:
    """Run exploration rounds (incremental — can be called multiple times)."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    plot_dir = _ensure_plot_dir()

    n_existing = len([p for p in state.all_phases if p == "exploration"])
    total_after = n_existing + args.n

    agent.console.print_phase_header(2, "Exploration",
                                      f"rounds {n_existing+1}..{total_after} (\u03ba={args.kappa})")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=args.validate)

    for i in range(args.n):
        round_num = n_existing + i + 1
        spec = agent.exploration_step(dm, kappa=args.kappa)
        proposed = params_from_spec(spec)
        params = with_dimensions({**state.prev_params, **proposed})
        exp_code = _next_code(state, "explore")

        # Snapshot acquisition landscape BEFORE running the experiment
        # (the optimizer chose this point on the current landscape)
        if args.plot:
            from visualization import plot_acquisition_topology
            acq_data = _compute_acquisition_grid(agent, dm, args.kappa, res=30)

        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("exploration", exp_code, params, perf)

        # Plot the pre-retrain landscape with the proposed point
        if args.plot:
            w, s, p, u, c = acq_data
            path = os.path.join(plot_dir, f"03_explore_round_{round_num:02d}.png")
            plot_acquisition_topology(path, w, s, p, u, c,
                                      experiment_pts=state.all_params[:-1],
                                      proposed=params,
                                      title=f"Exploration \u2014 Round {round_num}")
            show_plot(path, inline=True)

        dm.update()
        agent.train(dm, validate=False)

    _save_session(config, state)


def cmd_test_set(args: argparse.Namespace) -> None:
    """Create a held-out test set for model evaluation."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    load_physics_from_session(config)

    test_params = generate_test_params(args.n)
    print(f"\n  Creating {len(test_params)} test experiments...")

    for i, params in enumerate(test_params):
        exp_code = f"test_{i+1:02d}"
        if dataset.has_experiment(exp_code):
            continue
        params = with_dimensions(params)
        run_and_evaluate(dataset, agent, fab, params, exp_code)

    print(f"  Test set: {len(test_params)} experiments (test_01..test_{len(test_params):02d})")
    config["test_set_n"] = len(test_params)
    _save_session(config, state)


def cmd_analyse(args: argparse.Namespace) -> None:
    """Evaluate the prediction model on the test set and show sensitivity analysis."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = agent.calibration_system.performance_weights
    plot_dir = _ensure_plot_dir()

    # Train on all non-test data
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    # Evaluate on test set
    test_codes = [c for c in dataset.get_populated_experiment_codes() if c.startswith("test_")]
    if not test_codes:
        print("  No test set found. Run 'uv run cli.py test-set --n 20' first.")
        return

    print(f"\n  Evaluating model on {len(test_codes)} test experiments:")
    errors = []
    for code in sorted(test_codes):
        exp = dataset.get_experiment(code)
        params = exp.parameters.get_values_dict()
        true_perf = {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}
        true_score = combined_score(true_perf, perf_weights)

        try:
            pred_perf = agent.predict_performance(params)
            pred_score = combined_score(pred_perf, perf_weights)
        except Exception:
            pred_score = 0.0

        errors.append(abs(true_score - pred_score))

    mae = np.mean(errors)
    print(f"    MAE (combined score): {mae:.4f}")
    print(f"    Max error:            {max(errors):.4f}")

    # Topology comparison plot
    from visualization import plot_topology_comparison
    from visualization.helpers import physics_combined_at, evaluate_physics_grid
    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    true_grid = np.array([[physics_combined_at(w, spd, perf_weights) for w in waters] for spd in speeds])
    pred_grid = np.zeros_like(true_grid)
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                   "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(perf, perf_weights)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "05_analysis_topology.png")
    plot_topology_comparison(path, waters, speeds,
                              {"Ground Truth": true_grid, "Model Prediction": pred_grid},
                              title="Model Analysis on Test Set")
    show_plot(path, inline=args.plot)

    # Sensitivity analysis at predicted optimum
    opt_w, opt_s = _get_physics_optimum(perf_weights)
    opt_params = {"water_ratio": opt_w, "print_speed": opt_s,
                  "n_layers": N_LAYERS, "n_segments": N_SEGMENTS}

    tunable = agent.calibration_system.get_tunable_params(dm)
    sensitivities = compute_local_sensitivity(agent, opt_params, tunable, perf_weights)

    print(f"\n  Local sensitivity at optimum:")
    for code, val in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        print(f"    {code:<20s}  {val:.4f}")


def cmd_inference(args: argparse.Namespace) -> None:
    """Single-shot inference with design intent."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = agent.calibration_system.performance_weights
    plot_dir = _ensure_plot_dir()

    # Parse design intent
    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_layers = design_intent.get("n_layers", N_LAYERS)

    agent.console.print_phase_header(3, "Inference", "First-time-right")

    if design_intent:
        parts = [f"{k}={v}" for k, v in design_intent.items()]
        print(f"  Design intent: {', '.join(parts)}")
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    params.update(design_intent)
    exp_code = _next_code(state, "infer")

    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
    perf = get_performance(exp_data)
    state.record("inference", exp_code, params, perf)

    score = combined_score(perf, perf_weights)
    print(f"\n  Proposed parameters:")
    print(f"    water_ratio  = {params['water_ratio']:.3f}")
    print(f"    print_speed  = {params['print_speed']:.1f} mm/s")
    for k, v in design_intent.items():
        print(f"    {k:<13s} = {v}  (design intent)")
    print(f"\n  Performance:")
    for k, v in perf.items():
        print(f"    {k:<20s} = {v:.3f}")
    print(f"    {'combined':<20s} = {score:.3f}")

    # Compare to physics optimum
    from visualization.helpers import physics_combined_at
    opt_w, opt_s = _get_physics_optimum(perf_weights, n_layers=n_layers)
    opt_score = physics_combined_at(opt_w, opt_s, perf_weights, n_layers=n_layers)
    gap = opt_score - score
    print(f"\n  Physics optimum: combined={opt_score:.3f} (gap={gap:+.3f})")

    # Prediction grid for inference topology
    from visualization import plot_inference_result
    waters = np.linspace(0.30, 0.50, 40)
    speeds = np.linspace(20.0, 60.0, 40)
    pred_grid = np.zeros((40, 40))
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                p = agent.predict_performance({"water_ratio": w, "print_speed": spd,
                                                "n_layers": N_LAYERS, "n_segments": N_SEGMENTS})
                pred_grid[j, i] = combined_score(p, perf_weights)
            except Exception:
                pred_grid[j, i] = 0.0

    path = os.path.join(plot_dir, "05_inference.png")
    plot_inference_result(
        path, waters, speeds, pred_grid,
        proposed_water=params["water_ratio"],
        proposed_speed=params["print_speed"],
        proposed_score=score,
        opt_water=opt_w,
        opt_speed=opt_s,
        opt_score=opt_score,
        experiment_pts=state.all_params,
    )
    show_plot(path, inline=args.plot)

    _save_session(config, state)


# ── Advanced commands ────────────────────────────────────────────────────────

def cmd_explore_trajectory(args: argparse.Namespace) -> None:
    """Trajectory exploration: per-layer speed optimization with MPC lookahead."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    # Configure trajectory
    agent.configure_trajectory(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta={"print_speed": args.delta},
        smoothing=args.smoothing,
        mpc_lookahead=args.lookahead,
        mpc_discount=args.discount,
    )

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_layers = design_intent.get("n_layers", N_LAYERS)
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(4, "Trajectory Exploration",
                                      f"{args.n} rounds, \u0394speed=\u00b1{args.delta} mm/s")
    if design_intent:
        parts = [f"{k}={v}" for k, v in design_intent.items()]
        print(f"  Design intent: {', '.join(parts)}")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    prev = state.prev_params if state.prev_params else {}

    for i in range(args.n):
        spec = agent.exploration_step(dm, kappa=args.kappa, current_params=prev)
        proposed = params_from_spec(spec)
        params = with_dimensions({**prev, **proposed})
        params.update(design_intent)

        # Apply schedules to the experiment
        exp_code = _next_code(state, "traj")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        if spec.schedules:
            spec.apply_schedules(exp_data)

        perf = get_performance(exp_data)
        state.record("trajectory", exp_code, params, perf)

        dm.update()
        agent.train(dm, validate=False)
        prev = params

    _save_session(config, state)


def cmd_adapt(args: argparse.Namespace) -> None:
    """Online inference with layer-by-layer adaptation.

    1. Run inference to get optimal starting parameters
    2. Start fabrication
    3. After each layer: evaluate, tune model, adapt speed for next layer
    """
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    perf_weights = agent.calibration_system.performance_weights

    # Configure for adaptation
    agent.configure_trajectory(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta={"print_speed": args.delta},
    )

    design_intent = json.loads(args.design_intent) if args.design_intent else {}
    n_layers = design_intent.get("n_layers", N_LAYERS)
    if design_intent:
        agent.calibration_system.configure_fixed_params(design_intent, force=True)

    agent.console.print_phase_header(5, "Online Inference",
                                      f"Inference + layer-by-layer adaptation")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    # Step 1: Get initial parameters via inference (kappa=0)
    print(f"\n  Step 1: Initial inference...")
    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    params.update(design_intent)

    print(f"    Starting params: w={params['water_ratio']:.3f}, spd={params['print_speed']:.1f}")

    # Step 2: Create experiment, fabricate layer by layer with adaptation
    exp_code = _next_code(state, "adapt")
    exp_data = dataset.create_experiment(exp_code, parameters=params)

    print(f"\n  Step 2: Fabrication with online adaptation ({n_layers} layers):")
    print(f"    {'Layer':<8s}  {'Speed':>8s}  {'Adapted':>8s}  {'Deviation':>10s}")
    print(f"    {'─' * 38}")

    for layer_idx in range(n_layers):
        # Fabricate this layer
        fab.run_layer(params, layer_idx)

        # Evaluate what we have so far
        agent.evaluate(exp_data)

        speed_before = params["print_speed"]

        # Adapt for next layer (except the last)
        if layer_idx < n_layers - 1:
            agent.set_active_experiment(exp_data)
            adapt_spec = agent.adaptation_step(
                dimension="n_layers",
                step_index=layer_idx + 1,
                exp_data=exp_data,
                mode=__import__("pred_fab.utils", fromlist=["Mode"]).Mode.INFERENCE,
                kappa=0.0,
                record=True,
            )
            new_speed = adapt_spec.initial_params.get("print_speed", speed_before)
            params = {**params, "print_speed": float(new_speed)}
        else:
            new_speed = speed_before

        # Get deviation for this layer
        dev_vals = exp_data.features.get_value("path_deviation")
        if dev_vals is not None and hasattr(dev_vals, '__len__'):
            import numpy as _np
            flat = _np.array(dev_vals).flatten()
            n_segs = int(params.get("n_segments", N_SEGMENTS))
            start_idx = layer_idx * n_segs
            end_idx = min(start_idx + n_segs, len(flat))
            if end_idx > start_idx:
                layer_dev = float(_np.mean(flat[start_idx:end_idx]))
            else:
                layer_dev = 0.0
        else:
            layer_dev = 0.0

        adapted_str = f"{new_speed:.1f}" if new_speed != speed_before else "—"
        print(f"    {layer_idx+1:<8d}  {speed_before:8.1f}  {adapted_str:>8s}  {layer_dev:10.6f}")

    # Save experiment
    dataset.save_experiment(exp_code)
    perf = get_performance(exp_data)
    state.record("adaptation", exp_code, params, perf)

    score = combined_score(perf, perf_weights)
    print(f"\n  Result:")
    for k, v in perf.items():
        print(f"    {k:<20s} = {v:.3f}")
    print(f"    {'combined':<20s} = {score:.3f}")

    _save_session(config, state)


def cmd_summary(args: argparse.Namespace) -> None:
    """Show run summary across all phases."""
    config, state = _load_session()
    perf_weights = config.get("performance_weights") or {
        "path_accuracy": 1, "energy_efficiency": 1, "production_rate": 1,
    }

    print(f"\n  Run Summary:")
    print(f"  {'─' * 60}")
    print(f"  {'Phase':<15s}  {'Experiments':>11s}  {'Best Combined':>14s}")
    print(f"  {'─' * 60}")

    for phase in ["baseline", "exploration", "trajectory", "inference", "adaptation"]:
        indices = [i for i, p in enumerate(state.all_phases) if p == phase]
        if not indices:
            continue
        scores = [combined_score(state.perf_history[i][1], perf_weights)
                  for i in indices]
        best = max(scores)
        print(f"  {phase:<15s}  {len(indices):>11d}  {best:>14.3f}")

    print(f"  {'─' * 60}")
    total = len(state.all_params)
    test_n = config.get("test_set_n", 0)
    print(f"  Total: {total} training experiments + {test_n} test experiments")
    print()


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="PFAB mock CLI — predictive fabrication workflow step by step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  uv run cli.py reset
  uv run cli.py init-schema
  uv run cli.py init-agent
  uv run cli.py configure --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'
  uv run cli.py init-physics --seed 42 --plot
  uv run cli.py baseline --n 10 --plot
  uv run cli.py explore --n 5 --kappa 0.5 --plot
  uv run cli.py test-set --n 20
  uv run cli.py analyse --plot
  uv run cli.py inference --design-intent '{"n_layers":5}' --plot
  uv run cli.py summary
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # reset
    p = sub.add_parser("reset", help="Clear all session state and data")
    p.set_defaults(func=cmd_reset)

    # init-schema
    p = sub.add_parser("init-schema", help="Show the problem schema")
    p.set_defaults(func=cmd_init_schema)

    # init-agent
    p = sub.add_parser("init-agent", help="Initialize the agent")
    p.add_argument("--model", choices=["mlp", "rf"], default="mlp", help="Prediction model type (default: mlp)")
    p.set_defaults(func=cmd_init_agent)

    # init-physics
    p = sub.add_parser("init-physics", help="Randomize physics constants and show topology")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--plot", action="store_true", help="Show plots inline in terminal")
    p.set_defaults(func=cmd_init_physics)

    # configure
    p = sub.add_parser("configure", help="Set agent configuration",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="""
Configuration groups:

  Performance:
    --weights JSON           Performance attribute weights
                             Example: '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'

  Exploration:
    --radius FLOAT           KDE exploration radius (default: 0.15)
    --buffer FLOAT           Normalization buffer for perf/unc (default: 0.5)
    --decay-exp FLOAT        Bandwidth decay exponent (default: 0.5)

  Optimizer:
    --optimizer {de,lbfgsb}  Backend (default: de)
    --de-maxiter INT         DE max generations (default: 100)
    --de-popsize INT         DE population size (default: 10)

  Bounds:
    --bounds JSON            Parameter search bounds override
                             Example: '{"water_ratio":[0.35,0.45]}'
""")
    p.add_argument("--bounds", type=str, help="JSON: parameter bounds override")
    p.add_argument("--weights", type=str, help="JSON: performance attribute weights")
    p.add_argument("--optimizer", choices=["lbfgsb", "de"], default=None)
    p.add_argument("--radius", type=float, default=None, help="Exploration radius")
    p.add_argument("--buffer", type=float, default=None, help="Normalization buffer (default: 0.5)")
    p.add_argument("--decay-exp", type=float, default=None, help="Bandwidth decay exponent (default: 0.5)")
    p.add_argument("--de-maxiter", type=int, default=None)
    p.add_argument("--de-popsize", type=int, default=None)
    p.set_defaults(func=cmd_configure)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline experiments (space-filling)")
    p.add_argument("--n", type=int, default=10, help="Number of experiments")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=cmd_baseline)

    # explore
    p = sub.add_parser("explore", help="Run exploration rounds (incremental)")
    p.add_argument("--n", type=int, default=5, help="Number of rounds to add")
    p.add_argument("--kappa", type=float, default=0.5, help="Exploration weight (0=exploit, 1=explore)")
    p.add_argument("--plot", action="store_true", help="Show per-round plots inline")
    p.add_argument("--validate", action="store_true", help="Validate model during training")
    p.set_defaults(func=cmd_explore)

    # test-set
    p = sub.add_parser("test-set", help="Create held-out test experiments")
    p.add_argument("--n", type=int, default=20, help="Number of test experiments")
    p.set_defaults(func=cmd_test_set)

    # analyse
    p = sub.add_parser("analyse", help="Evaluate model on test set + sensitivity analysis")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=cmd_analyse)

    # inference
    p = sub.add_parser("inference", help="Single-shot first-time-right manufacturing")
    p.add_argument("--design-intent", type=str, default=None,
                   help="JSON: fix parameters for inference. Example: '{\"n_layers\":5}'")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=cmd_inference)

    # ── Advanced commands ──

    # explore-trajectory
    p = sub.add_parser("explore-trajectory", help="Trajectory exploration: per-layer speed optimization")
    p.add_argument("--n", type=int, default=3, help="Number of trajectory rounds")
    p.add_argument("--kappa", type=float, default=0.5, help="Exploration weight")
    p.add_argument("--delta", type=float, default=5.0, help="Trust region half-width for speed (mm/s)")
    p.add_argument("--smoothing", type=float, default=0.25, help="Smoothing penalty (0=off, 0.3=strong)")
    p.add_argument("--lookahead", type=int, default=2, help="MPC lookahead steps")
    p.add_argument("--discount", type=float, default=0.9, help="MPC discount factor")
    p.add_argument("--design-intent", type=str, default=None, help="JSON: fix parameters")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=cmd_explore_trajectory)

    # adapt
    p = sub.add_parser("adapt", help="Online inference with layer-by-layer adaptation")
    p.add_argument("--delta", type=float, default=5.0, help="Trust region half-width for speed (mm/s)")
    p.add_argument("--design-intent", type=str, default=None, help="JSON: fix parameters")
    p.set_defaults(func=cmd_adapt)

    # summary
    p = sub.add_parser("summary", help="Show run summary across all phases")
    p.set_defaults(func=cmd_summary)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
