"""Shared infrastructure for workflow steps: session, rebuild, helpers."""

import json
import os
import sys
from typing import Any

import numpy as np

from pred_fab.orchestration import Optimizer
from pred_fab.core import Dataset
from pred_fab import combined_score
from pred_fab.plotting import AxisSpec

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


SESSION_FILE = ".pfab_session.json"
PLOT_DIR = "./plots"


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def next_code(state: JourneyState, prefix: str) -> str:
    existing = [c for c in state.all_codes if c.startswith(prefix + "_")]
    return f"{prefix}_{len(existing) + 1:02d}"


def save_session(config: dict[str, Any], journey: JourneyState) -> None:
    data = {
        "config": config,
        "journey": {
            "all_params": journey.all_params,
            "all_phases": journey.all_phases,
            "all_codes": journey.all_codes,
            "perf_history": [(p, pf) for p, pf in journey.perf_history],
            "prev_params": journey.prev_params,
            "schedules": journey.schedules,
        },
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(to_native(data), f, indent=2)


def load_session() -> tuple[dict[str, Any], JourneyState]:
    if not os.path.exists(SESSION_FILE):
        print("No session found. Run 'uv run main.py init-schema' first.")
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
    state.schedules = j.get("schedules", {})
    return config, state


def rebuild(config: dict[str, Any], verbose: bool = False) -> tuple[Any, Dataset, FabricationSystem]:
    """Reconstruct agent + dataset + fab from session config."""
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
        if config.get("de_tol") is not None:
            opt_kwargs["de_tol"] = config["de_tol"]
        if opt_kwargs:
            agent.configure_optimizer(**opt_kwargs)

        if config.get("schedule_smoothing") is not None:
            agent.calibration_system.schedule_smoothing = config["schedule_smoothing"]

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


def ensure_plot_dir() -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR


def get_physics_optimum(perf_weights=None, n_layers=N_LAYERS):
    """Find the physics optimum location."""
    from visualization.helpers import evaluate_physics_grid
    _, _, phys_metrics = evaluate_physics_grid(50, perf_weights, n_layers=n_layers)
    combined = list(phys_metrics.values())[-1]
    opt_idx = np.unravel_index(np.argmax(combined), combined.shape)
    phys_waters = np.linspace(0.30, 0.50, 50)
    phys_speeds = np.linspace(20.0, 60.0, 50)
    return (phys_waters[opt_idx[1]], phys_speeds[opt_idx[0]])


def compute_acquisition_grid(agent, dm, kappa, res=30):
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

    combined_grid = (1 - kappa) * p_norm + kappa * unc_grid
    return waters, speeds, p_norm, unc_grid, combined_grid


# Schema-specific axis definitions used across all steps
X_AXIS = AxisSpec("water_ratio", "Water Ratio", bounds=(0.30, 0.50))
Y_AXIS = AxisSpec("print_speed", "Print Speed", unit="mm/s", bounds=(20.0, 60.0))
Z_AXIS = AxisSpec("n_layers", "Layers")
FIXED_DIMS = {"n_layers": N_LAYERS, "n_segments": N_SEGMENTS}


def print_config_set(label: str, old: Any, new: Any) -> None:
    """Print a configuration change line showing old → new."""
    _G = "\033[32m"
    _D = "\033[2m"
    _R = "\033[0m"
    if isinstance(new, dict):
        print(f"\n  {_D}{label}{_R}")
        old_d = old if isinstance(old, dict) else {}
        for k, v in new.items():
            prev = old_d.get(k)
            if prev is not None and prev != v:
                print(f"  {_G}\u2713{_R} {k} = {prev} \u2192 {v}")
            else:
                print(f"  {_G}\u2713{_R} {k} = {v}")
    else:
        if old is not None and old != new:
            print(f"  {_G}\u2713{_R} {label} = {old} \u2192 {new}")
        else:
            print(f"  {_G}\u2713{_R} {label} = {new}")


def print_config_show(config: dict[str, Any]) -> None:
    """Print all current configuration values, including defaults and schema bounds."""
    _D = "\033[2m"
    _R = "\033[0m"
    _B = "\033[1m"

    # Bounds from schema (stored at init) + user overrides
    all_bounds: dict[str, tuple[float, float]] = {}
    for k, v in (config.get("schema_bounds", {}) or {}).items():
        all_bounds[k] = (v[0], v[1])
    user_bounds = config.get("bounds", {}) or {}
    for k, v in user_bounds.items():
        all_bounds[k] = (v[0], v[1])

    # Config groups
    groups: list[tuple[str, list[tuple[str, str, Any]]]] = [
        ("Performance", [
            ("performance_weights", "Weights", {"path_accuracy": 1, "energy_efficiency": 1, "production_rate": 1}),
        ]),
        ("Exploration", [
            ("exploration_radius", "Radius", 0.20),
            ("buffer", "Buffer", 0.5),
            ("decay_exp", "Decay exponent", 0.5),
        ]),
        ("Optimizer", [
            ("optimizer", "Backend", "de"),
            ("de_maxiter", "DE max iterations", 1000),
            ("de_popsize", "DE population size", 15),
            ("de_tol", "DE tolerance", 0.001),
        ]),
        ("Schedule", [
            ("schedule_smoothing", "Smoothing", 0.05),
            ("schedule_delta", "Default delta", None),
        ]),
    ]

    print(f"\n  {_B}Current Configuration{_R}")
    for group_name, keys in groups:
        print(f"\n  {_D}{group_name}{_R}")
        for config_key, label, default in keys:
            val = config.get(config_key, default)
            if val is None:
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"    {k:<20s} = {v}")
            else:
                is_default = config_key not in config or config[config_key] is None
                suffix = f" {_D}(default){_R}" if is_default else ""
                print(f"    {label:<20s} = {val}{suffix}")

    # Bounds section — always show from schema + user overrides
    print(f"\n  {_D}Bounds{_R}")
    for code, (lo, hi) in sorted(all_bounds.items()):
        is_overridden = code in user_bounds
        suffix = "" if is_overridden else f" {_D}(schema){_R}"
        print(f"    {code:<20s} = [{lo}, {hi}]{suffix}")

    print()


def apply_schedule_args(agent: Any, args: Any) -> None:
    """Parse --schedule PARAM:DIM[:DELTA] flags and configure the agent."""
    schedules = getattr(args, "schedule", None)
    if not schedules:
        return
    smoothing = getattr(args, "smoothing", None)
    for spec in schedules:
        parts = spec.split(":")
        if len(parts) < 2:
            agent.logger.console_warning(
                f"Ignoring malformed --schedule '{spec}' (expected PARAM:DIM[:DELTA])"
            )
            continue
        param = parts[0].strip()
        dim = parts[1].strip()
        delta = float(parts[2]) if len(parts) >= 3 else None
        agent.configure_schedule(param, dim, delta=delta, smoothing=smoothing)


def extract_schedule_steps(spec: Any, base_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-step param dicts from an ExperimentSpec's schedules."""
    if not spec.schedules:
        return [base_params]
    # Determine L from the first schedule's entries
    first_sched = next(iter(spec.schedules.values()))
    L = max(idx for idx, _ in first_sched.entries) + 1 if first_sched.entries else 1
    steps: list[dict[str, Any]] = []
    for step_i in range(L):
        step_params = dict(base_params)
        for _dim, sched in spec.schedules.items():
            for idx, proposal in sched.entries:
                if idx == step_i:
                    step_params.update(proposal.to_dict())
        steps.append(step_params)
    return steps
