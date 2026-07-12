"""Shared infrastructure for workflow steps: session, rebuild, helpers."""

import json
import os
import sys
from typing import Any

import numpy as np

from pred_fab.core import Dataset
from pred_fab.utils.metrics import combined_score
from pred_fab.plotting import AxisSpec

from schema import (
    build_schema, PLOT_DIR, WATER_RATIO_BOUNDS, PRINT_SPEED_BOUNDS, DEFAULT_PERF_WEIGHTS,
)
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

# ANSI styles shared by all step console output
_B = "\033[1m"   # bold
_C = "\033[36m"  # cyan
_D = "\033[2m"   # dim
_G = "\033[32m"  # green
_R = "\033[0m"   # reset


def print_phase_banner(phase: str, title: str, subtitle: str = "") -> None:
    """Print the boxed phase banner used by the setup steps."""
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE {phase}{_R}{_B} ▸ {title}{_R}")
    if subtitle:
        print(f"  {_D}{subtitle}{_R}")
    print(f"{_B}{_C}{bar}{_R}")


def show_plot_with_header(path: str, title: str, *, inline: bool = True) -> None:
    """Print a dim-styled title line, then display the plot."""
    print(f"\n  {_D}{title}{_R}")
    show_plot(path, inline=inline)


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
            "trajectories": journey.trajectories,
        },
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(to_native(data), f, indent=2)


def load_session() -> tuple[dict[str, Any], JourneyState]:
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
    state.trajectories = j["trajectories"]
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
        agent = build_agent(schema, fab.camera, fab.energy, verbose=verbose)

        if config.get("performance_weights"):
            agent.configure_performance(weights=config["performance_weights"])

        explore_kwargs: dict[str, Any] = {}
        if config.get("sigma") is not None:
            explore_kwargs["sigma"] = config["sigma"]
        if config.get("kappa") is not None:
            explore_kwargs["kappa"] = config["kappa"]
        if explore_kwargs:
            agent.configure_exploration(**explore_kwargs)

        opt_kwargs: dict[str, Any] = {}
        for key in ("n_starts", "n_sobol", "lr"):
            if config.get(key) is not None:
                opt_kwargs[key] = config[key]
        if opt_kwargs:
            agent.configure_optimizer(**opt_kwargs)

        if config.get("bounds"):
            bounds = {k: tuple(v) for k, v in config["bounds"].items()}
            agent.calibration_system.configure_param_bounds(bounds)

        # Trust regions are the default delta for both scheduled steps and
        # online adaptation. Default = bounds_span / 10 per runtime param;
        # configure --trust-regions JSON overrides per-key.
        trust_regions = resolve_trust_regions(agent, config)
        if trust_regions:
            agent.calibration_system.configure_adaptation_delta(trust_regions, force=True)

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


def compute_acquisition_grid(agent, kappa, res=30):
    """Slice the agent's own acquisition into 2D grids over (water, speed).

    Returns (waters, speeds, perf_grid, evidence_gain_grid, acq_grid) —
    the same evidence_gain() / system_performance() path the optimizer uses.
    """
    xs, ys, evidence_grid, perf_grid, acq_grid = agent.calibration_system.compute_acquisition_grids(
        X_AXIS.key, Y_AXIS.key,
        X_AXIS.bounds, Y_AXIS.bounds,
        fixed_params=dict(FIXED_DIMS),
        kappa=kappa,
        resolution=res,
    )
    return xs, ys, perf_grid, evidence_grid, acq_grid


# Schema-specific axis definitions used across all steps
X_AXIS = AxisSpec("water_ratio", "Water Ratio", bounds=WATER_RATIO_BOUNDS)
Y_AXIS = AxisSpec("print_speed", "Print Speed", unit="mm/s", bounds=PRINT_SPEED_BOUNDS)
Z_AXIS = AxisSpec("n_layers", "Layers", integer=True)
LAYER_AXIS = AxisSpec("n_layers", "Layers", integer=True, bounds=(4.0, 8.0))
# n_segments is currently fixed at 4 in the schema; the padded bounds give the
# plot a visible y-extent so domain points don't degenerate to a single line.
SEGMENT_AXIS = AxisSpec("n_segments", "Segments", integer=True, bounds=(3.0, 5.0))
FIXED_DIMS = {"n_layers": N_LAYERS, "n_segments": N_SEGMENTS}


def predict_score_grid(
    agent: Any,
    perf_weights: dict[str, float],
    n_layers: int = N_LAYERS,
    resolution: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted combined-score grid over the schema's (water_ratio, print_speed) bounds.

    Returns (waters, speeds, grid) with grid[j, i] = score at (waters[i], speeds[j]).
    Failed predictions are recorded as NaN and reported in one aggregate warning.
    """
    waters = np.linspace(WATER_RATIO_BOUNDS[0], WATER_RATIO_BOUNDS[1], resolution)
    speeds = np.linspace(PRINT_SPEED_BOUNDS[0], PRINT_SPEED_BOUNDS[1], resolution)
    grid = np.full((resolution, resolution), np.nan)
    n_failed = 0
    first_exc: Exception | None = None
    for i, w in enumerate(waters):
        for j, spd in enumerate(speeds):
            try:
                perf = agent.predict_performance({"water_ratio": float(w), "print_speed": float(spd),
                                                  "n_layers": n_layers, "n_segments": N_SEGMENTS})
                grid[j, i] = combined_score(perf, perf_weights)
            except Exception as exc:
                n_failed += 1
                if first_exc is None:
                    first_exc = exc
    if n_failed:
        print(f"  Warning: {n_failed}/{resolution * resolution} grid predictions failed "
              f"(first: {first_exc})")
    return waters, speeds, grid


def print_config_set(label: str, old: Any, new: Any) -> None:
    """Print a configuration change line showing old → new."""
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
            ("performance_weights", "Weights", DEFAULT_PERF_WEIGHTS),
        ]),
        ("Exploration", [
            ("sigma", "Sigma", None),
            ("kappa", "Kappa default", None),
        ]),
        ("Optimizer", [
            ("n_starts", "Multi-start count", None),
            ("n_sobol", "Sobol candidates", None),
            ("lr", "Learning rate", None),
        ]),
        ("Trajectory", [
            ("default_schedule", "Default schedule", None),
            ("trust_regions", "Trust regions", None),
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


def default_trust_regions(agent: Any) -> dict[str, float]:
    """Default trust region per runtime parameter: 1/10 of the bounds span.

    Iterates the schema's runtime-flagged parameters and computes
    `(max − min) / 10` from each parameter's constraints. Non-runtime
    parameters are skipped (they have no concept of mid-run adjustability),
    as are params with infinite/missing bounds.
    """
    regions: dict[str, float] = {}
    for code, p in agent.schema.parameters.data_objects.items():
        if not p.runtime_adjustable:
            continue
        lo = p.constraints.get("min")
        hi = p.constraints.get("max")
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            continue
        regions[code] = (float(hi) - float(lo)) / 10.0
    return regions


def resolve_trust_regions(agent: Any, config: dict[str, Any]) -> dict[str, float]:
    """Merge default trust regions with any user override stored in `config`."""
    regions = default_trust_regions(agent)
    user = config.get("trust_regions") or {}
    for k, v in user.items():
        regions[k] = float(v)
    return regions


def apply_schedule_args(agent: Any, args: Any, config: dict[str, Any]) -> None:
    """Parse --schedule PARAM:DIM flags (or fall back to the configured default)
    and configure the agent. Per-step delta is the parameter's trust region."""
    per_call = getattr(args, "schedule", None)
    schedules = per_call if per_call else (config.get("default_schedule") or [])
    if not schedules:
        return
    trust_regions = resolve_trust_regions(agent, config)
    for spec in schedules:
        parts = spec.split(":")
        if len(parts) < 2:
            agent.logger.console_warning(
                f"Ignoring malformed --schedule '{spec}' (expected PARAM:DIM)"
            )
            continue
        param = parts[0].strip()
        dim = parts[1].strip()
        delta = trust_regions.get(param)
        agent.configure_trajectory(param, dim, delta=delta)


def extract_schedule_steps(spec: Any, base_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-step param dicts from an ExperimentSpec's schedules."""
    if not spec.trajectories:
        return [base_params]
    # Determine L from the first schedule's entries
    first_sched = next(iter(spec.trajectories.values()))
    L = max(idx for idx, _ in first_sched.entries) + 1 if first_sched.entries else 1
    steps: list[dict[str, Any]] = []
    for step_i in range(L):
        step_params = dict(base_params)
        for _dim, sched in spec.trajectories.items():
            for idx, proposal in sched.entries:
                if idx == step_i:
                    step_params.update(proposal.to_dict())
        steps.append(step_params)
    return steps


def run_and_record(
    dataset: Dataset,
    agent: Any,
    fab: FabricationSystem,
    spec: Any,
    exp_code: str,
    extra_params: dict[str, Any] | None = None,
    dataset_code: str | None = None,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]] | None]:
    """Run an experiment from a spec, apply trajectories, persist parameter_updates to disk.

    run_and_evaluate() saves the experiment before apply_trajectories can populate
    parameter_updates, so a second save_experiment is required after apply_trajectories to
    persist the trajectory across sessions. Without this, reload-from-disk strips the
    parameter_updates and the KDE never sees trajectory segments. Returns (exp_data,
    params, sched_data).
    """
    proposed = params_from_spec(spec)
    merged = dict(extra_params) if extra_params else {}
    merged.update(proposed)
    params = with_dimensions(merged)
    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code, dataset_code=dataset_code)
    if spec.trajectories:
        spec.apply_trajectories(exp_data)
        # run_and_evaluate saved pre-apply state; persist parameter_updates now.
        dataset.save_experiment(exp_code)
    sched_data = extract_schedule_steps(spec, params) if spec.trajectories else None
    return exp_data, params, sched_data
