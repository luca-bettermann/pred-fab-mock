"""Shared helpers across CLI steps for the ADVEI 2026 mock journey."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

from pred_fab.core import Dataset
from pred_fab.plotting import AxisSpec
from pred_fab.utils.metrics import combined_score

from schema import build_schema
from agent_setup import build_agent
from sensors import FabricationSystem
from utils import params_from_spec, get_performance
from workflow import JourneyState, with_dimensions, run_and_evaluate
from cli_helpers import show_plot, show_plot_with_header, ensure_plot_dir


SESSION_FILE = ".pfab_session.json"
PLOT_DIR = "./plots"

# ADVEI schema axis definitions for plotting
SPEED_AXIS = AxisSpec("print_speed", "Print Speed", unit="m/s", bounds=(0.004, 0.008))
CALIB_AXIS = AxisSpec("calibration_factor", "Calibration Factor", bounds=(1.6, 2.2))
OFFSET_AXIS = AxisSpec("path_offset", "Path Offset", unit="mm", bounds=(0.0, 3.0))
HEIGHT_AXIS = AxisSpec("layer_height", "Layer Height", unit="mm", bounds=(2.0, 3.0))
SLOWDOWN_AXIS = AxisSpec("slowdown_factor", "Slowdown Factor", bounds=(0.0, 1.0))

# Default fixed params for 2D topology slices
DEFAULT_FIXED = {
    "path_offset": 1.5,
    "layer_height": 2.5,
    "calibration_factor": 1.9,
    "print_speed": 0.006,
    "slowdown_factor": 0.3,
}

# Default equal weights when user hasn't configured any
DEFAULT_WEIGHTS: dict[str, float] = {
    "structural_integrity": 1.0,
    "material_deposition": 1.0,
    "extrusion_stability": 1.0,
    "energy_footprint": 1.0,
    "fabrication_time": 1.0,
}


def effective_weights(config: dict[str, Any]) -> dict[str, float]:
    """Return performance weights from config, or equal weights if unconfigured."""
    return config.get("performance_weights") or DEFAULT_WEIGHTS


# === Session persistence ====================================================

def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
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
    state.trajectories = j.get("trajectories", {})
    return config, state


# === Agent / fab reconstruction =============================================

def rebuild(config: dict[str, Any], verbose: bool = False) -> tuple[Any, Dataset, FabricationSystem]:
    """Reconstruct agent + dataset + fab from session config."""
    if not verbose:
        _real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    try:
        schema = build_schema()
        fab = FabricationSystem()
        agent = build_agent(schema, fab=fab, verbose=verbose)

        if config.get("performance_weights"):
            agent.configure_performance(weights=config["performance_weights"])

        explore_kwargs: dict[str, Any] = {}
        if config.get("exploration_radius") is not None:
            explore_kwargs["radius"] = config["exploration_radius"]
        if config.get("sigma") is not None:
            explore_kwargs["sigma"] = config["sigma"]
        if config.get("mc_exponent_offset") is not None:
            explore_kwargs["mc_exponent_offset"] = config["mc_exponent_offset"]
        if explore_kwargs:
            agent.configure_exploration(**explore_kwargs)

        opt_kwargs: dict[str, Any] = {}
        if config.get("de_maxiter") is not None:
            opt_kwargs["de_maxiter"] = config["de_maxiter"]
        if config.get("de_popsize") is not None:
            opt_kwargs["de_popsize"] = config["de_popsize"]
        if opt_kwargs:
            agent.configure_optimizer(**opt_kwargs)

        if config.get("bounds"):
            bounds = {k: tuple(v) for k, v in config["bounds"].items()}
            agent.calibration_system.configure_param_bounds(bounds)

        trust_regions = resolve_trust_regions(agent, config)
        if trust_regions:
            agent.calibration_system.configure_adaptation_delta(trust_regions, force=True)

        dataset = Dataset(schema=schema)
        dataset.populate()
    finally:
        if not verbose:
            sys.stdout.close()  # type: ignore[union-attr]
            sys.stdout = _real_stdout

    return agent, dataset, fab




# === Trust-region helpers (per-trajectory delta defaults) ===================

def default_trust_regions(agent: Any) -> dict[str, float]:
    """Default trust region per runtime parameter: ``(max - min) / 10``."""
    regions: dict[str, float] = {}
    for code, p in agent.schema.parameters.data_objects.items():
        if not getattr(p, "runtime_adjustable", False):
            continue
        lo = getattr(p, "min_val", None)
        hi = getattr(p, "max_val", None)
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            continue
        regions[code] = (float(hi) - float(lo)) / 10.0
    return regions


def resolve_trust_regions(agent: Any, config: dict[str, Any]) -> dict[str, float]:
    regions = default_trust_regions(agent)
    user = config.get("trust_regions") or {}
    for k, v in user.items():
        regions[k] = float(v)
    return regions


def apply_schedule_args(agent: Any, args: Any, config: dict[str, Any]) -> None:
    """Parse --schedule PARAM:DIM flags and configure the agent."""
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


# === Schedule extraction + run wrapper ======================================

def extract_schedule_steps(spec: Any, base_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-step param dicts from an ExperimentSpec's trajectories."""
    if not spec.trajectories:
        return [base_params]
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
    """Run an experiment from a spec, apply schedules, persist parameter_updates.

    ``run_and_evaluate`` saves the experiment before ``apply_schedules`` can
    populate ``parameter_updates``, so a second save_experiment call follows
    when a trajectory is present (so reload-from-disk includes the updates).
    """
    proposed = params_from_spec(spec)
    merged = dict(extra_params) if extra_params else {}
    merged.update(proposed)
    params = with_dimensions(merged)
    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code, dataset_code=dataset_code)
    if spec.trajectories:
        spec.apply_schedules(exp_data)
        dataset.save_experiment(exp_code)
    sched_data = extract_schedule_steps(spec, params) if spec.trajectories else None
    return exp_data, params, sched_data
