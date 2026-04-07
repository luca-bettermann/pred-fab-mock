"""Workflow helpers for the PFAB mock journey.

Encapsulates the experiment → evaluate → retrain loop and history tracking
so that main.py stays focused on the high-level agent operations.
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pred_fab.core import Dataset, DataModule
from pred_fab.orchestration import PfabAgent

from sensors import FabricationSystem
from utils import params_from_spec, get_performance, quiet_console


ExperimentLog = List[Tuple[str, Dict[str, Any], Dict[str, float]]]


class JourneyState:
    """Tracks experiment history and performance across all phases."""

    def __init__(self) -> None:
        self.all_params: List[Dict[str, Any]] = []
        self.all_phases: List[str] = []
        self.all_codes: List[str] = []
        self.perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        self.prev_params: Dict[str, Any] = {}

    def record(self, phase: str, code: str, params: Dict[str, Any], perf: Dict[str, float]) -> None:
        self.all_params.append(params)
        self.all_phases.append(phase)
        self.all_codes.append(code)
        self.perf_history.append((params, perf))
        self.prev_params = params


def clean_artifacts(plot_dirs: List[str]) -> None:
    """Remove previous run artefacts and create fresh plot directories."""
    for d in ["./pfab_data", "./plots"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for d in plot_dirs:
        os.makedirs(d, exist_ok=True)


def with_dimensions(params: Dict[str, Any], fab: FabricationSystem) -> Dict[str, Any]:
    """Return params with n_layers and n_segments derived from the design choice."""
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}


def run_and_evaluate(
    dataset: Dataset,
    agent: PfabAgent,
    fab: FabricationSystem,
    params: Dict[str, Any],
    exp_code: str,
) -> Any:
    """Create experiment, run fabrication, evaluate features + performance, persist."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data


def run_baseline_phase(
    agent: PfabAgent,
    dataset: Dataset,
    fab: FabricationSystem,
    n_baseline: int,
    state: JourneyState,
) -> Tuple[ExperimentLog, List[Any], List[Any]]:
    """Run baseline experiments and return (log, experiment_data_list, specs)."""
    with quiet_console(agent.logger):
        specs = agent.baseline_step(n=n_baseline)

    log: ExperimentLog = []
    exps = []
    for i, spec in enumerate(specs):
        params = with_dimensions(params_from_spec(spec), fab)
        exp_code = f"baseline_{i+1:02d}"
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        exps.append(exp_data)
        log.append((exp_code, params, perf))
        state.record("baseline", exp_code, params_from_spec(spec), perf)

    state.prev_params = with_dimensions(params_from_spec(specs[-1]), fab)
    return log, exps, specs


def run_exploration_round(
    agent: PfabAgent,
    dataset: Dataset,
    fab: FabricationSystem,
    datamodule: DataModule,
    state: JourneyState,
    round_idx: int,
    w_explore: float,
    n_optimization_rounds: int,
) -> Tuple[str, Dict[str, Any], Dict[str, float], float, Dict[str, Any]]:
    """Run one exploration round. Returns (code, params, perf, uncertainty, proposed_raw)."""
    with quiet_console(agent.logger):
        spec = agent.exploration_step(datamodule, w_explore=w_explore, n_optimization_rounds=n_optimization_rounds)

    proposed = params_from_spec(spec)
    proposed_full = {**state.prev_params, **proposed, "n_layers": 5, "n_segments": 4}
    params = with_dimensions({**state.prev_params, **proposed}, fab)
    exp_code = f"explore_{round_idx+1:02d}"

    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)

    with quiet_console(agent.logger):
        datamodule.update()
        agent.train(datamodule, validate=False)

    perf = get_performance(exp_data)
    state.record("exploration", exp_code, params, perf)
    u = agent.predict_uncertainty(proposed_full, datamodule)
    return exp_code, params, perf, u, proposed


def run_inference_round(
    agent: PfabAgent,
    dataset: Dataset,
    fab: FabricationSystem,
    datamodule: DataModule,
    state: JourneyState,
    round_idx: int,
    n_optimization_rounds: int,
) -> Tuple[str, Dict[str, Any], Dict[str, float], Any]:
    """Run one inference round. Returns (code, params, perf, exp_data)."""
    params = state.prev_params
    exp_code = f"infer_{round_idx+1:02d}"
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)

    with quiet_console(agent.logger):
        spec = agent.inference_step(exp_data, datamodule, w_explore=0.0,
                                     n_optimization_rounds=n_optimization_rounds, current_params=params)

    next_params = with_dimensions({**params, **params_from_spec(spec)}, fab)

    with quiet_console(agent.logger):
        dataset.save_experiment(exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)

    perf = get_performance(exp_data)
    state.record("inference", exp_code, params, perf)
    state.prev_params = next_params
    return exp_code, params, perf, exp_data


def run_adaptation_phase(
    agent: PfabAgent,
    dataset: Dataset,
    fab: FabricationSystem,
    state: JourneyState,
    start_speed: float = 40.0,
    n_layers: int = 5,
) -> Tuple[List[float], List[float], List[float]]:
    """Run layer-by-layer adaptation. Returns (speeds, deviations, counterfactual_devs)."""
    adapt_params = with_dimensions({**state.prev_params, "print_speed": start_speed}, fab)
    adapt_exp = dataset.create_experiment("adapt_01", parameters=adapt_params)
    agent.set_active_experiment(adapt_exp)

    speeds: List[float] = []
    deviations: List[float] = []

    for layer_idx in range(n_layers):
        fab.run_layer(adapt_params, layer_idx)
        start, end = adapt_exp.parameters.get_start_and_end_indices("n_layers", layer_idx)
        agent.feature_system.run_feature_extraction(  # type: ignore[union-attr]
            adapt_exp, evaluate_from=start, evaluate_to=end
        )

        speed = float(adapt_params["print_speed"])
        speeds.append(speed)

        feat = adapt_exp.features.get_value("path_deviation")
        dev = float(np.mean(feat[layer_idx, :]))
        deviations.append(dev)

        if layer_idx < n_layers - 1:
            with quiet_console(agent.logger):
                spec = agent.adaptation_step(
                    dimension="n_layers", step_index=layer_idx,
                    exp_data=adapt_exp, record=True,
                )
            adapt_params["print_speed"] = float(spec.initial_params.get("print_speed", speed))

    with quiet_console(agent.logger):
        dataset.save_experiment("adapt_01")

    # Counterfactual: constant speed
    from sensors.physics import path_deviation as _phys_dev
    design = str(adapt_params.get("design", "A"))
    water = float(adapt_params.get("water_ratio", 0.36))
    material = str(adapt_params.get("material", "concrete"))
    n_segs = int(adapt_params["n_segments"])
    counterfactual = [
        float(sum(
            _phys_dev(start_speed, design, s, water, material, li) for s in range(n_segs)
        ) / n_segs)
        for li in range(n_layers)
    ]

    return speeds, deviations, counterfactual


def get_physics_optimum(design_intent: Dict[str, Any]) -> Tuple[float, float]:
    """Return (optimal_speed, optimal_water) for the given design intent."""
    from sensors.physics import DELTA, THETA, DESIGN_COMPLEXITY, MAT_SAG, W_OPTIMAL
    complexity = DESIGN_COMPLEXITY[str(design_intent["design"])]
    sag = MAT_SAG[str(design_intent["material"])]
    w_opt = W_OPTIMAL[str(design_intent["material"])]
    spd_opt = float(np.clip(np.sqrt(THETA * sag / (DELTA * complexity)), 20.0, 60.0))
    return spd_opt, w_opt
