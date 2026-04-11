"""Workflow helpers for the PFAB mock journey.

Encapsulates the experiment → evaluate → retrain loop and history tracking
so that main.py stays focused on the high-level agent operations.
"""

import os
import shutil
from typing import Any

import numpy as np

from pred_fab.core import Dataset
from pred_fab.orchestration import PfabAgent

from sensors import FabricationSystem


ExperimentLog = list[tuple[str, dict[str, Any], dict[str, float]]]


class JourneyState:
    """Tracks experiment history and performance across all phases."""

    def __init__(self) -> None:
        self.all_params: list[dict[str, Any]] = []
        self.all_phases: list[str] = []
        self.all_codes: list[str] = []
        self.perf_history: list[tuple[dict[str, Any], dict[str, float]]] = []
        self.prev_params: dict[str, Any] = {}

    def record(self, phase: str, code: str, params: dict[str, Any], perf: dict[str, float]) -> None:
        self.all_params.append(params)
        self.all_phases.append(phase)
        self.all_codes.append(code)
        self.perf_history.append((params, perf))
        self.prev_params = params


def clean_artifacts(plot_dirs: list[str]) -> None:
    """Remove previous run artefacts and create fresh plot directories."""
    for d in ["./local", "./plots"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for d in plot_dirs:
        os.makedirs(d, exist_ok=True)


def with_dimensions(params: dict[str, Any], fab: FabricationSystem) -> dict[str, Any]:
    """Return params with n_layers and n_segments derived from the design choice."""
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}


def run_and_evaluate(
    dataset: Dataset,
    agent: PfabAgent,
    fab: FabricationSystem,
    params: dict[str, Any],
    exp_code: str,
) -> Any:
    """Create experiment, run fabrication, evaluate features + performance, persist."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data


def get_physics_optimum(design_intent: dict[str, Any]) -> tuple[float, float]:
    """Return (optimal_speed, optimal_water) for the given design intent."""
    from sensors.physics import DELTA, THETA, DESIGN_COMPLEXITY, MAT_SAG, W_OPTIMAL
    complexity = DESIGN_COMPLEXITY[str(design_intent["design"])]
    sag = MAT_SAG[str(design_intent["material"])]
    w_opt = W_OPTIMAL[str(design_intent["material"])]
    spd_opt = float(np.clip(np.sqrt(THETA * sag / (DELTA * complexity)), 20.0, 60.0))
    return spd_opt, w_opt
