"""Workflow helpers for the ADVEI 2026 mock journey.

Encapsulates the experiment → evaluate → retrain loop and history tracking
so each ``steps/<phase>.py`` script stays focused on its high-level concern.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from pred_fab.core import Dataset
from pred_fab.orchestration import PfabAgent

from sensors import FabricationSystem
from sensors.physics import COMPONENT_HEIGHT_MM, n_layers_for_height


ExperimentLog = list[tuple[str, dict[str, Any], dict[str, float]]]


# Canonical curved-wall sample-point count.
N_NODES = 7


class JourneyState:
    """Tracks experiment history and performance across all phases."""

    def __init__(self) -> None:
        self.all_params: list[dict[str, Any]] = []
        self.all_phases: list[str] = []
        self.all_codes: list[str] = []
        self.perf_history: list[tuple[dict[str, Any], dict[str, float]]] = []
        self.prev_params: dict[str, Any] = {}
        # Per-experiment trajectory data: code → list of per-step param dicts (or None)
        self.trajectories: dict[str, list[dict[str, Any]]] = {}

    def record(
        self,
        phase: str,
        code: str,
        params: dict[str, Any],
        perf: dict[str, float],
        trajectory: list[dict[str, Any]] | None = None,
    ) -> None:
        self.all_params.append(params)
        self.all_phases.append(phase)
        self.all_codes.append(code)
        self.perf_history.append((params, perf))
        self.prev_params = params
        if trajectory is not None:
            self.trajectories[code] = trajectory


def clean_artifacts(plot_dirs: list[str]) -> None:
    """Remove previous run artefacts and create fresh plot directories."""
    for d in ["./local", "./plots"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for d in plot_dirs:
        os.makedirs(d, exist_ok=True)


def with_dimensions(params: dict[str, Any]) -> dict[str, Any]:
    """Return params with fixed tensor dimensions.

    ``n_layers`` is always MAX_N_LAYERS (15) — the fixed tensor dimension.
    All layers are simulated (no padding); ``layer_height`` affects the
    per-layer physics but doesn't truncate the sequence.
    """
    from sensors.physics import MAX_N_LAYERS
    result = {**params}
    result["n_layers"] = MAX_N_LAYERS
    if "n_nodes" not in result:
        result["n_nodes"] = N_NODES
    return result


def run_and_evaluate(
    dataset: Dataset,
    agent: PfabAgent,
    fab: FabricationSystem,
    params: dict[str, Any],
    exp_code: str,
    dataset_code: str | None = None,
) -> Any:
    """Create experiment, run fab, evaluate features + performance, persist.

    ``dataset_code`` tags the experiment as belonging to a named phase
    (``baseline`` / ``exploration`` / ``inference`` / ``test`` / ``grid``).
    """
    exp_data = dataset.create_experiment(exp_code, parameters=params, dataset_code=dataset_code)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data
