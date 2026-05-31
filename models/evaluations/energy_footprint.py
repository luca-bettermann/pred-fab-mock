"""EnergyFootprint ← robot_energy.

Per-layer robot energy consumption. Lower energy = higher score.
"""
from __future__ import annotations

from typing import Any

from pred_fab.utils import PfabLogger

from models.evaluations.base import TargetEvaluationModel
from models.schema import AttributeCode, FeatureCode


class EnergyFootprintEval(TargetEvaluationModel):
    """Experiment-level energy footprint — sums per-layer robot_energy across
    the whole print, then scores the total once. Lower total = higher score.

    ``aggregate_input="sum"`` (pred-fab API) reduces the per-layer feature
    array along the layer dim before ``_score_row`` / ``_score_tensor`` fire.
    Bounds (``target_energy`` / ``max_energy``) are therefore total-energy
    bounds, derived from observed totals across the dataset.
    """

    def __init__(
        self, logger: PfabLogger, *,
        target_energy: float = 50.0,
        max_energy: float = 200.0,
    ) -> None:
        self._target = target_energy
        self._max = max_energy
        super().__init__(logger)

    @property
    def aggregate_input(self) -> str:
        return "sum"

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_feature(self) -> str:
        return FeatureCode.ROBOT_ENERGY

    @property
    def output_performance(self) -> str:
        return AttributeCode.ENERGY_FOOTPRINT

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return self._target

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return self._max - self._target
