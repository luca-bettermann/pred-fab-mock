"""FabricationTime ← printing_duration.

Cost attribute (smaller is better). Target is the fastest achievable
duration; scaling covers the observed range.
"""
from __future__ import annotations

from typing import Any, Mapping

from pred_fab.utils import PfabLogger

from models.evaluations.base import TargetEvaluationModel
from models.schema import AttributeCode, FeatureCode

DURATION_MIN_KEY = "duration_min_s"
DURATION_MAX_KEY = "duration_max_s"


class FabricationTimeEval(TargetEvaluationModel):
    """Experiment-level fabrication-time score — sums per-layer printing
    duration across the whole print, then scores the total once. Lower
    total = higher score.

    ``aggregate_input="sum"`` (pred-fab API) reduces the per-layer feature
    array along the layer dim before ``_score_row`` / ``_score_tensor`` fire.
    ``duration_min_s`` / ``duration_max_s`` are therefore total-duration
    bounds, derived from observed totals across the dataset.
    """

    def __init__(
        self, logger: PfabLogger, *,
        duration_min_s: float,
        duration_max_s: float,
    ) -> None:
        self._min = duration_min_s
        self._max = duration_max_s
        super().__init__(logger)

    @property
    def aggregate_input(self) -> str:
        return "sum"

    @classmethod
    def from_study_constants(
        cls, logger: PfabLogger, constants: Mapping[str, object],
    ) -> "FabricationTimeEval":
        return cls(
            logger,
            duration_min_s=float(constants[DURATION_MIN_KEY]),
            duration_max_s=float(constants[DURATION_MAX_KEY]),
        )

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_feature(self) -> str:
        return FeatureCode.PRINTING_DURATION

    @property
    def output_performance(self) -> str:
        return AttributeCode.FABRICATION_TIME

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return self._min

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return self._max - self._min
