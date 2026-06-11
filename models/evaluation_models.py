"""Evaluation models for the extrusion printing simulation.

Each model scores a single feature linearly against a target:
``score = 1 − |feature − target| / scale``, clipped to [0, 1] by the framework.
"""

from typing import Any

import torch

from pred_fab import IEvaluationModel
from pred_fab.core import Parameters
from pred_fab.utils import PfabLogger


class _LinearTargetScore(IEvaluationModel):
    """Shared linear target/scale scoring for single-feature models."""

    TARGET: float
    SCALE: float

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    def _score_row(
        self,
        feature_values: dict[str, float],
        params: dict[str, Any],
        **dimensions: int,
    ) -> float:
        val = feature_values[self.input_features[0]]
        return 1.0 - abs(val - self.TARGET) / self.SCALE

    def _score_tensor(
        self,
        feature_tensors: dict[str, torch.Tensor],
        parameters_list: list[Parameters],
    ) -> torch.Tensor:
        t = feature_tensors[self.input_features[0]]
        scores = 1.0 - (t - self.TARGET).abs() / self.SCALE
        return scores.clamp(0.0, 1.0).mean(dim=1)


class PathAccuracy(_LinearTargetScore):
    """Scores path_deviation against a zero-deviation target."""

    MAX_DEVIATION = 0.003  # m — deviation at which score = 0
    TARGET = 0.0
    SCALE = MAX_DEVIATION

    @property
    def input_features(self) -> list[str]:
        return ["path_deviation"]

    @property
    def output_performance(self) -> str:
        return "path_accuracy"


class EnergyEfficiency(_LinearTargetScore):
    """Scores energy_per_segment against the minimum achievable energy.

    TARGET_ENERGY is the minimum achievable (~speed=20). Scores fall off
    linearly as energy rises toward MAX_ENERGY. Lower speed is always better
    for energy, creating genuine tension with path accuracy.
    """

    TARGET_ENERGY = 4.5   # J  — minimum achievable (low speed)
    MAX_ENERGY = 24.0     # J  — maximum of the energy scale
    TARGET = TARGET_ENERGY
    SCALE = MAX_ENERGY

    @property
    def input_features(self) -> list[str]:
        return ["energy_per_segment"]

    @property
    def output_performance(self) -> str:
        return "energy_efficiency"


class ProductionRate(_LinearTargetScore):
    """Scores effective production_rate [mm/s] against the maximum achievable.

    production_rate = print_speed × slip_factor, so MAX_RATE = 60 mm/s
    (no slip, full speed). score = rate / MAX_RATE — higher is better.
    """

    MAX_RATE = 60.0  # mm/s — max achievable (no slip, print_speed=60)
    TARGET = MAX_RATE
    SCALE = MAX_RATE

    @property
    def input_features(self) -> list[str]:
        return ["production_rate"]

    @property
    def output_performance(self) -> str:
        return "production_rate"
