"""Evaluation models for the extrusion printing simulation."""

from typing import Any

from pred_fab import IEvaluationModel
from pred_fab.utils import PfabLogger


class PathAccuracy(IEvaluationModel):
    """Scores path_deviation against a zero-deviation target."""

    MAX_DEVIATION = 0.003  # m — deviation at which score = 0
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "path_deviation"

    @property
    def output_performance(self) -> str:
        return "path_accuracy"

    def _compute_target_value(self, params: dict, **dimensions: Any) -> float:
        return 0.0

    def _compute_scaling_factor(self, params: dict, **dimensions: Any) -> float | None:
        return self.MAX_DEVIATION


class EnergyEfficiency(IEvaluationModel):
    """Scores energy_per_segment against a target energy consumption.

    TARGET_ENERGY is the minimum achievable (~speed=20, clay, A).
    Scores fall off linearly as energy rises toward MAX_ENERGY (speed=60, concrete, B).
    Lower speed is always better for energy, creating genuine tension with path accuracy.
    """

    TARGET_ENERGY = 4.5   # J  — minimum achievable (low speed, clay, simple design)
    MAX_ENERGY = 24.0     # J  — max physically reachable (~speed=60, concrete, B)
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "energy_per_segment"

    @property
    def output_performance(self) -> str:
        return "energy_efficiency"

    def _compute_target_value(self, params: dict, **dimensions: Any) -> float:
        return self.TARGET_ENERGY

    def _compute_scaling_factor(self, params: dict, **dimensions: Any) -> float | None:
        return self.MAX_ENERGY


class ProductionRate(IEvaluationModel):
    """Scores effective production_rate [mm/s] against maximum achievable rate.

    production_rate = print_speed × slip_factor, so MAX_RATE = 60 mm/s (no slip, full speed).
    score = rate / MAX_RATE — higher is better.
    """

    MAX_RATE = 60.0  # mm/s — max achievable (no slip, print_speed=60)
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "production_rate"

    @property
    def output_performance(self) -> str:
        return "production_rate"

    def _compute_target_value(self, params: dict, **dimensions: Any) -> float:
        return self.MAX_RATE

    def _compute_scaling_factor(self, params: dict, **dimensions: Any) -> float | None:
        return self.MAX_RATE
