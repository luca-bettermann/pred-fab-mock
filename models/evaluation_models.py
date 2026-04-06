"""Evaluation models for the extrusion printing simulation."""

from typing import Any, Dict, List, Optional

from pred_fab import IEvaluationModel
from pred_fab.utils import PfabLogger


class PathAccuracyModel(IEvaluationModel):
    """Scores path_deviation against a zero-deviation target."""

    MAX_DEVIATION = 0.003  # m — deviation at which score = 0

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> List[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "path_deviation"

    @property
    def output_performance(self) -> str:
        return "path_accuracy"

    def _compute_target_value(self, params: Dict, **dimensions: Any) -> float:
        return 0.0

    def _compute_scaling_factor(self, params: Dict, **dimensions: Any) -> Optional[float]:
        return self.MAX_DEVIATION


class EnergyConsumptionModel(IEvaluationModel):
    """Scores energy_per_segment against a target energy consumption.

    TARGET_ENERGY is the minimum achievable (~speed=20, clay, A).
    Scores fall off linearly as energy rises toward MAX_ENERGY (speed=60, concrete, B).
    Lower speed is always better for energy, creating genuine tension with path accuracy.
    """

    TARGET_ENERGY = 4.5   # J  — minimum achievable (low speed, clay, simple design)
    MAX_ENERGY = 24.0     # J  — max physically reachable (~speed=60, concrete, B)

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> List[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "energy_per_segment"

    @property
    def output_performance(self) -> str:
        return "energy_efficiency"

    def _compute_target_value(self, params: Dict, **dimensions: Any) -> float:
        return self.TARGET_ENERGY

    def _compute_scaling_factor(self, params: Dict, **dimensions: Any) -> Optional[float]:
        return self.MAX_ENERGY


class ProductionRateModel(IEvaluationModel):
    """Scores production_rate (print_speed) against maximum achievable speed.

    Higher speed = higher production rate = higher score.
    score = print_speed / MAX_SPEED, via target=MAX_SPEED, scaling=MAX_SPEED.
    """

    MAX_SPEED = 60.0  # mm/s

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> List[str]:
        return []

    @property
    def input_feature(self) -> str:
        return "production_rate"

    @property
    def output_performance(self) -> str:
        return "production_rate"

    def _compute_target_value(self, params: Dict, **dimensions: Any) -> float:
        return self.MAX_SPEED

    def _compute_scaling_factor(self, params: Dict, **dimensions: Any) -> Optional[float]:
        return self.MAX_SPEED
