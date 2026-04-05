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
    """Scores energy_per_segment against a target energy consumption."""

    TARGET_ENERGY = 14.0  # J  — achieved at ~speed=35 mm/s, standard material
    MAX_ENERGY = 40.0     # J  — max physically reachable (~speed=60, reinforced)

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
