"""Evaluation models for the ADVEI 2026 mock.

Five evaluators map the schema's eight features to five performance
attributes (three quality + two cost). Each scores a single feature
against a fixed target with a fixed scaling tolerance — the framework's
``compute_performance`` returns ``max(0, 1 - |feat - target| / scaling)``,
clipped to ``[0, 1]`` and averaged across all cells of the feature.

For multi-feature performance attributes (e.g. ADVEI's true
``energy_footprint = V · sum(I_motors) · duration`` formula), the mock
substitutes a single-feature proxy: scoring against the dominant motor
current. learning-by-printing's real evaluator is free to override
``compute_performance`` for the multi-feature derivation.
"""

from __future__ import annotations

from typing import Any

from pred_fab import IEvaluationModel
from pred_fab.utils import PfabLogger

from sensors.physics import (
    TARGET_FILAMENT_WIDTH_MM,
    TARGET_NODE_OVERLAP_MM,
)


class StructuralIntegrityEval(IEvaluationModel):
    """Per-node node_overlap deviation from ``TARGET_NODE_OVERLAP_MM``.

    Score collapses to 0 once |overlap − target| ≥ 1.0 mm.
    """
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_feature(self) -> str: return "node_overlap"
    @property
    def output_performance(self) -> str: return "structural_integrity"

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return TARGET_NODE_OVERLAP_MM

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return 1.0  # 1 mm tolerance window


class MaterialDepositionEval(IEvaluationModel):
    """Per-node filament_width deviation from ``TARGET_FILAMENT_WIDTH_MM``."""
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_feature(self) -> str: return "filament_width"
    @property
    def output_performance(self) -> str: return "material_deposition"

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return TARGET_FILAMENT_WIDTH_MM

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return 3.0  # ±3 mm window


class ExtrusionStabilityEval(IEvaluationModel):
    """Per-layer extrusion_consistency (R²) against the perfect-line target of 1.0.

    Anything below 0.5 R² scores 0 — the ramp is meaningfully steep so
    optimiser distinguishes 0.7 from 0.95 (typical good vs. excellent fit).
    """
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_feature(self) -> str: return "extrusion_consistency"
    @property
    def output_performance(self) -> str: return "extrusion_stability"

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return 1.0

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return 0.5


class EnergyFootprintEval(IEvaluationModel):
    """Per-layer feeder-motor current as a proxy for energy footprint.

    A real implementation in learning-by-printing combines feeder + nozzle
    currents and printing_duration into total ampere-seconds; the mock uses
    feeder current alone since the framework's evaluator interface is
    single-feature. Trade-off direction is preserved (lower is better).
    """
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_feature(self) -> str: return "current_mean_feeder"
    @property
    def output_performance(self) -> str: return "energy_footprint"

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return 0.6   # A — low-current ideal

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return 1.5   # A — falloff window


class FabricationTimeEval(IEvaluationModel):
    """Per-layer printing_duration scored against a fast-print target."""
    TARGETS_CONSTANT = True

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]: return []
    @property
    def input_feature(self) -> str: return "printing_duration"
    @property
    def output_performance(self) -> str: return "fabrication_time"

    def _compute_target_value(self, params: dict, **dims: Any) -> float:
        return 80.0  # s — fast-print target

    def _compute_scaling_factor(self, params: dict, **dims: Any) -> float | None:
        return 120.0  # s — falloff window
