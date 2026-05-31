"""Feature models for the ADVEI 2026 mock.

Each model wraps :class:`FabricationSystem` and exposes one or more schema
feature codes. The harness has already simulated the per-(layer, node) and
per-layer values; these models simply route the right scalar to the right
feature code — mirroring the modality split of learning-by-printing's real
feature models (vision, loadcell, energy, duration, environment, aggregate).
"""
from __future__ import annotations

from typing import Any

from pred_fab import IFeatureModel
from pred_fab.utils import PfabLogger

from fabrication import FabricationSystem
from models.schema import FeatureCode, ParamCode


class _FabFeature(IFeatureModel):
    """Base for mock feature models: holds the shared FabricationSystem."""

    def __init__(self, logger: PfabLogger, fab: FabricationSystem) -> None:
        self.fab = fab
        super().__init__(logger)

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {"dimensions": dimensions, "params": params}


class NodeVisionFeature(_FabFeature):
    """Per-(layer, node) vision features: ``node_overlap`` and ``filament_width`` (depth 2)."""

    @property
    def input_parameters(self) -> list[str]:
        return [ParamCode.PATH_OFFSET, ParamCode.CALIBRATION_FACTOR,
                ParamCode.PRINT_SPEED, ParamCode.LAYER_HEIGHT]

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.NODE_OVERLAP, FeatureCode.FILAMENT_WIDTH]

    def _compute_feature_logic(self, data: dict, params: dict, visualize: bool = False,
                               **dimensions: Any) -> dict[str, float]:
        l, n = int(dimensions["layer_idx"]), int(dimensions["node_idx"])
        return {
            FeatureCode.NODE_OVERLAP: self.fab.get_node_feature(params, FeatureCode.NODE_OVERLAP, l, n),
            FeatureCode.FILAMENT_WIDTH: self.fab.get_node_feature(params, FeatureCode.FILAMENT_WIDTH, l, n),
        }


class NodeAggregateFeature(_FabFeature):
    """Per-layer means of the per-node quality features (depth 1) — what the MLP predicts."""

    @property
    def input_parameters(self) -> list[str]:
        return [ParamCode.PATH_OFFSET, ParamCode.CALIBRATION_FACTOR,
                ParamCode.PRINT_SPEED, ParamCode.LAYER_HEIGHT]

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.NODE_OVERLAP_MEAN, FeatureCode.FILAMENT_WIDTH_MEAN]

    def _compute_feature_logic(self, data: dict, params: dict, visualize: bool = False,
                               **dimensions: Any) -> dict[str, float]:
        l = int(dimensions["layer_idx"])
        return {
            FeatureCode.NODE_OVERLAP_MEAN: self.fab.get_layer_feature(params, FeatureCode.NODE_OVERLAP_MEAN, l),
            FeatureCode.FILAMENT_WIDTH_MEAN: self.fab.get_layer_feature(params, FeatureCode.FILAMENT_WIDTH_MEAN, l),
        }


class LoadcellFeature(_FabFeature):
    """Per-layer ``loadcell_residual`` (depth 1) — extrusion-stability MSE proxy."""

    @property
    def input_parameters(self) -> list[str]:
        return [ParamCode.PRINT_SPEED, ParamCode.CALIBRATION_FACTOR]

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.LOADCELL_RESIDUAL]

    def _compute_feature_logic(self, data: dict, params: dict, visualize: bool = False,
                               **dimensions: Any) -> dict[str, float]:
        l = int(dimensions["layer_idx"])
        return {FeatureCode.LOADCELL_RESIDUAL: self.fab.get_layer_feature(params, FeatureCode.LOADCELL_RESIDUAL, l)}


class RobotEnergyFeature(_FabFeature):
    """Per-layer ``robot_energy`` (depth 1) — joint-power integral over the layer."""

    @property
    def input_parameters(self) -> list[str]:
        return [ParamCode.PRINT_SPEED, ParamCode.LAYER_HEIGHT]

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.ROBOT_ENERGY]

    def _compute_feature_logic(self, data: dict, params: dict, visualize: bool = False,
                               **dimensions: Any) -> dict[str, float]:
        l = int(dimensions["layer_idx"])
        return {FeatureCode.ROBOT_ENERGY: self.fab.get_layer_feature(params, FeatureCode.ROBOT_ENERGY, l)}


class DurationFeature(_FabFeature):
    """Per-layer ``printing_duration`` (depth 1) — toolpath length / print speed."""

    @property
    def input_parameters(self) -> list[str]:
        return [ParamCode.PRINT_SPEED]

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.PRINTING_DURATION]

    def _compute_feature_logic(self, data: dict, params: dict, visualize: bool = False,
                               **dimensions: Any) -> dict[str, float]:
        l = int(dimensions["layer_idx"])
        return {FeatureCode.PRINTING_DURATION: self.fab.get_layer_feature(params, FeatureCode.PRINTING_DURATION, l)}


class EnvironmentFeature(_FabFeature):
    """Per-layer context (depth 1): ``temperature``, ``humidity``, ``material_age``."""

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.TEMPERATURE, FeatureCode.HUMIDITY, FeatureCode.MATERIAL_AGE]

    def _compute_feature_logic(self, data: dict, params: dict, visualize: bool = False,
                               **dimensions: Any) -> dict[str, float]:
        l = int(dimensions["layer_idx"])
        return {
            FeatureCode.TEMPERATURE: self.fab.get_layer_feature(params, FeatureCode.TEMPERATURE, l),
            FeatureCode.HUMIDITY: self.fab.get_layer_feature(params, FeatureCode.HUMIDITY, l),
            FeatureCode.MATERIAL_AGE: self.fab.get_layer_feature(params, FeatureCode.MATERIAL_AGE, l),
        }


__all__ = [
    "NodeVisionFeature", "NodeAggregateFeature", "LoadcellFeature",
    "RobotEnergyFeature", "DurationFeature", "EnvironmentFeature",
]
