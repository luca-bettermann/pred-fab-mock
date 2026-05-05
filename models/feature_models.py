"""Feature models for the ADVEI 2026 mock fabrication.

Each model wraps :class:`FabricationSystem` and exposes one or more feature
codes from the ADVEI schema. The fabrication harness has already simulated
the per-(layer, node) and per-layer values; these models simply route the
right scalar to the right feature code.

Two-feature models (vision-style, depth-2) and single-feature models
(per-layer, depth-1 or context) cover all seven ADVEI features.
"""

from __future__ import annotations

from typing import Any

from pred_fab import IFeatureModel
from pred_fab.utils import PfabLogger

from sensors.fabrication import FabricationSystem


class NodeVisionFeature(IFeatureModel):
    """Per-(layer, node) vision features: ``node_overlap`` and ``filament_width``.

    Mirrors learning-by-printing's ``NodeVisionFeatureModel`` interface
    (one model emits both depth-2 features that come from camera).
    """

    def __init__(self, logger: PfabLogger, fab: FabricationSystem) -> None:
        self.fab = fab
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return [
            "path_offset", "layer_height", "calibration_factor",
            "print_speed", "slowdown_factor",
        ]

    @property
    def outputs(self) -> list[str]:
        return ["node_overlap", "filament_width"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {
            "layer_idx": int(dimensions["layer_idx"]),
            "node_idx": int(dimensions["node_idx"]),
            "params": params,
        }

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        l = data["layer_idx"]
        n = data["node_idx"]
        return {
            "node_overlap":   self.fab.get_node_feature(params, "node_overlap", l, n),
            "filament_width": self.fab.get_node_feature(params, "filament_width", l, n),
        }


class LoadcellConsistencyFeature(IFeatureModel):
    """Per-layer ``extrusion_consistency`` (R² proxy)."""

    def __init__(self, logger: PfabLogger, fab: FabricationSystem) -> None:
        self.fab = fab
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return ["calibration_factor", "print_speed", "slowdown_factor"]

    @property
    def outputs(self) -> list[str]:
        return ["extrusion_consistency"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {"layer_idx": int(dimensions["layer_idx"]), "params": params}

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        return {
            "extrusion_consistency": self.fab.get_layer_feature(
                params, "extrusion_consistency", data["layer_idx"],
            ),
        }


class RobotEnergyFeature(IFeatureModel):
    """Per-layer ``robot_energy`` — energy consumption from robot joint currents."""

    def __init__(self, logger: PfabLogger, fab: FabricationSystem) -> None:
        self.fab = fab
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "slowdown_factor", "layer_height"]

    @property
    def outputs(self) -> list[str]:
        return ["robot_energy"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {"layer_idx": int(dimensions["layer_idx"]), "params": params}

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        return {
            "robot_energy": self.fab.get_layer_feature(
                params, "robot_energy", data["layer_idx"],
            ),
        }


class DurationFeature(IFeatureModel):
    """Per-layer ``printing_duration``. Deterministic from speed + slowdown."""

    def __init__(self, logger: PfabLogger, fab: FabricationSystem) -> None:
        self.fab = fab
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "slowdown_factor"]

    @property
    def outputs(self) -> list[str]:
        return ["printing_duration"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {"layer_idx": int(dimensions["layer_idx"]), "params": params}

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        return {
            "printing_duration": self.fab.get_layer_feature(
                params, "printing_duration", data["layer_idx"],
            ),
        }


class EnvironmentFeature(IFeatureModel):
    """Per-layer ambient context: ``temperature`` + ``humidity``."""

    def __init__(self, logger: PfabLogger, fab: FabricationSystem) -> None:
        self.fab = fab
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return ["temperature", "humidity"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {"layer_idx": int(dimensions["layer_idx"]), "params": params}

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        l = data["layer_idx"]
        return {
            "temperature": self.fab.get_layer_feature(params, "temperature", l),
            "humidity":    self.fab.get_layer_feature(params, "humidity", l),
        }
