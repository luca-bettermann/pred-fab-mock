"""Feature models for the extrusion printing simulation."""

import numpy as np
from typing import Any

from pred_fab import IFeatureModel
from pred_fab.utils import PfabLogger

from sensors.camera import CameraSystem
from sensors.energy import EnergySensor
from sensors.physics import production_rate as _physics_production_rate


class PrintingFeatureModel(IFeatureModel):
    """Extracts path deviation from CameraSystem per (layer, segment)."""

    def __init__(self, logger: PfabLogger, camera: CameraSystem) -> None:
        self.camera = camera
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return ["water_ratio", "print_speed", "design", "material"]

    @property
    def outputs(self) -> list[str]:
        return ["path_deviation"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return self.camera.get_segment_data(
            params,
            int(dimensions["layer_idx"]),
            int(dimensions["segment_idx"]),
        )

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        return {"path_deviation": self._mean_deviation(data)}

    def _mean_deviation(self, data: dict) -> float:
        """Mean point-to-point distance between measured and designed paths."""
        return float(np.mean([
            np.linalg.norm(np.array(p) - np.array(t))
            for p, t in zip(data["measured_path"], data["designed_path"])
        ]))


class EnergyFeatureModel(IFeatureModel):
    """Extracts energy_per_segment from EnergySensor per (layer, segment)."""

    def __init__(self, logger: PfabLogger, energy_sensor: EnergySensor) -> None:
        self.energy_sensor = energy_sensor
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return ["water_ratio", "print_speed", "design", "material"]

    @property
    def outputs(self) -> list[str]:
        return ["energy_per_segment"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return self.energy_sensor.get_segment_energy(
            params,
            int(dimensions["layer_idx"]),
            int(dimensions["segment_idx"]),
        )

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        return {"energy_per_segment": float(data["energy_per_segment"])}


class ProductionRateFeatureModel(IFeatureModel):
    """Effective production rate per position, accounting for nozzle-slip at high water ratios.

    At low-to-moderate water ratios, rate tracks print_speed. Above the slip threshold
    (W_SLIP in physics.py) extrusion rate falls below commanded speed.
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio", "material"]

    @property
    def outputs(self) -> list[str]:
        return ["production_rate"]

    def _load_data(self, params: dict, **dimensions: Any) -> dict:
        return {}

    def _compute_feature_logic(
        self, data: dict, params: dict, visualize: bool = False, **dimensions: Any
    ) -> dict[str, float]:
        rate = _physics_production_rate(
            print_speed=float(params["print_speed"]),
            water_ratio=float(params["water_ratio"]),
            material=str(params["material"]),
        )
        return {"production_rate": rate}
