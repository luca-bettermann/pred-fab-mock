"""FabricationSystem: coordinates all sensors for a single print run."""

from typing import Any

from .camera import CameraSystem
from .energy import EnergySensor
from .physics import N_LAYERS, N_SEGMENTS, TARGET_HEIGHT, PATH_LENGTH


class FabricationSystem:
    """Coordinates CameraSystem and EnergySensor for a simulated print run.

    Iterates dimensions layer by layer so that online adaptation can interleave
    sensor data collection with agent decisions between layers.
    """

    def __init__(self, camera: CameraSystem, energy: EnergySensor) -> None:
        self.camera = camera
        self.energy = energy

    @staticmethod
    def get_dimensions() -> tuple[int, int]:
        """Return (n_layers, n_segments)."""
        return N_LAYERS, N_SEGMENTS

    @staticmethod
    def get_layer_height() -> float:
        """Return layer_height [m] = target_height / n_layers."""
        return TARGET_HEIGHT / N_LAYERS

    @staticmethod
    def get_layer_time(print_speed: float) -> float:
        """Return layer_time [s] = path_length / print_speed (print_speed in mm/s)."""
        return PATH_LENGTH / (print_speed * 1e-3)  # convert mm/s → m/s

    def run_layer(self, params: dict[str, Any], layer_idx: int) -> None:
        """Populate sensor caches for all segments of a single layer."""
        self.camera.run_layer(params, layer_idx)
        self.energy.run_layer(params, layer_idx)

    def run_experiment(self, params: dict[str, Any]) -> None:
        """Populate sensor caches for all layers of a full experiment."""
        n_layers = int(params.get("n_layers", N_LAYERS))
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx)
