"""FabricationSystem: coordinates all sensors for a single print run."""

from typing import Any, Dict

from .camera import CameraSystem
from .energy import EnergySensor


class FabricationSystem:
    """Coordinates CameraSystem and EnergySensor for a simulated print run.

    Iterates over dimensions layer by layer so that online adaptation can
    interleave sensor data collection with agent decisions between layers.
    """

    def __init__(self, camera: CameraSystem, energy: EnergySensor) -> None:
        self.camera = camera
        self.energy = energy

    def run_layer(self, params: Dict[str, Any], layer_idx: int) -> None:
        """Populate sensor caches for all segments of a single layer."""
        self.camera.run_layer(params, layer_idx)
        self.energy.run_layer(params, layer_idx)

    def run_experiment(self, params: Dict[str, Any]) -> None:
        """Populate sensor caches for all layers of a full experiment."""
        n_layers = int(params["n_layers"])
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx)
