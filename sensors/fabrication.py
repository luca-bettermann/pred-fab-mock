"""FabricationSystem: coordinates all sensors for a single print run."""

from typing import Any, Dict, Tuple

from .camera import CameraSystem
from .energy import EnergySensor

# Time spent per layer — a fabrication process constant, not an optimization parameter.
LAYER_TIME: float = 40.0  # seconds

# Number of layers and segments for each design type.
# In a real system these would reflect the actual path geometry; in this mock
# all designs share the same 5×4 grid so data structures stay consistent.
DESIGN_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "A": (5, 4),
    "B": (5, 4),
    "C": (5, 4),
}


class FabricationSystem:
    """Coordinates CameraSystem and EnergySensor for a simulated print run.

    Iterates dimensions layer by layer so that online adaptation can interleave
    sensor data collection with agent decisions between layers.

    Owns all fabrication-process constants (layer_time, design dimensions) that
    are not optimization parameters and therefore not part of the schema.
    """

    layer_time: float = LAYER_TIME

    def __init__(self, camera: CameraSystem, energy: EnergySensor) -> None:
        self.camera = camera
        self.energy = energy

    def get_dimensions(self, design: str) -> Tuple[int, int]:
        """Return (n_layers, n_segments) for the given design."""
        return DESIGN_DIMENSIONS[design]

    def _effective_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge fabrication-process constants into params for sensor calls."""
        return {**params, "layer_time": self.layer_time}

    def run_layer(self, params: Dict[str, Any], layer_idx: int) -> None:
        """Populate sensor caches for all segments of a single layer."""
        print(f"Printing layer {layer_idx}...")
        effective = self._effective_params(params)
        self.camera.run_layer(effective, layer_idx)
        self.energy.run_layer(effective, layer_idx)

    def run_experiment(self, params: Dict[str, Any]) -> None:
        """Populate sensor caches for all layers of a full experiment."""
        n_layers, _ = self.get_dimensions(params["design"])
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx)
