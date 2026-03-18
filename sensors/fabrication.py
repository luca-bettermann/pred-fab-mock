"""FabricationSystem: coordinates all sensors for a single print run."""

from typing import Any, Dict, Tuple

from .camera import CameraSystem
from .energy import EnergySensor

# Time spent per layer — a fabrication process constant, not an optimization parameter.
LAYER_TIME: float = 40.0  # seconds

# Per-design fabrication geometry.
# Each design defines the target component height and the layer/segment structure.
# layer_height is derived as target_height / n_layers, ensuring the component
# height is preserved regardless of how other parameters are tuned.
DESIGN_CONFIG: Dict[str, Dict] = {
    "A": {"n_layers": 5, "n_segments": 4, "target_height": 0.040},  # 40 mm, 8 mm/layer
    "B": {"n_layers": 5, "n_segments": 4, "target_height": 0.045},  # 45 mm, 9 mm/layer
    "C": {"n_layers": 5, "n_segments": 4, "target_height": 0.050},  # 50 mm, 10 mm/layer
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
        cfg = DESIGN_CONFIG[design]
        return cfg["n_layers"], cfg["n_segments"]

    def get_layer_height(self, design: str) -> float:
        """Return layer_height [m] derived from the design's target component height."""
        cfg = DESIGN_CONFIG[design]
        return cfg["target_height"] / cfg["n_layers"]

    def _effective_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge fabrication-process constants into params for sensor calls.

        Injects layer_time and layer_height (derived from design) so that
        physics functions receive all required inputs without these values
        being part of the optimizable schema.
        """
        return {
            **params,
            "layer_time": self.layer_time,
            "layer_height": self.get_layer_height(params["design"]),
        }

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
