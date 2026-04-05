"""FabricationSystem: coordinates all sensors for a single print run."""

from typing import Any, Dict, Tuple

from .camera import CameraSystem
from .energy import EnergySensor

# Per-design fabrication geometry.
# Each design defines the target component height, layer/segment structure,
# and total path length per layer. layer_height and layer_time are both
# derived quantities — not optimization parameters.
#
#   layer_height = target_height / n_layers   (preserves component height)
#   layer_time   = path_length / print_speed  (derived from travel speed)
#
DESIGN_CONFIG: Dict[str, Dict] = {
    "A": {"n_layers": 5, "n_segments": 4, "target_height": 0.040, "path_length": 0.40},
    "B": {"n_layers": 5, "n_segments": 4, "target_height": 0.045, "path_length": 0.48},
    "C": {"n_layers": 5, "n_segments": 4, "target_height": 0.050, "path_length": 0.60},
}


class FabricationSystem:
    """Coordinates CameraSystem and EnergySensor for a simulated print run.

    Iterates dimensions layer by layer so that online adaptation can interleave
    sensor data collection with agent decisions between layers.

    Derives layer_height and layer_time from design geometry and print_speed,
    keeping them out of the optimization schema.
    """

    def __init__(self, camera: CameraSystem, energy: EnergySensor) -> None:
        self.camera = camera
        self.energy = energy

    def get_dimensions(self, design: str) -> Tuple[int, int]:
        """Return (n_layers, n_segments) for the given design."""
        cfg = DESIGN_CONFIG[design]
        return cfg["n_layers"], cfg["n_segments"]

    def get_layer_height(self, design: str) -> float:
        """Return layer_height [m] = target_height / n_layers for the given design."""
        cfg = DESIGN_CONFIG[design]
        return cfg["target_height"] / cfg["n_layers"]

    def get_layer_time(self, design: str, print_speed: float) -> float:
        """Return layer_time [s] = path_length / print_speed (print_speed in mm/s)."""
        cfg = DESIGN_CONFIG[design]
        return cfg["path_length"] / (print_speed * 1e-3)  # convert mm/s → m/s

    def _effective_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge derived fabrication quantities into params for sensor calls.

        Injects layer_height (from design geometry) and layer_time (from design
        path length and current print_speed) so that physics functions receive
        all required inputs without these appearing in the optimizable schema.
        """
        design = params["design"]
        print_speed = float(params["print_speed"])
        return {
            **params,
            "layer_height": self.get_layer_height(design),
            "layer_time": self.get_layer_time(design, print_speed),
        }

    def run_layer(self, params: Dict[str, Any], layer_idx: int) -> None:
        """Populate sensor caches for all segments of a single layer."""
        effective = self._effective_params(params)
        self.camera.run_layer(effective, layer_idx)
        self.energy.run_layer(effective, layer_idx)

    def run_experiment(self, params: Dict[str, Any]) -> None:
        """Populate sensor caches for all layers of a full experiment."""
        n_layers, _ = self.get_dimensions(params["design"])
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx)
