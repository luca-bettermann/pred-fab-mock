"""Simulated camera-based geometry sensing for extrusion printing."""

import numpy as np
from typing import Any, Dict, Tuple

from .physics import path_deviation


class CameraSystem:
    """Simulates camera-based geometry sensing for each print segment.

    run_experiment() iterates all (layer, segment) positions, calls physics + noise,
    caches raw visual data keyed by parameter hash. PrintingFeatureModel reads from
    cache position-by-position via get_segment_data().
    """

    NOISE_DEVIATION = 0.00010  # m (~7% of typical deviation)

    def __init__(self, random_seed: int = 42) -> None:
        self._rng = np.random.RandomState(random_seed)
        self._cache: Dict[Tuple, Dict] = {}

    def _cache_key(self, params: Dict[str, Any], layer_idx: int, segment_idx: int) -> Tuple:
        return (
            params["water_ratio"], params["print_speed"],
            params["design"], params["material"],
            layer_idx, segment_idx,
        )

    def run_experiment(self, params: Dict[str, Any]) -> None:
        """Simulate and cache all (layer, segment) positions for the given experiment params."""
        n_layers = int(params["n_layers"])
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx)

    def run_layer(self, params: Dict[str, Any], layer_idx: int) -> None:
        """Simulate and cache all segments for a single layer."""
        n_segments = int(params["n_segments"])
        for segment_idx in range(n_segments):
            key = self._cache_key(params, layer_idx, segment_idx)
            if key not in self._cache:
                self._cache[key] = self._simulate_segment(params, layer_idx, segment_idx)

    def _simulate_segment(
        self, params: Dict[str, Any], layer_idx: int, segment_idx: int
    ) -> Dict:
        d = path_deviation(
            params["print_speed"], params["design"], segment_idx,
            params["water_ratio"], params["material"], layer_idx=layer_idx,
        )

        # Measured vs designed path: 5 sample points.
        # Noise is lateral-only (y axis); x is along-path and not a source of deviation.
        designed_path = [(float(i) * 0.01, 0.0) for i in range(5)]
        measured_path = [
            (p[0], p[1] + d + self._rng.normal(0, self.NOISE_DEVIATION))
            for p in designed_path
        ]

        return {
            "measured_path": measured_path,
            "designed_path": designed_path,
        }

    def get_segment_data(
        self, params: Dict[str, Any], layer_idx: int, segment_idx: int
    ) -> Dict:
        """Return cached visual sensor data for a (layer, segment) position."""
        key = self._cache_key(params, layer_idx, segment_idx)
        if key not in self._cache:
            self.run_layer(params, layer_idx)
        return self._cache[key]
