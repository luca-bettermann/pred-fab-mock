"""Simulated camera-based geometry sensing for extrusion printing."""

import numpy as np
from typing import Any

from .physics import path_deviation


class CameraSystem:
    """Simulates camera-based geometry sensing for each print segment.

    run_experiment() iterates all (layer, segment) positions, calls physics + noise,
    caches raw visual data keyed by parameter hash. PrintingFeatureModel reads from
    cache position-by-position via get_segment_data().
    """

    NOISE_DEVIATION = 0.00010  # m (~7% of typical deviation)

    def __init__(self, random_seed: int | None = None) -> None:
        self._rng = np.random.RandomState(random_seed)
        self._cache: dict[tuple, dict] = {}

    def _cache_key(self, params: dict[str, Any], layer_idx: int, segment_idx: int) -> tuple:
        return (
            params["water_ratio"], params["print_speed"],
            layer_idx, segment_idx,
        )

    def run_experiment(self, params: dict[str, Any]) -> None:
        """Simulate and cache all (layer, segment) positions."""
        n_layers = int(params["n_layers"])
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx)

    def run_layer(self, params: dict[str, Any], layer_idx: int) -> None:
        """Simulate and cache all segments for a single layer."""
        n_segments = int(params["n_segments"])
        for segment_idx in range(n_segments):
            key = self._cache_key(params, layer_idx, segment_idx)
            if key not in self._cache:
                self._cache[key] = self._simulate_segment(params, layer_idx, segment_idx)

    def _simulate_segment(
        self, params: dict[str, Any], layer_idx: int, segment_idx: int
    ) -> dict:
        d = path_deviation(
            params["print_speed"], segment_idx,
            params["water_ratio"], layer_idx=layer_idx,
        )

        # Measured vs designed path: 5 sample points.
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
        self, params: dict[str, Any], layer_idx: int, segment_idx: int
    ) -> dict:
        """Return cached visual sensor data for a (layer, segment) position."""
        key = self._cache_key(params, layer_idx, segment_idx)
        if key not in self._cache:
            self.run_layer(params, layer_idx)
        return self._cache[key]
