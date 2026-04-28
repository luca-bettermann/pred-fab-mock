"""Simulated energy meter readings for extrusion printing segments."""

import numpy as np
from typing import Any

from .physics import energy_per_segment as physics_energy


class EnergySensor:
    """Simulates energy meter readings per print segment.

    run_experiment() caches energy data for all positions. EnergyFeature
    reads per-segment energy via get_segment_energy().
    """

    NOISE_ENERGY = 0.3  # J (~3% of typical 10 J per segment)

    def __init__(self, random_seed: int | None = None) -> None:
        self._rng = np.random.RandomState(random_seed)
        self._cache: dict[tuple, dict] = {}

    def _cache_key(self, params: dict[str, Any], layer_idx: int, segment_idx: int) -> tuple:
        return (
            params["print_speed"], params["water_ratio"],
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
        e = physics_energy(
            params["print_speed"], params["water_ratio"],
            segment_idx=segment_idx, layer_idx=layer_idx,
        )
        energy_noisy = e + self._rng.normal(0, self.NOISE_ENERGY)
        return {"energy_per_segment": float(max(0.0, energy_noisy))}

    def get_segment_energy(
        self, params: dict[str, Any], layer_idx: int, segment_idx: int
    ) -> dict:
        """Return cached power readings dict for a (layer, segment) position."""
        key = self._cache_key(params, layer_idx, segment_idx)
        if key not in self._cache:
            self.run_layer(params, layer_idx)
        return self._cache[key]
