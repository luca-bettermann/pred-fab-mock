"""Simulated energy meter readings for extrusion printing segments."""

import numpy as np
from typing import Any, Dict, List, Tuple

from .physics import energy_per_segment as physics_energy


class EnergySensor:
    """Simulates energy meter readings per print segment.

    run_experiment() caches energy data for all positions. EnergyFeatureModel
    reads per-segment energy via get_segment_energy().
    """

    NOISE_ENERGY = 0.5  # J (~4% of typical 12 J per segment)

    def __init__(self, random_seed: int = 99) -> None:
        self._rng = np.random.RandomState(random_seed)
        self._cache: Dict[Tuple, Dict] = {}

    def _cache_key(self, params: Dict[str, Any], layer_idx: int, segment_idx: int) -> Tuple:
        # layer_height is derived from design, so design already uniquely identifies it.
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
        # Deterministic energy + noise — 10 Hz power readings over segment duration
        e = physics_energy(
            params["print_speed"], params["layer_height"],
            params["material"], params["layer_time"],
        )
        segment_duration = float(params["layer_time"]) / 4.0
        n_samples = max(1, int(segment_duration * 10))
        avg_power = e / segment_duration
        power_readings = [
            avg_power + self._rng.normal(0, self.NOISE_ENERGY / segment_duration)
            for _ in range(n_samples)
        ]
        # Pre-compute energy_per_segment so feature models don't need layer_time
        energy_per_segment = float(np.mean(power_readings)) * segment_duration
        return {"power_readings": power_readings, "energy_per_segment": energy_per_segment}

    def get_segment_energy(
        self, params: Dict[str, Any], layer_idx: int, segment_idx: int
    ) -> Dict:
        """Return cached power readings dict for a (layer, segment) position."""
        key = self._cache_key(params, layer_idx, segment_idx)
        if key not in self._cache:
            self.run_layer(params, layer_idx)
        return self._cache[key]
