"""Simulated energy meter readings for extrusion printing segments."""

from typing import Any

from ._segment_sensor import _SegmentSensor
from .physics import energy_per_segment as physics_energy


class EnergySensor(_SegmentSensor):
    """Simulates energy meter readings per print segment.

    run_layer() caches noisy energy data for every segment of a layer;
    EnergyFeature reads per-segment energy via get_segment_energy().
    """

    NOISE_ENERGY = 0.3  # J (~3% of typical 10 J per segment)

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
        return self._get_segment(params, layer_idx, segment_idx)
