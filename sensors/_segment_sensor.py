"""Shared cache/iteration machinery for simulated per-segment sensors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class _SegmentSensor(ABC):
    """Caches simulated readings per (params, layer, segment) position.

    run_layer() simulates and caches every segment of a layer; subclasses
    implement _simulate_segment() and expose a typed getter that serves
    single positions via _get_segment().
    """

    def __init__(self, random_seed: int | None = None) -> None:
        self._rng = np.random.RandomState(random_seed)
        self._cache: dict[tuple, dict] = {}

    def _cache_key(self, params: dict[str, Any], layer_idx: int, segment_idx: int) -> tuple:
        return (
            params["water_ratio"], params["print_speed"],
            layer_idx, segment_idx,
        )

    def run_layer(self, params: dict[str, Any], layer_idx: int) -> None:
        """Simulate and cache all segments for a single layer."""
        n_segments = int(params["n_segments"])
        for segment_idx in range(n_segments):
            key = self._cache_key(params, layer_idx, segment_idx)
            if key not in self._cache:
                self._cache[key] = self._simulate_segment(params, layer_idx, segment_idx)

    def _get_segment(
        self, params: dict[str, Any], layer_idx: int, segment_idx: int
    ) -> dict:
        """Return cached data for a position, simulating the layer on a miss."""
        key = self._cache_key(params, layer_idx, segment_idx)
        if key not in self._cache:
            self.run_layer(params, layer_idx)
        return self._cache[key]

    @abstractmethod
    def _simulate_segment(
        self, params: dict[str, Any], layer_idx: int, segment_idx: int
    ) -> dict:
        """Return the simulated sensor reading for one (layer, segment)."""
