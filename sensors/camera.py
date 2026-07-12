"""Simulated camera-based geometry sensing for extrusion printing."""

from typing import Any

from ._segment_sensor import _SegmentSensor
from .physics import PATH_SAMPLES, SAMPLE_SPACING, path_deviation


class CameraSystem(_SegmentSensor):
    """Simulates camera-based geometry sensing for each print segment.

    run_layer() calls physics + noise for every segment of a layer and
    caches the raw visual data; DevFeature reads from the cache
    position-by-position via get_segment_data().
    """

    NOISE_DEVIATION = 0.00010  # m (~7% of typical deviation)

    def _simulate_segment(
        self, params: dict[str, Any], layer_idx: int, segment_idx: int
    ) -> dict:
        d = path_deviation(
            params["print_speed"], segment_idx,
            params["water_ratio"], layer_idx=layer_idx,
        )

        designed_path = [(float(i) * SAMPLE_SPACING, 0.0) for i in range(PATH_SAMPLES)]
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
        return self._get_segment(params, layer_idx, segment_idx)
