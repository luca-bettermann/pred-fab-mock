"""Prediction models for the ADVEI 2026 mock.

Two models cover the structural domain:

- ``StructuralTransformer`` — multi-depth ``TransformerModel`` predicting
  four learned features (depth-1: ``extrusion_consistency``,
  ``robot_energy``; depth-2: ``node_overlap``, ``filament_width``). Encoder
  sequences over the layer axis; depth-2 outputs expand over the node axis
  via the default ``PerNodeMLPDecoder``.
- ``DeterministicDuration`` — closed-form ``DeterministicModel`` for
  ``printing_duration``: ``L / (speed · (1 − 0.45 · slowdown))``. Mirrors
  the simulator's formula in ``sensors.physics.feature_printing_duration``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pred_fab import DeterministicModel
from pred_fab.models import TransformerModel
from pred_fab.utils import PfabLogger

from sensors.physics import PATH_LENGTH_PER_LAYER_M


class StructuralTransformer(TransformerModel):
    """Multi-depth transformer predicting four structural target features.

    Depth-1 outputs (per layer): extrusion_consistency, robot_energy.
    Depth-2 outputs (per layer × node): node_overlap, filament_width.
    """

    # Smoke-test sized — bump for real runs.
    D_MODEL = 8
    N_HEADS = 2
    N_LAYERS = 1
    DIM_FEEDFORWARD = 16
    DROPOUT = 0.15
    WEIGHT_DECAY = 1e-2
    EPOCHS = 10

    def __init__(self, logger: Optional[PfabLogger] = None) -> None:
        super().__init__(logger or PfabLogger())

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return "structural", [1, 2]

    @property
    def sequence_axis_code(self) -> tuple[str, ...]:
        return ("n_layers",)

    @property
    def input_parameters(self) -> list[str]:
        return [
            "path_offset", "layer_height", "calibration_factor",
            "print_speed", "slowdown_factor",
            "n_layers", "n_nodes",
        ]

    @property
    def input_features(self) -> list[str]:
        return ["temperature", "humidity"]

    @property
    def outputs(self) -> list[str]:
        return [
            # depth 1
            "extrusion_consistency",
            "robot_energy",
            # depth 2
            "node_overlap",
            "filament_width",
        ]


class DeterministicDuration(DeterministicModel):
    """Closed-form ``printing_duration`` from ``print_speed`` + ``slowdown_factor``.

    Mirrors :func:`sensors.physics.feature_printing_duration` so the offline
    predictor agrees with the fab-side simulator on the planning layer.
    """

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return "structural", 1

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "slowdown_factor"]

    @property
    def input_features(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return ["printing_duration"]

    def formula(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """X columns: [print_speed, slowdown_factor]."""
        speed = X[:, 0]
        slowdown = X[:, 1]
        effective = speed * (1.0 - 0.45 * slowdown)
        # Avoid divide-by-zero for very low speeds.
        effective = np.where(effective < 1e-6, 1e-6, effective)
        return {"printing_duration": PATH_LENGTH_PER_LAYER_M / effective}
