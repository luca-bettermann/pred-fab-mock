"""Closed-form ``printing_duration`` predictor.

``printing_duration = path_length / print_speed`` is analytic, not learned —
predicting it with the MLP both wastes a learnable channel and lets its large
magnitude skew the shared multi-output loss. A ``DeterministicModel`` mirrors
``fabrication.physics.feature_printing_duration`` exactly, so the offline
predictor agrees with the fab-side simulator on the planning layer.
"""
from __future__ import annotations

import numpy as np

from pred_fab import DeterministicModel

from models.schema import (
    FeatureCode, ParamCode, STRUCTURAL_DOMAIN, PATH_LENGTH_M,
)


class DeterministicDuration(DeterministicModel):
    """Per-layer ``printing_duration`` from ``print_speed`` — no training."""

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return STRUCTURAL_DOMAIN, 1

    @property
    def input_parameters(self) -> list[str]:
        return [ParamCode.PRINT_SPEED]

    @property
    def input_features(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return [FeatureCode.PRINTING_DURATION]

    def formula(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """X columns: [print_speed]."""
        speed = np.where(X[:, 0] < 1e-6, 1e-6, X[:, 0])
        return {FeatureCode.PRINTING_DURATION: PATH_LENGTH_M / speed}
