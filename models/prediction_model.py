"""Prediction models for the extrusion printing simulation.

MLPs subclass `pred_fab.models.TorchMLPModel`. Each subclass declares
HIDDEN topology and the IPredictionModel properties; the framework
owns the training loop, forward path, and torch lifecycle.

`RateMLP` is deterministic — it wraps the physics formula via
`IDeterministicModel` (no training, identity encode).
"""

import numpy as np

from pred_fab import IDeterministicModel
from pred_fab.models import TorchMLPModel

from sensors.physics import production_rate as _physics_production_rate


class DevMLP(TorchMLPModel):
    """Predicts path_deviation from process parameters.

    U-shaped response to print_speed with shear-thinning coupling in water_ratio.
    """

    HIDDEN = (24, 12)

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio"]

    @property
    def input_features(self) -> list[str]:
        return [
            "prev_layer_dev_1",
            "prev_seg_dev_1",
            "layer_idx_pos",
            "segment_idx_pos",
        ]

    @property
    def outputs(self) -> list[str]:
        return ["path_deviation"]


class EnergyMLP(TorchMLPModel):
    """Predicts energy_per_segment from process parameters."""

    HIDDEN = (24, 12)

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio"]

    @property
    def input_features(self) -> list[str]:
        return ["layer_idx_pos", "segment_idx_pos"]

    @property
    def outputs(self) -> list[str]:
        return ["energy_per_segment"]


class RateMLP(IDeterministicModel):
    """Deterministic production_rate [mm/s] from physics formula."""

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio"]

    @property
    def input_features(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return ["production_rate"]

    def formula(self, X: np.ndarray) -> np.ndarray:
        """X columns: [print_speed, water_ratio]."""
        results = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            ps = float(X[i, 0])
            wr = float(X[i, 1])
            results[i] = _physics_production_rate(ps, wr)
        return results.reshape(-1, 1)
