"""Prediction models for the extrusion printing simulation.

`DevTransformer` (path_deviation) — sequence-aware along the layer axis;
deviation at layer k attends causally to predicted deviations at layers
0..k-1 (each layer-stack inherits substrate wobble from prior layers).

`EnergyMLP` (energy_per_segment) — flat per-(layer, segment) MLP.

`RateMLP` — deterministic, wraps the physics formula via
`DeterministicModel` (no training, identity encode).
"""

import numpy as np

from pred_fab import DeterministicModel
from pred_fab.models import MLPModel, TransformerModel

from sensors.physics import production_rate as _physics_production_rate


class DevTransformer(TransformerModel):
    """Predicts path_deviation per (layer, segment).

    Causal attention along the layer axis — each layer's predicted
    deviation depends on prior layers' predictions, modelling
    substrate-wobble propagation up the build. Segments at each layer
    are parallel sequences over layers.

    Sized for the small mock dataset: D_MODEL=16, N_LAYERS=1 — bigger
    nets overfit visibly here.
    """

    D_MODEL = 16
    N_HEADS = 2
    N_LAYERS = 1
    DIM_FEEDFORWARD = 32
    DROPOUT = 0.2
    WEIGHT_DECAY = 1e-2
    EPOCHS = 200

    @property
    def sequence_axis_code(self) -> tuple[str, ...]:
        return ("n_layers",)

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return "spatial_segment", 2

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio", "n_layers", "n_segments"]

    @property
    def input_features(self) -> list[str]:
        # No per-segment iterator features: under the new design (encoder over
        # axis_depth, decoder expands to deeper axes), input depth must be
        # ≤ axis_depth. layer_idx_pos is handled by the encoder's positional
        # embedding; segment positions are handled by the depth decoder.
        return []

    @property
    def outputs(self) -> list[str]:
        return ["path_deviation"]


class EnergyMLP(MLPModel):
    """Predicts energy_per_segment from process parameters."""

    HIDDEN = (24, 12)
    WEIGHT_DECAY = 1e-2
    DROPOUT = 0.15

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return "spatial_segment", 2

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio"]

    @property
    def input_features(self) -> list[str]:
        return ["layer_idx_pos", "segment_idx_pos"]

    @property
    def outputs(self) -> list[str]:
        return ["energy_per_segment"]


class RateMLP(DeterministicModel):
    """Deterministic production_rate [mm/s] from physics formula."""

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return None, 0

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio"]

    @property
    def input_features(self) -> list[str]:
        return []

    @property
    def outputs(self) -> list[str]:
        return ["production_rate"]

    def formula(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """X columns: [print_speed, water_ratio]."""
        results = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            ps = float(X[i, 0])
            wr = float(X[i, 1])
            results[i] = _physics_production_rate(ps, wr)
        return {"production_rate": results}
