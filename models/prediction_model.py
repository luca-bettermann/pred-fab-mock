"""Prediction models for the extrusion printing simulation.

MLPs subclass `pred_fab.models.TorchMLPModel` — the framework owns the
torch.nn.Module + training loop. Each subclass declares only its
HIDDEN topology and the IPredictionModel properties.

Random Forests stay on sklearn (no clean PyTorch RF analog; tree
traversal isn't hurt by sklearn's per-call overhead the way pipelined
MLPs are).
"""

from typing import Any

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pred_fab import IDeterministicModel, IPredictionModel
from pred_fab.models import TorchMLPModel
from pred_fab.utils import PfabLogger

from sensors.physics import production_rate as _physics_production_rate


# ── PyTorch MLPs (framework-provided base) ───────────────────────────────

class DevMLP(TorchMLPModel):
    """Predicts path_deviation from process parameters.

    U-shaped response to print_speed with shear-thinning coupling in water_ratio.
    """

    HIDDEN = (48, 24, 12)

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


# ── Random Forest variants (kept on sklearn — no clean PyTorch analog) ───

class DevRF(IPredictionModel):
    """Random Forest model for path_deviation."""

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Pipeline | None = None
        self._is_trained = False

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

    def train(
        self,
        train_batches: list[tuple[np.ndarray, np.ndarray]],
        val_batches: list[tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            return
        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches]).ravel()
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features="sqrt")),
        ])
        self._model.fit(X, y)
        self._is_trained = True

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return self._model.predict(X).reshape(-1, 1)

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return X
        return self._model.named_steps["scaler"].transform(X)


class EnergyRF(IPredictionModel):
    """Random Forest model for energy_per_segment."""

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Pipeline | None = None
        self._is_trained = False

    @property
    def input_parameters(self) -> list[str]:
        return ["print_speed", "water_ratio"]

    @property
    def input_features(self) -> list[str]:
        return ["layer_idx_pos", "segment_idx_pos"]

    @property
    def outputs(self) -> list[str]:
        return ["energy_per_segment"]

    def train(
        self,
        train_batches: list[tuple[np.ndarray, np.ndarray]],
        val_batches: list[tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            return
        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches]).ravel()
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features="sqrt")),
        ])
        self._model.fit(X, y)
        self._is_trained = True

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return self._model.predict(X).reshape(-1, 1)

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return X
        return self._model.named_steps["scaler"].transform(X)
