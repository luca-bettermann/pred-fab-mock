"""sklearn-based prediction models for the extrusion printing simulation."""

import warnings

import numpy as np
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pred_fab import IPredictionModel, IDeterministicModel
from pred_fab.utils import PfabLogger

from sensors.physics import production_rate as _physics_production_rate


def _penultimate_activations(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Extract last hidden layer activations from a StandardScaler + MLPRegressor pipeline."""
    scaler = pipeline.named_steps["scaler"]
    mlp = pipeline.named_steps["mlp"]
    activation = scaler.transform(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
            activation = np.maximum(0.0, activation @ W + b)
    return activation


class DevMLP(IPredictionModel):
    """Predicts path_deviation from process parameters.

    U-shaped response to print_speed with shear-thinning coupling in water_ratio.
    """

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
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(48, 24, 12), max_iter=2000,
                alpha=0.01,
            )),
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
        return _penultimate_activations(self._model, X)


class EnergyMLP(IPredictionModel):
    """Predicts energy_per_segment from process parameters."""

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
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(24, 12), max_iter=2000,
                alpha=0.01,
            )),
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
        return _penultimate_activations(self._model, X)


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
