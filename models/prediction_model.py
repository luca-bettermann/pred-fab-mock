"""Prediction models for the extrusion printing simulation.

MLPs are PyTorch-based for low per-call inference overhead — important
during exploration / acquisition where the autoregressive loop runs the
model thousands of times on single-row inputs. Random Forests stay on
sklearn (no clean PyTorch analog; tree traversal isn't hurt by sklearn's
per-call overhead the way pipelined MLPs are).
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pred_fab import IPredictionModel, IDeterministicModel
from pred_fab.utils import PfabLogger

from sensors.physics import production_rate as _physics_production_rate


# ── PyTorch MLP infrastructure ────────────────────────────────────────────

class _TorchMLP(nn.Module):
    """Plain feed-forward MLP. Drop-in replacement for sklearn MLPRegressor.

    Inputs already arrive normalised from DataModule (zero-mean / unit-variance
    per column), so no internal StandardScaler is needed.
    """

    def __init__(self, n_inputs: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        # All layers except the final Linear; matches sklearn's
        # _penultimate_activations semantics for KDE encode.
        layers = list(self.net.children())
        for layer in layers[:-1]:
            x = layer(x)
        return x


def _train_torch_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: tuple[int, ...],
    *,
    epochs: int = 1500,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> _TorchMLP:
    """Train a _TorchMLP with Adam + MSE on the given (X, y) batch."""
    torch.manual_seed(seed)
    n_inputs = X.shape[1]
    model = _TorchMLP(n_inputs, hidden)

    X_t = torch.from_numpy(X.astype(np.float32))  # type: ignore[reportPrivateImportUsage]
    y_t = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)  # type: ignore[reportPrivateImportUsage]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _torch_predict(model: _TorchMLP, X: np.ndarray) -> np.ndarray:
    """Single forward pass with autograd disabled — the inference hot path."""
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32))  # type: ignore[reportPrivateImportUsage]
        return model(X_t).numpy().reshape(-1, 1)


def _torch_penultimate(model: _TorchMLP, X: np.ndarray) -> np.ndarray:
    """Penultimate-layer activations for KDE encode."""
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32))  # type: ignore[reportPrivateImportUsage]
        return model.penultimate(X_t).numpy()


# ── Models ────────────────────────────────────────────────────────────────

class DevMLP(IPredictionModel):
    """Predicts path_deviation from process parameters.

    U-shaped response to print_speed with shear-thinning coupling in water_ratio.
    """

    HIDDEN = (48, 24, 12)

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: _TorchMLP | None = None
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
        self._model = _train_torch_mlp(X, y, hidden=self.HIDDEN)
        self._is_trained = True

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return _torch_predict(self._model, X)

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return X
        return _torch_penultimate(self._model, X)


class EnergyMLP(IPredictionModel):
    """Predicts energy_per_segment from process parameters."""

    HIDDEN = (24, 12)

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: _TorchMLP | None = None
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
        self._model = _train_torch_mlp(X, y, hidden=self.HIDDEN)
        self._is_trained = True

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return _torch_predict(self._model, X)

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return X
        return _torch_penultimate(self._model, X)


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
