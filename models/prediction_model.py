"""sklearn-based prediction models for the extrusion printing simulation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pred_fab import IPredictionModel
from pred_fab.utils import PfabLogger


class DeviationPredictionModel(IPredictionModel):
    """Predicts path_deviation from geometry-relevant process parameters.

    path_deviation has a U-shaped response to print_speed (see physics.py):
    too slow causes material sag, too fast causes inertia overshoot.  The
    optimal speed varies by design complexity, material viscosity, and
    water_ratio (via flowability).
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Optional[Pipeline] = None
        self._is_trained = False

    @property
    def input_parameters(self) -> List[str]:
        # deviation depends on print_speed, design complexity, water_ratio (flowability),
        # and material viscosity (see physics.py).
        # n_layers=layer_idx (0-4), n_segments=segment_idx (0-3) for per-position prediction.
        return ["design", "print_speed", "water_ratio", "material", "n_layers", "n_segments"]

    @property
    def input_features(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return ["path_deviation"]

    def train(
        self,
        train_batches: List[Tuple[np.ndarray, np.ndarray]],
        val_batches: List[Tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            self.logger.warning("DeviationPredictionModel.train() called with no training data.")
            return
        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches]).ravel()
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=2000,
                random_state=42, alpha=0.01,
            )),
        ])
        self._model.fit(X, y)
        self._is_trained = True
        self.logger.info(f"DeviationPredictionModel trained on {len(X)} samples.")

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return self._model.predict(X).reshape(-1, 1)  # type: ignore[return-value]

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X


class EnergyPredictionModel(IPredictionModel):
    """Predicts energy_per_segment from energy-relevant process parameters.

    Energy scales with print_speed, material stiffness, and design path length.
    water_ratio has no effect on energy and is excluded.
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Optional[Pipeline] = None
        self._is_trained = False

    @property
    def input_parameters(self) -> List[str]:
        # energy depends on print_speed, material stiffness, and design path length
        # (see DESIGN_ENERGY_SCALE in physics.py).
        # n_layers=layer_idx (0-4), n_segments=segment_idx (0-3) for per-position prediction.
        return ["design", "material", "print_speed", "n_layers", "n_segments"]

    @property
    def input_features(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return ["energy_per_segment"]

    def train(
        self,
        train_batches: List[Tuple[np.ndarray, np.ndarray]],
        val_batches: List[Tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            self.logger.warning("EnergyPredictionModel.train() called with no training data.")
            return
        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches]).ravel()
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=2000,
                random_state=42, alpha=0.01,
            )),
        ])
        self._model.fit(X, y)
        self._is_trained = True
        self.logger.info(f"EnergyPredictionModel trained on {len(X)} samples.")

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return self._model.predict(X).reshape(-1, 1)  # type: ignore[return-value]

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X


class ProductionRatePredictionModel(IPredictionModel):
    """Deterministic production_rate prediction: rate = print_speed / 60.

    No ML needed — forward_pass implements the formula directly.
    The input is normalized print_speed (schema bounds [20, 60] → [0, 1]).
    """

    SPEED_MIN = 20.0
    SPEED_MAX = 60.0

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    def input_parameters(self) -> List[str]:
        return ["print_speed"]

    @property
    def input_features(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return ["production_rate"]

    def train(
        self,
        train_batches: List[Tuple[np.ndarray, np.ndarray]],
        val_batches: List[Tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        pass  # deterministic — no training needed

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        # X[:,0] = normalized print_speed in [0,1] via min-max on [20, 60]
        speed = X[:, 0] * (self.SPEED_MAX - self.SPEED_MIN) + self.SPEED_MIN
        return (speed / self.SPEED_MAX).reshape(-1, 1)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X
