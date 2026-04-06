"""sklearn-based prediction models for the extrusion printing simulation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pred_fab import IPredictionModel
from pred_fab.utils import PfabLogger

# Import for documentation: production_rate is used directly by ProductionRateFeatureModel;
# ProductionRatePredictionModel fits a linear approximation of the same formula from data.
from sensors.physics import production_rate as _physics_production_rate  # noqa: F401


def _penultimate_activations(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Extract last hidden layer activations from a StandardScaler + MLPRegressor pipeline.

    Passes X through the scaler and all hidden layers (ReLU), stopping before the
    output layer.  Returns the penultimate representation used for KDE uncertainty.
    """
    scaler = pipeline.named_steps["scaler"]
    mlp = pipeline.named_steps["mlp"]
    activation = scaler.transform(X)
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        activation = np.maximum(0.0, activation @ W + b)
    return activation


class DeviationPredictionModel(IPredictionModel):
    """Predicts path_deviation from geometry-relevant process parameters.

    path_deviation has a U-shaped response to print_speed (see physics.py):
    too slow causes material sag, too fast causes inertia overshoot.  The
    optimal speed varies by design complexity, material viscosity, and
    water_ratio via shear-thinning coupling (diagonal valley in speed×water
    space).  Three hidden layers capture these cross-term interactions.
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Optional[Pipeline] = None
        self._is_trained = False

    @property
    def input_parameters(self) -> List[str]:
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
                hidden_layer_sizes=(128, 64, 32), max_iter=2000,
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
        """Return penultimate hidden layer activations as learned latent representation."""
        if self._model is None or not self._is_trained:
            return X
        return _penultimate_activations(self._model, X)


class EnergyPredictionModel(IPredictionModel):
    """Predicts energy_per_segment from energy-relevant process parameters.

    Energy has a U-shaped water_ratio response (W_ENERGY_OPT differs from W_OPTIMAL
    for deviation), creating a genuine Pareto conflict with path accuracy.  The
    layer slope is material-specific (clay dries → less load; concrete cures → more).
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Optional[Pipeline] = None
        self._is_trained = False

    @property
    def input_parameters(self) -> List[str]:
        return ["design", "material", "print_speed", "water_ratio", "n_layers", "n_segments"]

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
        """Return penultimate hidden layer activations as learned latent representation."""
        if self._model is None or not self._is_trained:
            return X
        return _penultimate_activations(self._model, X)


class ProductionRatePredictionModel(IPredictionModel):
    """Predicts production_rate [mm/s] from print_speed alone.

    Production rate is dominated by print_speed (linear baseline); nozzle-slip
    at high water ratios is a secondary effect the calibration system accounts
    for via the physics-based feature model.  A LinearRegression trivially fits
    this relationship in normalised parameter space (R² ≈ 1.0), making training
    effectively a one-shot matrix solve rather than an iterative optimisation.

    The underlying physics formula is ``sensors.physics.production_rate``; this
    model approximates its speed-dominant term so the calibration system can
    score production rate without needing a full multi-input model here.
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Optional[Pipeline] = None
        self._is_trained = False

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
        if not train_batches:
            self.logger.warning("ProductionRatePredictionModel.train() called with no training data.")
            return
        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches]).ravel()
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        self._model.fit(X, y)
        self._is_trained = True
        self.logger.info(f"ProductionRatePredictionModel fitted on {len(X)} samples (linear, one-shot).")

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], 1))
        return self._model.predict(X).reshape(-1, 1)  # type: ignore[return-value]
