"""sklearn-based prediction model for the extrusion printing simulation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from pred_fab import IPredictionModel
from pred_fab.utils import PfabLogger


class PrintingPredictionModel(IPredictionModel):
    """Predicts path_deviation and energy_per_segment from process parameters.

    Uses a MultiOutputRegressor wrapping a RandomForestRegressor. The encode()
    method returns the identity, so uncertainty is estimated via KDE in the
    CalibrationSystem.
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)
        self._model: Optional[MultiOutputRegressor] = None
        self._is_trained = False

    @property
    def input_parameters(self) -> List[str]:
        return [
            "layer_height", "water_ratio", "print_speed",
            "design", "material", "n_layers", "n_segments",
        ]

    @property
    def input_features(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return ["path_deviation", "energy_per_segment"]

    def train(
        self,
        train_batches: List[Tuple[np.ndarray, np.ndarray]],
        val_batches: List[Tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        if not train_batches:
            self.logger.warning("PrintingPredictionModel.train() called with no training data.")
            return

        X = np.vstack([b[0] for b in train_batches])
        y = np.vstack([b[1] for b in train_batches])

        self._model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        self._model.fit(X, y)
        self._is_trained = True
        self.logger.info(f"PrintingPredictionModel trained on {len(X)} samples.")

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.zeros((X.shape[0], len(self.outputs)))
        return self._model.predict(X)  # type: ignore[return-value]

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X
