"""ExtrusionStability ← loadcell_residual (MSE).

Linear: score 1 at MSE=0, score 0 at MSE>=mse_max.
score = clip(1 - mse / mse_max, 0, 1)

Lower is better, no target — same shape as energy_footprint and
fabrication_time. See [[PFAB - Feature Roles]] for the rationale
(minimise → first-order; hit-the-target → second-order).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pred_fab import IEvaluationModel
from pred_fab.utils import PfabLogger

from models.schema import AttributeCode, FeatureCode


class ExtrusionStabilityEval(IEvaluationModel):
    """Per-layer extrusion-stability score — linear, decreasing in MSE."""

    def __init__(self, logger: PfabLogger, *, mse_max: float = 800.0) -> None:
        self._mse_max = mse_max
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_features(self) -> list[str]:
        return [FeatureCode.LOADCELL_RESIDUAL]

    @property
    def output_performance(self) -> str:
        return AttributeCode.EXTRUSION_STABILITY

    def _score_row(self, feature_values: dict[str, float], params: dict, **dims: Any) -> float:
        mse = feature_values[FeatureCode.LOADCELL_RESIDUAL]
        if np.isnan(mse):
            return float("nan")
        return float(np.clip(1.0 - mse / self._mse_max, 0.0, 1.0))

    def _score_tensor(self, feature_tensors: dict[str, torch.Tensor], parameters_list: list) -> torch.Tensor:
        mse_t = feature_tensors[FeatureCode.LOADCELL_RESIDUAL]
        s = (1.0 - mse_t / self._mse_max).clamp(0.0, 1.0)
        nan_mask = torch.isnan(mse_t)
        s = torch.where(nan_mask, torch.zeros_like(s), s)
        valid = (~nan_mask).sum(dim=1).to(s.dtype)
        safe = torch.where(valid > 0, valid, torch.ones_like(valid))
        avgs = s.sum(dim=1) / safe
        return torch.where(valid > 0, avgs, torch.full_like(avgs, float("nan")))
