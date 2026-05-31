"""Node-level quality evaluation models.

Raised-cosine scoring: bell-shaped curve that hits exactly 0 at the
boundaries and 1 at the target. More discriminating than quadratic
near the edges.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pred_fab import IEvaluationModel
from pred_fab.utils import PfabLogger

from models.schema import AttributeCode, FeatureCode


class StructuralIntegrityEval(IEvaluationModel):
    """Node overlap quality: raised-cosine scoring.

    Measurement (``_score_row``): per-node — input_features=[NODE_OVERLAP]
    (depth=2). Framework iterates one row per node; each gets its own
    raised-cosine score; framework averages all nodes → experiment scalar.
    Penalises within-layer variance: a single off-target node drags the
    mean down.

    Acquisition (``_score_tensor``): per-layer mean — acquisition_features
    =[NODE_OVERLAP_MEAN]. MLP only predicts the per-layer mean, so the
    optimizer scores predicted means via the same cosine. Surface is
    smoother than measurement; intentional discrepancy flags
    high-variance configurations after fabrication.

    score = 0.5 * (1 + cos(π * (x - target) / target))
    Zero at x=0 and x=2*target, peak of 1.0 at x=target.
    """

    def __init__(self, logger: PfabLogger, *, target_overlap_mm: float) -> None:
        self._target = target_overlap_mm
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_features(self) -> list[str]:
        return [FeatureCode.NODE_OVERLAP]

    @property
    def acquisition_features(self) -> list[str]:
        return [FeatureCode.NODE_OVERLAP_MEAN]

    @property
    def output_performance(self) -> str:
        return AttributeCode.STRUCTURAL_INTEGRITY

    def _score_row(self, feature_values: dict[str, float], params: dict, **dims: Any) -> float:
        x = feature_values[FeatureCode.NODE_OVERLAP]
        if np.isnan(x):
            return float("nan")
        if x <= 0.0 or x >= 2.0 * self._target:
            return 0.0
        return float(0.5 * (1.0 + np.cos(np.pi * (x - self._target) / self._target)))

    def _score_tensor(self, feature_tensors: dict[str, torch.Tensor], parameters_list: list) -> torch.Tensor:
        x = feature_tensors[FeatureCode.NODE_OVERLAP_MEAN]
        score = 0.5 * (1.0 + torch.cos(torch.pi * (x - self._target) / self._target))
        score = torch.where((x <= 0) | (x >= 2 * self._target), torch.zeros_like(score), score)
        nan_mask = torch.isnan(x)
        score = torch.where(nan_mask, torch.zeros_like(score), score)
        valid = (~nan_mask).sum(dim=1).to(score.dtype)
        safe = torch.where(valid > 0, valid, torch.ones_like(valid))
        avgs = score.sum(dim=1) / safe
        return torch.where(valid > 0, avgs, torch.full_like(avgs, float("nan")))


class MaterialDepositionEval(IEvaluationModel):
    """Filament width quality: raised-cosine scoring.

    Measurement (``_score_row``): per-node — input_features=[FILAMENT_WIDTH]
    (depth=2). Each node scored independently; framework averages.
    Acquisition (``_score_tensor``): per-layer mean — uses MLP-predicted
    FILAMENT_WIDTH_MEAN. See StructuralIntegrityEval for the rationale.

    score = 0.5 * (1 + cos(π * (x - target) / half_range))
    Zero at target ± half_range, peak of 1.0 at target.
    """

    def __init__(
        self, logger: PfabLogger, *,
        target_width_mm: float,
        half_range_mm: float = 1.0,
    ) -> None:
        self._target = target_width_mm
        self._half_range = half_range_mm
        super().__init__(logger)

    @property
    def input_parameters(self) -> list[str]:
        return []

    @property
    def input_features(self) -> list[str]:
        return [FeatureCode.FILAMENT_WIDTH]

    @property
    def acquisition_features(self) -> list[str]:
        return [FeatureCode.FILAMENT_WIDTH_MEAN]

    @property
    def output_performance(self) -> str:
        return AttributeCode.MATERIAL_DEPOSITION

    def _score_row(self, feature_values: dict[str, float], params: dict, **dims: Any) -> float:
        x = feature_values[FeatureCode.FILAMENT_WIDTH]
        if np.isnan(x):
            return float("nan")
        if abs(x - self._target) >= self._half_range:
            return 0.0
        return float(0.5 * (1.0 + np.cos(np.pi * (x - self._target) / self._half_range)))

    def _score_tensor(self, feature_tensors: dict[str, torch.Tensor], parameters_list: list) -> torch.Tensor:
        x = feature_tensors[FeatureCode.FILAMENT_WIDTH_MEAN]
        score = 0.5 * (1.0 + torch.cos(torch.pi * (x - self._target) / self._half_range))
        score = torch.where((x - self._target).abs() >= self._half_range, torch.zeros_like(score), score)
        nan_mask = torch.isnan(x)
        score = torch.where(nan_mask, torch.zeros_like(score), score)
        valid = (~nan_mask).sum(dim=1).to(score.dtype)
        safe = torch.where(valid > 0, valid, torch.ones_like(valid))
        avgs = score.sum(dim=1) / safe
        return torch.where(valid > 0, avgs, torch.full_like(avgs, float("nan")))
