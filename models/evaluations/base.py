"""Convenience base classes for ADVEI evaluation models.

TargetEvaluationModel: single-feature, scores by closeness to target.
Implements the `1 - |feat - target| / denom` pattern that most ADVEI
eval models share. Lives in lbp, not pred-fab — it's a domain
convenience, not a framework concept.

Depends on the unified multi-feature IEvaluationModel interface
(pred-fab task: Unified multi-feature evaluation interface).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from pred_fab import IEvaluationModel
from pred_fab.utils import PfabLogger


class TargetEvaluationModel(IEvaluationModel):
    """Scores a single feature by closeness to a parameter-dependent target.

    Subclasses override ``input_feature``, ``output_performance``,
    ``_compute_target_value``, and optionally ``_compute_scaling_factor``.
    """

    def __init__(self, logger: PfabLogger) -> None:
        super().__init__(logger)

    @property
    @abstractmethod
    def input_feature(self) -> str: ...

    @property
    def input_features(self) -> list[str]:
        return [self.input_feature]

    @abstractmethod
    def _compute_target_value(self, params: dict, **dimensions: Any) -> float: ...

    def _compute_scaling_factor(self, params: dict, **dimensions: Any) -> float | None:
        return None

    def _score_row(
        self,
        feature_values: dict[str, float],
        params: dict,
        **dimensions: Any,
    ) -> float:
        value = feature_values[self.input_feature]
        if value is None or np.isnan(value):
            return float("nan")
        target = self._compute_target_value(params, **dimensions)
        scaling = self._compute_scaling_factor(params, **dimensions)
        denom = scaling if scaling and scaling > 0 else (target if target > 0 else 1.0)
        return float(np.clip(1.0 - max(0.0, value - target) / denom, 0.0, 1.0))

    def _score_tensor(
        self,
        feature_tensors: dict[str, torch.Tensor],
        parameters_list: list,
    ) -> torch.Tensor:
        feat_t = feature_tensors[self.input_feature]  # (S, n_rows)
        S = feat_t.shape[0]
        targets = torch.empty(S, dtype=feat_t.dtype)
        denoms = torch.empty(S, dtype=feat_t.dtype)
        for s, params_obj in enumerate(parameters_list):
            params = params_obj.get_values_dict()
            t = float(self._compute_target_value(params))
            sc = self._compute_scaling_factor(params)
            targets[s] = t
            if sc is not None and sc > 0:
                denoms[s] = float(sc)
            elif t > 0:
                denoms[s] = t
            else:
                denoms[s] = float("nan")
        perfs = 1.0 - (feat_t - targets[:, None]).clamp(min=0.0) / denoms[:, None]
        perfs = torch.clamp(perfs, 0.0, 1.0)
        nan_mask = torch.isnan(feat_t)
        perfs_safe = torch.where(nan_mask, torch.zeros_like(perfs), perfs)
        valid_count = (~nan_mask).sum(dim=1).to(perfs.dtype)
        safe_count = torch.where(valid_count > 0, valid_count, torch.ones_like(valid_count))
        avgs = perfs_safe.sum(dim=1) / safe_count
        return torch.where(valid_count > 0, avgs, torch.full_like(avgs, float("nan")))
