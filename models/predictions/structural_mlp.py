"""MLP prediction model for the ADVEI structural domain.

Predicts 4 layer-level features from process parameters + context.
Right-sized for the ADVEI data regime (10-22 experiments, 100-220 rows).

Input: [path_offset, layer_height, calibration_factor, print_speed,
        material_age, temperature, humidity, layer_pos]
Output: [loadcell_residual, robot_energy,
         mean_overlap, mean_width]

printing_duration is handled by DeterministicDuration (experiment-level).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from pred_fab.models import MLPModel
from pred_fab.utils import PfabLogger

from models.schema import FeatureCode, LAYER_ITERATOR_CODE, STRUCTURAL_DOMAIN, ParamCode


class StructuralMLP(MLPModel):

    HIDDEN = (32, 16)
    EPOCHS = 500
    LR = 5e-4
    WEIGHT_DECAY = 1e-2
    DROPOUT = 0.0
    COMPILE = False
    USE_LAYER_POS = True

    def __init__(self, logger: Optional[PfabLogger] = None) -> None:
        super().__init__(logger or PfabLogger())

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Return raw input for KDE — evidence lives in parameter space, not a learned latent."""
        return X

    @property
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        return (STRUCTURAL_DOMAIN, 1)

    @property
    def input_parameters(self) -> list[str]:
        return [
            ParamCode.PATH_OFFSET,
            ParamCode.LAYER_HEIGHT,
            ParamCode.CALIBRATION_FACTOR,
            ParamCode.PRINT_SPEED,
        ]

    @property
    def input_features(self) -> list[str]:
        feats = [
            FeatureCode.MATERIAL_AGE,
            FeatureCode.TEMPERATURE,
            FeatureCode.HUMIDITY,
        ]
        if self.USE_LAYER_POS:
            feats.append(f"{LAYER_ITERATOR_CODE}_pos")
        return feats

    @property
    def outputs(self) -> list[str]:
        return [
            FeatureCode.LOADCELL_RESIDUAL,
            FeatureCode.ROBOT_ENERGY,
            FeatureCode.PRINTING_DURATION,
            FeatureCode.NODE_OVERLAP_MEAN,
            FeatureCode.FILAMENT_WIDTH_MEAN,
        ]
