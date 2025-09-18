from typing import List, Optional, Type
from dataclasses import dataclass
import numpy as np

from lbp_package import IEvaluationModel, IFeatureModel
from lbp_package.utils import dim_parameter, study_parameter, exp_parameter

from implementations.features import PathDeviationFeature
from utils import visualize_geometry

@dataclass
class PathEvaluation(IEvaluationModel):
    """Example evaluation model for path deviation assessment."""

    # Model parameters
    target_deviation: Optional[float] = study_parameter(default=0.1)
    max_deviation: Optional[float] = study_parameter()
    
    # Experiment parameters
    n_layers: Optional[int] = exp_parameter()
    n_segments: Optional[int] = exp_parameter()

    # Dimensionality parameters
    layer_id: Optional[int] = dim_parameter()
    segment_id: Optional[int] = dim_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @property
    def dim_names(self) -> List[str]:
        """Return the declared dimensions."""
        return ['layers', 'segments']

    @property
    def dim_param_names(self) -> List[str]:
        """Return the dimension parameter names."""
        return ['n_layers', 'n_segments']

    @property
    def dim_iterator_names(self) -> List[str]:
        """Return the dimension iterator names."""
        return ['layer_id', 'segment_id']

    @property
    def feature_model_type(self) -> Type[IFeatureModel]:
        """Declare the feature model type to use for feature extraction."""
        return PathDeviationFeature

    @property
    def target_value(self) -> Optional[float]:
        """Return target deviation (ideally 0)."""
        return self.target_deviation

    @property
    def scaling_factor(self) -> Optional[float]:
        """Return scaling factor for the current evaluation context."""
        return self.max_deviation
    
    # === OPTIONAL METHODS ===

    # We only use this for visualization during the _cleanup step, i.e.
    def _cleanup_step(self, exp_code: str, exp_folder: str, visualize_flag: bool, debug_flag: bool) -> None:
        # skip if we don't visualize
        if not visualize_flag:
            return

        # type safety checks
        assert isinstance(self.n_segments, int), "n_segments must be an integer"
        assert isinstance(self.n_layers, int), "n_layers must be an integer"

        # Action: final evaluation is done
        if self.layer_id == self.n_layers - 1 and self.segment_id == self.n_segments - 1:

            # validate that the feature model has path coordinates
            feature_model = self.feature_model
            if not isinstance(feature_model, PathDeviationFeature):
                raise ValueError("Feature model is missing path coordinates")
            
            # Type cast to access attributes after runtime check
            assert isinstance(self.layer_id, int), "layer_id must be an integer"
            visualize_geometry(
                exp_code,
                feature_model.designed_path_coords,
                feature_model.measured_path_coords,
                float(np.average(feature_model.features['path_deviation']))
            )

            # Reset stored coordinates for next evaluation
            assert isinstance(self.feature_model, PathDeviationFeature), "Feature model must be of type PathDeviationFeature"
            self.feature_model.designed_path_coords = {}
            self.feature_model.measured_path_coords = {}
