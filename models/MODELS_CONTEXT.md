# models/ — Context

## Purpose
Implements the pred-fab interfaces for feature extraction, evaluation, and prediction specific to the extrusion printing simulation.

## Modules

| Module | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `PrintingFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `layer_width`, `path_deviation` |
| `feature_models.py` | `EnergyFeatureModel` | `IFeatureModel` | Reads from EnergySensor → outputs `energy_per_segment` |
| `evaluation_models.py` | `PathAccuracyModel` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyConsumptionModel` | `IEvaluationModel` | Scores `energy_per_segment` against 8 J target; MAX_ENERGY = 20 J |
| `prediction_model.py` | `PrintingPredictionModel` | `IPredictionModel` | sklearn MultiOutputRegressor(RandomForestRegressor) |

## Key Points
- Categorical params (`design`, `material`) are in feature model inputs but **not** in the prediction model, due to DataModule one-hot encoding incompatibility with `_filter_batches_for_model`.
- `PrintingPredictionModel.encode()` is identity → KDE uncertainty estimation is used.
