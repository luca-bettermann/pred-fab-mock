# models/ — Context

## Purpose
Implements the pred-fab interfaces for feature extraction, evaluation, and prediction specific to the extrusion printing simulation.

## Modules

| Module | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `PrintingFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `path_deviation` |
| `feature_models.py` | `EnergyFeatureModel` | `IFeatureModel` | Reads from EnergySensor → outputs `energy_per_segment` |
| `evaluation_models.py` | `PathAccuracyModel` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyConsumptionModel` | `IEvaluationModel` | Scores `energy_per_segment` against 0.38 J target; MAX_ENERGY = 1.0 J |
| `prediction_model.py` | `PrintingPredictionModel` | `IPredictionModel` | sklearn MultiOutputRegressor(RandomForestRegressor) |

## Key Points
- **Schema features must match the prediction model's `outputs` (count + order).** The datamodule builds the target matrix from all schema features; a feature with no prediction output shifts the columns (the cause of the old `path_accuracy=0` bug).
- Categorical params (`design`, `material`) are in feature model inputs but **not** in the prediction model, due to DataModule one-hot encoding incompatibility with `_filter_batches_for_model`.
- `PrintingPredictionModel.encode()` is identity → KDE uncertainty estimation is used.
- The fixed spatial-domain axis sizes (`n_layers`, `n_segments`) are pushed into calibration via `agent.update_context_snapshot()` so candidate params (which carry no dimension axes) can size the prediction grid.
