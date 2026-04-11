# models/ — Context

## Purpose
Implements the pred-fab interfaces for feature extraction, evaluation, and prediction specific to the extrusion printing simulation.

## Modules

| Module | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `PrintingFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `path_deviation` |
| `feature_models.py` | `EnergyFeatureModel` | `IFeatureModel` | Reads from EnergySensor → outputs `energy_per_segment` |
| `feature_models.py` | `ProductionRateFeatureModel` | `IFeatureModel` | Calls `physics.production_rate()` → outputs `production_rate` |
| `evaluation_models.py` | `PathAccuracyModel` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyConsumptionModel` | `IEvaluationModel` | Scores `energy_per_segment` against 4.5 J target; MAX_ENERGY = 24 J |
| `evaluation_models.py` | `ProductionRateModel` | `IEvaluationModel` | Scores `production_rate`; MAX_RATE = 60 mm/s |
| `prediction_model.py` | `DeviationPredictionModel` | `IPredictionModel` | sklearn MLP (48,24,12) for path_deviation |
| `prediction_model.py` | `EnergyPredictionModel` | `IPredictionModel` | sklearn MLP (24,12) for energy_per_segment |
| `prediction_model.py` | `DeviationRFModel` | `IPredictionModel` | Random Forest alternative for path_deviation |
| `prediction_model.py` | `EnergyRFModel` | `IPredictionModel` | Random Forest alternative for energy_per_segment |
| `prediction_model.py` | `ProductionRatePredictionModel` | `IDeterministicModel` | Calls `physics.production_rate()` directly; no training |

## Key Points
- Input parameters for all models: `print_speed`, `water_ratio` (+ `n_layers`, `n_segments` for dimensional models).
- `DeviationPredictionModel` also takes recursive features: `prev_layer_deviation`, `prev_segment_deviation`.
- Calibration weights: path_accuracy=2, energy_efficiency=1, production_rate=1 (2:1:1).
- MLP models override `encode()` → penultimate hidden layer activations for KDE uncertainty.
- RF models override `encode()` → scaler-transformed raw features. Selected via `model_type="rf"` in `build_agent()`.
- `ProductionRatePredictionModel` is deterministic — no training, no KDE contribution.
