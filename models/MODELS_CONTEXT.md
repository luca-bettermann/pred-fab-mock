# models/ — Context

## Purpose
Implements the pred-fab interfaces for feature extraction, evaluation, and prediction specific to the extrusion printing simulation.

## Modules

| Module | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `PrintingFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `path_deviation` |
| `feature_models.py` | `EnergyFeatureModel` | `IFeatureModel` | Reads from EnergySensor → outputs `energy_per_segment` |
| `feature_models.py` | `ProductionRateFeatureModel` | `IFeatureModel` | Returns `print_speed` as `production_rate` proxy at each spatial position |
| `feature_models.py` | `ContextFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `prev_layer_deviation`, `prev_segment_deviation` (context features) |
| `evaluation_models.py` | `PathAccuracyModel` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyConsumptionModel` | `IEvaluationModel` | Scores `energy_per_segment` against 4.5 J target; MAX_ENERGY = 24 J |
| `evaluation_models.py` | `ProductionRateModel` | `IEvaluationModel` | Scores `production_rate` (higher speed = higher score); MAX_RATE = 60 mm/s |
| `prediction_model.py` | `DeviationPredictionModel` | `IPredictionModel` | sklearn MLP (64,32,16) for path_deviation; `encode()` returns penultimate activations |
| `prediction_model.py` | `EnergyPredictionModel` | `IPredictionModel` | sklearn MLP (32,16) for energy_per_segment; `encode()` returns penultimate activations |
| `prediction_model.py` | `DeviationRFModel` | `IPredictionModel` | Random Forest alternative for path_deviation; `encode()` returns scaler-transformed features |
| `prediction_model.py` | `EnergyRFModel` | `IPredictionModel` | Random Forest alternative for energy_per_segment; `encode()` returns scaler-transformed features |
| `prediction_model.py` | `ProductionRatePredictionModel` | `IDeterministicModel` | Calls `physics.production_rate()` directly; no training; inputs: print_speed+water_ratio+material |

## Key Points
- `EnergyConsumptionModel` scores lower-is-better: TARGET_ENERGY=4.5 J ≈ minimum achievable (low speed, clay, A). Combined score optimum is near the physics deviation optimum because layer drift makes deviation sensitive to speed error.
- `ProductionRateModel` scores higher-is-better: score = speed / 60. Creates genuine tension with path_accuracy (high speed → poor accuracy) and energy_efficiency (high speed → more energy).
- Calibration weights: path_accuracy=2, energy_efficiency=1, production_rate=1 (2:1:1). Combined = (2·acc + eff + rate) / 4.
- `DeviationPredictionModel` includes `design` and `material` as inputs; both categoricals are one-hot encoded by DataModule and recognized via matching column names.
- `ProductionRatePredictionModel` extends `IDeterministicModel`: calls `physics.production_rate()` directly in `formula()`. No training needed. `forward_pass()` handles denorm→formula→renorm automatically.
- `ProductionRateFeatureModel` reads inputs `["print_speed", "water_ratio", "material"]` and calls physics `production_rate()` (not a speed pass-through).
- `EnergyPredictionModel` includes `water_ratio` as input: U-shaped energy response — too dry or too wet both increase energy; MLP learns this.
- `DeviationPredictionModel` and `EnergyPredictionModel` override `encode()` to return penultimate hidden layer activations for KDE uncertainty; `ProductionRatePredictionModel` returns identity (no learned latent).
- `DeviationRFModel` and `EnergyRFModel` are RF alternatives; `encode()` returns scaler-transformed raw features (no learned latent). Selected via `model_type="rf"` in `agent_setup.build_agent()`.
- `ContextFeatureModel` computes `prev_layer_deviation` and `prev_segment_deviation` from CameraSystem cache — neighboring path_deviation values that feed into deviation prediction models as context features (observable but uncontrollable).
