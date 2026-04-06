# models/ — Context

## Purpose
Implements the pred-fab interfaces for feature extraction, evaluation, and prediction specific to the extrusion printing simulation.

## Modules

| Module | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `PrintingFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `path_deviation` |
| `feature_models.py` | `EnergyFeatureModel` | `IFeatureModel` | Reads from EnergySensor → outputs `energy_per_segment` |
| `feature_models.py` | `ProductionRateFeatureModel` | `IFeatureModel` | Returns `print_speed` as `production_rate` proxy at each spatial position |
| `evaluation_models.py` | `PathAccuracyModel` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyConsumptionModel` | `IEvaluationModel` | Scores `energy_per_segment` against 4.5 J target; MAX_ENERGY = 24 J |
| `evaluation_models.py` | `ProductionRateModel` | `IEvaluationModel` | Scores `production_rate` (higher speed = higher score); MAX_SPEED = 60 mm/s |
| `prediction_model.py` | `DeviationPredictionModel` | `IPredictionModel` | sklearn MLP for path_deviation; inputs include design+material+speed+water+position |
| `prediction_model.py` | `EnergyPredictionModel` | `IPredictionModel` | sklearn MLP for energy_per_segment; inputs include design+material+speed+water_ratio+position |
| `prediction_model.py` | `ProductionRatePredictionModel` | `IPredictionModel` | sklearn MLP for production_rate; inputs include print_speed+water_ratio+material |

## Key Points
- `EnergyConsumptionModel` scores lower-is-better: TARGET_ENERGY=4.5 J ≈ minimum achievable (low speed, clay, A). Combined score optimum is near the physics deviation optimum because layer drift makes deviation sensitive to speed error.
- `ProductionRateModel` scores higher-is-better: score = speed / 60. Creates genuine tension with path_accuracy (high speed → poor accuracy) and energy_efficiency (high speed → more energy).
- Calibration weights: path_accuracy=2, energy_efficiency=1, production_rate=1 (2:1:1). Combined = (2·acc + eff + rate) / 4.
- `DeviationPredictionModel` includes `design` and `material` as inputs; both categoricals are one-hot encoded by DataModule and recognized via matching column names.
- `ProductionRatePredictionModel` is a trained MLP (hidden=[32,16]): nozzle-slip physics means rate collapses below the W_SLIP water threshold — non-linear, requires learning.
- `ProductionRateFeatureModel` reads inputs `["print_speed", "water_ratio", "material"]` and calls physics `production_rate()` (not a speed pass-through).
- `EnergyPredictionModel` includes `water_ratio` as input: U-shaped energy response — too dry or too wet both increase energy; MLP learns this.
- `encode()` is identity for all models → KDE uncertainty estimation operates in raw parameter space.
