# models/ — Context

## Purpose
Implements the pred-fab interfaces for feature extraction, evaluation, and prediction specific to the extrusion printing simulation.

## Modules

| Module | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `PrintingFeatureModel` | `IFeatureModel` | Reads from CameraSystem → outputs `path_deviation` |
| `feature_models.py` | `EnergyFeatureModel` | `IFeatureModel` | Reads from EnergySensor → outputs `energy_per_segment` |
| `evaluation_models.py` | `PathAccuracyModel` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyConsumptionModel` | `IEvaluationModel` | Scores `energy_per_segment` against 4.5 J target; MAX_ENERGY = 24 J |
| `prediction_model.py` | `DeviationPredictionModel` | `IPredictionModel` | sklearn MLP for path_deviation; inputs include design+material+speed+water+position |
| `prediction_model.py` | `EnergyPredictionModel` | `IPredictionModel` | sklearn MLP for energy_per_segment; inputs include design+material+speed+position |

## Key Points
- `EnergyConsumptionModel` scores lower-is-better: TARGET_ENERGY=4.5 J ≈ minimum achievable (low speed, clay, A). Combined score optimum is near the physics deviation optimum because layer drift makes deviation sensitive to speed error.
- `DeviationPredictionModel` includes `design` and `material` as inputs; both categoricals are one-hot encoded by DataModule and recognized via matching column names.
- `EnergyPredictionModel` includes `design` because energy now depends on DESIGN_ENERGY_SCALE (B has longer path → more energy).
- `encode()` is identity for both models → KDE uncertainty estimation operates in raw parameter space.
