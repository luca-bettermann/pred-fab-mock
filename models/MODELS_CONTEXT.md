# Models

Fabrication-specific model implementations for the extrusion printing simulation.

| File | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `DevFeature` | `IFeatureModel` | Reads from CameraSystem -> outputs `path_deviation` |
| `feature_models.py` | `EnergyFeature` | `IFeatureModel` | Reads from EnergySensor -> outputs `energy_per_segment` |
| `feature_models.py` | `RateFeature` | `IFeatureModel` | Calls `physics.production_rate()` -> outputs `production_rate` |
| `evaluation_models.py` | `PathAccuracy` | `IEvaluationModel` | Scores `path_deviation` against 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyEfficiency` | `IEvaluationModel` | Scores `energy_per_segment` against 4.5 J target; MAX_ENERGY = 24 J |
| `evaluation_models.py` | `ProductionRate` | `IEvaluationModel` | Scores `production_rate`; MAX_RATE = 60 mm/s |
| `prediction_model.py` | `DevMLP` | `IPredictionModel` | sklearn MLP (48,24,12) for path_deviation |
| `prediction_model.py` | `EnergyMLP` | `IPredictionModel` | sklearn MLP (24,12) for energy_per_segment |
| `prediction_model.py` | `DevRF` | `IPredictionModel` | Random Forest alternative for path_deviation |
| `prediction_model.py` | `EnergyRF` | `IPredictionModel` | Random Forest alternative for energy_per_segment |
| `prediction_model.py` | `RateMLP` | `IDeterministicModel` | Calls `physics.production_rate()` directly; no training |

## Key Points

- `DevMLP` takes recursive features: `prev_layer_deviation` and `prev_segment_deviation` at lag 1-2.
- `EnergyMLP` takes only parameters (no lag features).
- `RateMLP` is deterministic — no training, no KDE contribution.
- RF variants (`DevRF`, `EnergyRF`) are for comparison in dev/03_prediction.py.
