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
| `prediction_model.py` | `DevMLP` | `pred_fab.models.TorchMLPModel` | PyTorch MLP (48,24,12) for path_deviation |
| `prediction_model.py` | `EnergyMLP` | `pred_fab.models.TorchMLPModel` | PyTorch MLP (24,12) for energy_per_segment |
| `prediction_model.py` | `RateMLP` | `IDeterministicModel` | Calls `physics.production_rate()` directly; no training |

## Key Points

- MLPs subclass `pred_fab.models.TorchMLPModel` — the framework owns the `nn.Module` + Adam/MSE training loop. Subclasses declare only `HIDDEN` and the IPredictionModel properties.
- `DevMLP` takes recursive features: `prev_layer_dev_1`, `prev_seg_dev_1`, plus `layer_idx_pos`, `segment_idx_pos` iterators.
- `EnergyMLP` takes only iterator features (no recursive lag).
- `RateMLP` is deterministic — no training, no KDE contribution.
