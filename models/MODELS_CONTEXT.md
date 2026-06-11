# Models

Fabrication-specific model implementations for the extrusion printing simulation.

| File | Class | Interface | Description |
|---|---|---|---|
| `feature_models.py` | `DevFeature` | `IFeatureModel` | Reads from CameraSystem -> outputs `path_deviation` |
| `feature_models.py` | `EnergyFeature` | `IFeatureModel` | Reads from EnergySensor -> outputs `energy_per_segment` |
| `feature_models.py` | `RateFeature` | `IFeatureModel` | Calls `physics.production_rate()` -> outputs `production_rate` |
| `evaluation_models.py` | `_LinearTargetScore` | `IEvaluationModel` | Shared base: `score = 1 − |feature − TARGET| / SCALE` (`_score_row` + batched `_score_tensor`) |
| `evaluation_models.py` | `PathAccuracy` | `_LinearTargetScore` | `path_deviation` vs 0 target; MAX_DEVIATION = 0.003 m |
| `evaluation_models.py` | `EnergyEfficiency` | `_LinearTargetScore` | `energy_per_segment` vs 4.5 J target; MAX_ENERGY = 24 J |
| `evaluation_models.py` | `ProductionRate` | `_LinearTargetScore` | `production_rate` vs 60 mm/s; MAX_RATE = 60 mm/s |
| `prediction_model.py` | `DevTransformer` | `pred_fab.models.TransformerModel` | Causal attention along the layer axis for path_deviation (D_MODEL=16, 1 layer) |
| `prediction_model.py` | `EnergyMLP` | `pred_fab.models.MLPModel` | PyTorch MLP (24,12) for energy_per_segment |
| `prediction_model.py` | `RateMLP` | `DeterministicModel` | Calls `physics.production_rate()` directly; no training |

## Key Points

- Prediction models subclass the framework's torch models — the framework owns the `nn.Module` + training loop; subclasses declare sizes and the IPredictionModel properties.
- `DevTransformer` takes no per-segment iterator features (input depth must be ≤ axis depth); layer position comes from the encoder's positional embedding, segment positions from the depth decoder. **No variable-length sequences yet** — `n_layers` must be pinned during acquisition.
- `EnergyMLP` takes `layer_idx_pos` + `segment_idx_pos` iterator features.
- `RateMLP` is deterministic — no training, no evidence contribution.
- Evaluation models implement the multi-feature `IEvaluationModel` contract (`input_features` list, `_score_row` per-row numpy, `_score_tensor` batched torch for gradient acquisition); the framework orchestrates row iteration, batching, and NaN handling.
