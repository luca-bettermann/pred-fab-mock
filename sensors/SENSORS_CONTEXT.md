# sensors/ — Context

## Purpose
Simulates the two sensor systems of the extrusion printing rig. All physics is deterministic; noise is reproducible via a fixed random seed.

## Modules

| Module | Class | Description |
|---|---|---|
| `physics.py` | — | Pure physics: `path_deviation`, `energy_per_segment`; 2 designs (A,B) × 2 materials (clay, concrete) |
| `camera.py` | `CameraSystem` | Simulates camera readings; caches `{measured_path, designed_path}` per (layer, segment) |
| `energy.py` | `EnergySensor` | Simulates energy meter; caches `{energy_per_segment}` per (layer, segment); key includes design+water_ratio |
| `fabrication.py` | `FabricationSystem` | Coordinates camera+energy for one experiment or layer-by-layer |

## Key Points
- Physics has 4 distinct optima: (A,clay)≈40, (B,clay)≈33, (A,concrete)≈25, (B,concrete)≈20 mm/s.
- Layer drift: `deviation += (LAYER_DRIFT_BASE + LAYER_DRIFT_COUPLING × |speed−spd_opt_layer|) × layer_idx` — amplifies speed error, makes adaptation meaningful.
- Optimal speed shifts per layer: clay +0.4 mm/s/layer (softens), concrete −0.55 mm/s/layer (cures).
- `FILAMENT_RADIUS = 0.004` m is used in 3D visualization plots.
- Both sensors cache results by a tuple key of all relevant params + position; re-calls are free.
- `run_experiment()` populates the full cache; `run_layer()` fills one layer for online adaptation.
