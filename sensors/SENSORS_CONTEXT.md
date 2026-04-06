# sensors/ — Context

## Purpose
Simulates the two sensor systems of the extrusion printing rig. All physics is deterministic; noise is reproducible via a fixed random seed.

## Modules

| Module | Class | Description |
|---|---|---|
| `physics.py` | — | Pure physics: `path_deviation`, `energy_per_segment`, `production_rate`; 2 designs × 2 materials |
| `camera.py` | `CameraSystem` | Simulates camera readings; caches `{measured_path, designed_path}` per (layer, segment) |
| `energy.py` | `EnergySensor` | Simulates energy meter; caches `{energy_per_segment}` per (layer, segment); key includes design+water_ratio |
| `fabrication.py` | `FabricationSystem` | Coordinates camera+energy for one experiment or layer-by-layer |

## Key Points
- Physics has 4 distinct optima: (A,clay)≈40, (B,clay)≈33, (A,concrete)≈25, (B,concrete)≈20 mm/s.
- **Shear-thinning coupling**: water optimum shifts with speed — `w_opt_eff = w_opt + ALPHA_WS × (speed − spd_opt) / spd_opt` (ALPHA_WS: clay=0.08, concrete=0.06). Creates a diagonal valley in (speed, water) space.
- **Design-specific curvature**: `segment_curvature(segment_idx, design)` returns a per-design multiplier; design B peaks at segment 2. Affects path deviation magnitude per segment.
- **Energy layer slope**: clay energy decreases with layers (material softens, −0.012 J/layer); concrete increases (cures, +0.022 J/layer). Direction is material-specific.
- **Nozzle-slip production rate**: `production_rate(speed, water, material)` applies a slip floor at low water_ratio — rate collapses below `W_SLIP` threshold (clay=0.45, concrete=0.41) using a logistic penalty. Requires MLP prediction model.
- Layer drift amplifies speed error: `deviation += (LAYER_DRIFT_BASE + LAYER_DRIFT_COUPLING × |speed−spd_opt_layer|) × layer_idx`.
- Optimal speed shifts per layer: clay +0.4 mm/s/layer (softens), concrete −0.55 mm/s/layer (cures).
- `FILAMENT_RADIUS = 0.004` m is used in 3D visualization plots.
- Both sensors cache results by a tuple key of all relevant params + position; re-calls are free.
- `run_experiment()` populates the full cache; `run_layer()` fills one layer for online adaptation.
