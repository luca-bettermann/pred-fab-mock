# sensors/ — Context

## Purpose
Simulates the two sensor systems of the extrusion printing rig. All physics is deterministic; noise is reproducible via a fixed random seed.

## Modules

| Module | Class | Description |
|---|---|---|
| `physics.py` | — | Pure physics functions: `filament_width`, `path_deviation`, `energy_per_segment` |
| `camera.py` | `CameraSystem` | Simulates camera readings; caches `{width_readings, measured_path, designed_path}` per (layer, segment) |
| `energy.py` | `EnergySensor` | Simulates energy meter; caches `{power_readings}` per (layer, segment) |

## Key Points
- Both sensors cache results by a tuple key of all relevant params + position; re-calls are free.
- `run_experiment()` populates the full cache; `run_layer()` fills one layer for online adaptation.
- Feature models read from these caches via `get_segment_data()` / `get_segment_energy()`.
