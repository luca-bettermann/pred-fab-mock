# sensors/ — Context

## Purpose
Simulates the sensor systems of the extrusion printing rig. All physics is deterministic; noise is reproducible via a fixed random seed.

## Setup
Single design (non-linear curvature) × single material (clay). Two continuous parameters: `water_ratio` [0.30–0.50] and `print_speed` [20–60 mm/s]. Fixed spatial domain: 5 layers × 4 segments = 20 evaluation steps per experiment.

## Modules

| Module | Class | Description |
|---|---|---|
| `physics.py` | — | Pure physics: `path_deviation`, `energy_per_segment`, `production_rate` |
| `camera.py` | `CameraSystem` | Simulates camera readings; caches `{measured_path, designed_path}` per (layer, segment) |
| `energy.py` | `EnergySensor` | Simulates energy meter; caches `{energy_per_segment}` per (layer, segment) |
| `fabrication.py` | `FabricationSystem` | Coordinates camera+energy for one experiment or layer-by-layer |

## Key Points
- Physics optimum: speed ≈ 40 mm/s, water ≈ 0.42.
- **Shear-thinning coupling**: water optimum shifts with speed — creates a diagonal valley in (speed, water) space.
- **Non-linear segment curvature**: `[0.85, 1.15, 0.95, 1.05]` — alternating pattern creates segment-dependent deviation behaviour.
- **Pareto conflict**: W_ENERGY_OPT (0.38) ≠ W_OPTIMAL (0.42) — minimising deviation and energy require different water ratios.
- **Smooth adhesion sigmoid**: deviation rises steeply but continuously above ADHESION_SPEED (52 mm/s).
- **Nozzle-slip production rate**: rate collapses above W_SLIP (0.45) using a quadratic penalty.
- Layer drift amplifies speed error with layer index. Optimal speed shifts +0.4 mm/s/layer (clay softens).
- Energy decreases with layers (−0.012 J/layer, clay dries → less pump resistance).
- Both sensors cache results by parameter+position tuple key; re-calls are free.
- `run_experiment()` populates the full cache; `run_layer()` fills one layer for online adaptation.
