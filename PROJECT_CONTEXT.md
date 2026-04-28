# pred-fab-mock — Project Context

## Purpose
Self-contained showcase of the full PFAB journey (baseline → exploration → inference) using a simulated extrusion printing process. Built against the local `pred-fab` package.

## Structure

| File / Folder | Description |
|---|---|
| `main.py` | End-to-end workflow: baseline → training → exploration → single-shot inference |
| `cli.py` | Step-by-step CLI with JSON session persistence |
| `workflow.py` | Workflow helpers: `JourneyState`, `run_and_evaluate`, `with_dimensions` |
| `schema.py` | `build_schema()` → DatasetSchema (water_ratio + print_speed) |
| `agent_setup.py` | `build_agent()` → configured PfabAgent |
| `utils.py` | Small helpers: `params_from_spec`, `get_performance` |
| `sensors/` | Simulated sensor systems (camera, energy) and physics engine |
| `models/` | Feature, evaluation, and prediction model implementations (MLP + RF) |
| `visualization/` | Shared plot functions split by category (physics, prediction, exploration, inference, trajectory) |
| `dev/` | Progressive validation scripts: 01_physics → 07_inference |

## Key Points
- Schema v7: single design (non-linear curvature) × single material (clay). 2 continuous parameters (water_ratio [0.30-0.50], print_speed [20-60 mm/s]). Spatial domain: 5 layers × 4 segments.
- Physics optimum: speed ≈ 40 mm/s, water ≈ 0.42. Pareto conflict between path_accuracy, energy_efficiency, and production_rate.
- Calibration weights: path_accuracy=2, energy_efficiency=1, production_rate=1.
- Default optimizer: DE (global, with L-BFGS-B polish) for offline; L-BFGS-B for online adaptation.
- Trajectory mode: virtual KDE points for within-trajectory spacing + smoothing penalty for monotonic trajectories.
- Inference is single-shot (first-time-right manufacturing), not iterative.
