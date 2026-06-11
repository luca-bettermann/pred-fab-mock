# pred-fab-mock — Project Context

## Purpose
Self-contained showcase of the full PFAB journey (baseline → exploration → inference) using a simulated extrusion printing process.

**Owns:** the simulated process (sensors + physics), mock-specific feature/evaluation/prediction models, the demo CLI and its plots.
**Out of scope → who:** generic plotting, agent/calibration/optimizer logic → `pred-fab` (pred-fab agent); changes there go via handoff cards.
**Depends on:** `pred-fab` git dep pinned `@main` (see [[Repo Dependency Graph]]).

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
| `models/` | Feature models, linear target/scale evaluation models, prediction models (DevTransformer, EnergyMLP, deterministic RateMLP) |
| `visualization/` | Mock-specific plots only: physics grid helpers + 3D filament view. Generic plots live in `pred_fab.plotting` |
| `dev/` | Progressive validation scripts: 01_physics → 07_inference |

## Key Points
- Schema v9: single design (non-linear curvature) × single material (clay). 2 continuous parameters (water_ratio [0.30-0.50], print_speed [20-60 mm/s]). Spatial domain: n_layers design-intent [4..8] × 4 segments; the journey pins n_layers=5 via fixed params.
- Physics optimum: speed ≈ 40 mm/s, water ≈ 0.42. Pareto conflict between path_accuracy, energy_efficiency, and production_rate. `init-physics` randomizes constants per session.
- Calibration weights: path_accuracy=2, energy_efficiency=1, production_rate=1.
- Optimizer (pred-fab main): Sobol candidates + multi-start local optimizer (`configure_optimizer(n_starts, n_sobol, lr)`); evidence-based acquisition (κ-blend of performance and evidence gain) replaced the old DE + KDE-uncertainty path.
- Trajectory mode: virtual KDE points for within-trajectory spacing + smoothing penalty for monotonic trajectories.
- Inference is single-shot (first-time-right manufacturing), not iterative.

## Open Risks
- `dev/` scripts still target the pre-rename pred-fab API (`baseline_step`, `predict_uncertainty`, removed plot functions) — broken against main.
- `adapt` crashes upstream: `pred_system.tune()` uses `forward_pass`, unsupported by `TransformerModel` multi-depth outputs.
- Acquisition over variable `n_layers` crashes upstream (TransformerModel lacks variable-length sequences) — all acquisition paths must pin `n_layers` until that lands.
