# pred-fab-mock (ADVEI 2026)

A self-contained showcase of the [PFAB](https://github.com/luca-bettermann/pred-fab) predictive fabrication framework using a simulated curved-wall clay extrusion process.

## Quick start

```bash
cd pred-fab-mock
uv venv && uv pip install -e ".[dev]"

# Phase 0: Setup
uv run cli.py reset
uv run cli.py init-schema
uv run cli.py init-agent
uv run cli.py init-physics --plot
uv run cli.py configure \
    --weights '{"structural_integrity":2,"material_deposition":1,"extrusion_stability":1,"energy_footprint":1,"fabrication_time":1}' \
    --schedule print_speed:n_layers --schedule slowdown_factor:n_layers

# Static grids (reference + test)
uv run cli.py grid --dataset-code reference --low-pct 0.25 --high-pct 0.75 \
    --fractional-x 1 --half-face-centers --n-center 1
uv run cli.py test-set --n-center 3

# Phase 1: Baseline (space-filling, no model)
uv run cli.py baseline --n 5 --plot

# Phase 2: Exploration (model-guided active learning)
uv run cli.py explore --n 5 --kappa 0.5 --plot

# Evaluate model quality
uv run cli.py analyse --plot --test-set 20

# Phase 3: Inference (first-time-right manufacturing)
uv run cli.py inference --design-intent '{"layer_height":2.5}' --plot
uv run cli.py report infer_01 --plot

# Summary
uv run cli.py summary
```

## CLI commands

| Command | Description |
|---|---|
| `reset` | Clear all session state, data, and plots |
| `init-schema` | Show the problem schema (parameters, features, performance) |
| `init-agent` | Initialize the agent and show its state |
| `init-physics [--seed N] [--plot]` | Show physics constants and ground truth topology |
| `configure [--weights JSON] [--schedule P:D] [--trust-regions JSON]` | Set agent configuration |
| `baseline --n N [--plot]` | Run N space-filling baseline experiments |
| `grid [--dataset-code S] [--low-pct F] [--high-pct F]` | Run a CCF static-grid dataset |
| `test-set [--n-center N]` | Run ADVEI test dataset (full CCF, 0.15/0.85) |
| `explore --n N [--kappa F] [--plot]` | Run N exploration rounds (incremental) |
| `analyse [--plot] [--test-set N]` | Compare ground truth vs. prediction + optional MAE |
| `inference [--design-intent JSON] [--plot]` | Single-shot first-time-right proposal |
| `report EXP_CODE [--plot]` | Generate visual report for an experiment |
| `summary` | Print run summary across all phases |

## Simulated process

Each experiment = one print run on a curved-wall clay component (25 mm height, 13 layers, 7 nodes per layer).

**Parameters (5):**

| Parameter | Range | Type |
|---|---|---|
| `path_offset` | 0–3 mm | Static (per-print) |
| `layer_height` | 2–3 mm | Static (per-print) |
| `calibration_factor` | 1.6–2.2 | Static (per-print) |
| `print_speed` | 0.004–0.008 m/s | Trajectory (per-layer) |
| `slowdown_factor` | 0–1 | Trajectory (per-layer) |

**Five-way Pareto conflict (3 quality + 2 cost):**
- `structural_integrity` — node overlap at corners
- `material_deposition` — filament width accuracy
- `extrusion_stability` — deposition consistency (R²)
- `energy_footprint` — robot energy per layer
- `fabrication_time` — printing duration per layer

## Models

| Type | Class | Outputs |
|---|---|---|
| Feature | `NodeVisionFeature` | node_overlap, filament_width (depth-2) |
| Feature | `LoadcellConsistencyFeature` | extrusion_consistency |
| Feature | `RobotEnergyFeature` | robot_energy |
| Feature | `DurationFeature` | printing_duration |
| Feature | `EnvironmentFeature` | temperature, humidity (context) |
| Evaluation | `StructuralIntegrityEval` | structural_integrity [0,1] |
| Evaluation | `MaterialDepositionEval` | material_deposition [0,1] |
| Evaluation | `ExtrusionStabilityEval` | extrusion_stability [0,1] |
| Evaluation | `EnergyFootprintEval` | energy_footprint [0,1] |
| Evaluation | `FabricationTimeEval` | fabrication_time [0,1] |
| Prediction | `StructuralTransformer` | 4 features (causal attention over layers) |
| Prediction | `DeterministicDuration` | printing_duration (closed-form) |

## Repository structure

```
pred-fab-mock/
├── cli.py                # Step-by-step CLI (main entry point)
├── cli_helpers.py        # Inline plot display
├── schema.py             # ADVEI 2026 schema definition
├── agent_setup.py        # Agent construction + model registration
├── workflow.py           # Session state, experiment helpers
├── sensors/
│   ├── physics.py        # Deterministic feature-level physics
│   └── fabrication.py    # Coordinates simulation per experiment
├── models/
│   ├── feature_models.py     # 5 feature models (vision, loadcell, energy, duration, env)
│   ├── evaluation_models.py  # 5 evaluators (one per performance attribute)
│   └── prediction_model.py   # StructuralTransformer + DeterministicDuration
├── steps/                # CLI step implementations
├── visualization/        # ADVEI-specific helpers (physics grid evaluation)
└── dev/                  # Diagnostic scripts (01-07)
```
