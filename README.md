# pred-fab-mock

A self-contained showcase of the [PFAB](https://github.com/luca-bettermann/pred-fab) predictive fabrication framework using a simulated robotic extrusion printing process.

## Quick start

```bash
cd pred-fab-mock
uv venv && uv pip install -e ".[dev]"

# Phase 0: Setup
uv run cli.py reset
uv run cli.py init-schema
uv run cli.py init-agent
uv run cli.py configure --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'
uv run cli.py init-physics --seed 42 --plot

# Phase 1: Baseline (space-filling, no model)
uv run cli.py baseline --n 10 --plot

# Phase 2: Exploration (model-guided active learning)
uv run cli.py explore --n 1 --kappa 0.5 --plot    # one round, inspect
uv run cli.py explore --n 9 --kappa 0.5            # continue to 10 total

# Evaluate model quality
uv run cli.py test-set --n 20
uv run cli.py analyse --plot

# Phase 3: Inference (first-time-right manufacturing)
uv run cli.py inference --design-intent '{"n_layers":5}' --plot

# Summary
uv run cli.py summary
```

## CLI commands

| Command | Description |
|---|---|
| `reset` | Clear all session state, data, and plots |
| `init-schema` | Show the problem schema (parameters, features, performance) |
| `init-agent [--model {mlp,rf}]` | Initialize the agent and show its state |
| `init-physics [--seed N] [--plot]` | Randomize physics constants and show topology |
| `configure [--weights JSON] [--radius F] [--optimizer {de,lbfgsb}]` | Set agent configuration |
| `baseline --n N [--plot]` | Run N space-filling baseline experiments |
| `explore --n N [--kappa F] [--plot] [--validate]` | Run N exploration rounds (incremental) |
| `test-set --n N` | Create held-out test experiments for model evaluation |
| `analyse [--plot]` | Evaluate model on test set + sensitivity analysis |
| `inference [--design-intent JSON] [--plot]` | Single-shot first-time-right proposal |
| `summary` | Print run summary across all phases |

All commands support `--plot` to display plots. Plots are always saved to `./plots/`.

> **Inline plots** require [iTerm2](https://iterm2.com/) (`brew install --cask iterm2`). In other terminals, `--plot` saves the plot to disk without displaying it.

### Advanced commands

| Command | Description |
|---|---|
| `explore-trajectory --n N [--delta F] [--smoothing F] [--lookahead N]` | Per-layer speed optimization with MPC |
| `adapt [--delta F] [--design-intent JSON]` | Inference + layer-by-layer online adaptation |

```bash
# Trajectory exploration: optimize print_speed per layer
uv run cli.py explore-trajectory --n 3 --kappa 0.5 --delta 5.0 --smoothing 0.25

# Online adaptation: inference + real-time layer-by-layer tuning
uv run cli.py adapt --delta 5.0 --design-intent '{"n_layers":5}'
```

## Simulated process

Each experiment = one print run with variable layers (3-8) and 4 segments per layer.

**Parameters:** `water_ratio` [0.30, 0.50], `print_speed` [20, 60] mm/s, `n_layers` [3, 8]

**Three-way Pareto conflict:**
- `path_accuracy` — U-shaped speed response (sag vs inertia)
- `energy_efficiency` — different optimum than path accuracy
- `production_rate` — favors high speed, penalized by nozzle slip

Combined score: `(2 * path_accuracy + energy_efficiency + production_rate) / 4`

## Models

| Type | Class | Outputs |
|---|---|---|
| Feature | `DevFeature` | path_deviation |
| Feature | `EnergyFeature` | energy_per_segment |
| Feature | `RateFeature` | production_rate |
| Evaluation | `PathAccuracy` | path_accuracy [0,1] |
| Evaluation | `EnergyEfficiency` | energy_efficiency [0,1] |
| Evaluation | `ProductionRate` | production_rate [0,1] |
| Prediction | `DevMLP` | path_deviation (sklearn MLP) |
| Prediction | `EnergyMLP` | energy_per_segment (sklearn MLP) |
| Prediction | `RateMLP` | production_rate (deterministic) |

## Repository structure

```
pred-fab-mock/
├── cli.py                # Step-by-step CLI (main entry point)
├── cli_helpers.py        # Plot display, physics randomization, sensitivity analysis
├── main.py               # Legacy full journey script
├── schema.py             # Schema definition
├── agent_setup.py        # Agent construction
├── workflow.py           # Session state, experiment helpers
├── sensors/
│   ├── physics.py        # Deterministic physics simulation
│   ├── camera.py         # Simulated camera system
│   ├── energy.py         # Simulated energy sensor
│   └── fabrication.py    # Coordinates sensors per experiment
├── models/
│   ├── feature_models.py     # DevFeature, EnergyFeature, RateFeature
│   ├── evaluation_models.py  # PathAccuracy, EnergyEfficiency, ProductionRate
│   └── prediction_model.py   # DevMLP, EnergyMLP, RateMLP (+ RF variants)
├── visualization/            # Plot functions (topology, scatter, acquisition, sensitivity)
└── dev/                      # Diagnostic scripts for individual pipeline stages
```
