# pred-fab-mock

A self-contained showcase of the full [PFAB](../pred-fab) journey using a simulated robotic extrusion printing process (clay / concrete AM).

## What it demonstrates

| Phase | Description |
|---|---|
| 0 — Setup | Schema, simulation, agent, calibration bounds (water_ratio, print_speed) |
| 1 — Baseline | 20 space-filling experiments (Sobol sequence, no model) |
| 2 — Initial Training | Fit deviation + energy + production rate prediction models; validate on held-out data |
| 3 — Exploration | 10 rounds with w_explore=0.7 — model guides search toward uncertain regions |
| 4 — Inference | 3 rounds optimising performance with design intent fixed (e.g. design=A, material=clay) |
| 5 — Online Adaptation | Layer-by-layer print_speed tuning based on live deviation feedback |

## Simulated process

Each experiment = one print run: **5 layers × 4 segments = 20 evaluation steps**.

**Parameters optimised:** `water_ratio` (0.30–0.50), `print_speed` (20–60 mm/s)
**Fixed per design intent:** `design` (A / B), `material` (clay / concrete)

**Physics:** path deviation has a U-shaped response to print speed — too slow causes material sag, too fast causes inertia overshoot. The optimal speed varies by design complexity, material viscosity, and water ratio (via flowability). Energy is U-shaped with minimum near 35 mm/s. These objectives create a genuine three-way trade-off.

**Simulated sensors:**
- `CameraSystem` → `path_deviation` per (layer, segment)
- `EnergySensor` → `energy_per_segment` per (layer, segment)
- `ProductionRateFeatureModel` → `production_rate` (deterministic, from speed and material)

**Performance scores (all in [0, 1]):**
- `path_accuracy` — derived from mean path_deviation across the print
- `energy_efficiency` — derived from total energy_per_segment
- `production_rate` — derived from print speed relative to material capacity

Combined score: (2 · path_accuracy + energy_efficiency + production_rate) / 4.

## Output plots

All plots are saved to `./plots/`. The console prints a one-line description after each.

| File | Phase | What it shows |
|---|---|---|
| `physics_topology.png` | 1 | Ground truth performance landscape across the parameter space |
| `baseline_scatter.png` | 1 | Sobol-sequence coverage visualisation |
| `prediction_accuracy.png` | 2 | Predicted vs actual scatter with R² and R²_adj for each model |
| `explore_*_topology.png` | 3 | Per-round acquisition landscape (performance + uncertainty + combined) |
| `parameter_space.png` | 3 | water_ratio vs speed scatter — score, phase, and design encoded |
| `performance_trajectory.png` | 4 | Score history across all experiments with phase bands |
| `inference_convergence.png` | 4 | Convergence trajectory toward the physics optimum |
| `adaptation.png` | 5 | Adapted speed vs counterfactual, deviation reduction shown as fill |

## Quick start

```bash
cd pred-fab-mock
uv venv
uv pip install -e ".[dev]"
python main.py
```

Plots are saved to `./plots/`. The final console output includes a run summary comparing the best found parameters against the physics optimum.

## CLI

The CLI (`cli.py`) runs each phase as a separate command with JSON session persistence, so you can iterate on individual phases without re-running the entire journey.

```bash
# 1. Configure agent (bounds, weights, optimizer, material)
python cli.py configure \
  --bounds '{"water_ratio":[0.30,0.50],"print_speed":[20.0,60.0]}' \
  --weights '{"path_accuracy":2.0,"energy_efficiency":1.0,"production_rate":1.0}' \
  --material clay --optimizer de --buffer 0.10 0.8 2.0

# 2. Run phases step by step
python cli.py baseline --n 20
python cli.py train --val-size 0.25
python cli.py explore --n 10 --w-explore 0.7
python cli.py infer --n 3 --intent '{"design":"A","material":"clay"}'
python cli.py adapt --start-speed 40.0 --delta '{"print_speed":5.0}'

# 3. Print summary and reset
python cli.py summary
python cli.py reset
```

| Command | Description |
|---|---|
| `configure` | Set bounds, performance weights, optimizer, material, boundary buffer, MPC settings |
| `baseline --n N` | Run N Sobol-sequence baseline experiments |
| `train --val-size F` | Train prediction models with validation split fraction F |
| `explore --n N --w-explore W` | Run N exploration rounds with exploration weight W |
| `infer --n N --intent '{...}'` | Run N inference rounds for a given design intent |
| `adapt --start-speed S --delta '{...}'` | Run online adaptation from starting speed S |
| `summary` | Print final run summary table |
| `reset` | Clear session state, data, plots, and logs |

Session state is stored in `.pfab_session.json` and persists across commands.

## Repository structure

```
pred-fab-mock/
├── main.py               # Full journey Phases 0–5
├── cli.py                # Step-by-step CLI with session persistence
├── schema.py             # build_schema()
├── agent_setup.py        # build_agent(schema, camera, energy)
├── utils.py              # Shared helpers (params_from_spec, get_performance)
├── workflow.py           # JourneyState, run_and_evaluate, with_dimensions
├── reporting.py          # Phase reporters: console output (via agent.console) + plot generation
├── sensors/
│   ├── physics.py        # Deterministic physics simulation (U-shaped deviation, energy, production rate)
│   ├── camera.py         # CameraSystem — simulated path deviation measurements
│   ├── energy.py         # EnergySensor — simulated energy readings
│   └── fabrication.py    # FabricationSystem — coordinates physics + sensors per experiment
├── models/
│   ├── feature_models.py    # PrintingFeatureModel, EnergyFeatureModel, ProductionRateFeatureModel
│   ├── evaluation_models.py # PathAccuracyModel, EnergyConsumptionModel, ProductionRateModel
│   └── prediction_model.py  # MLP and RF prediction models (sklearn)
└── visualization/
    └── plots.py          # Per-phase plot helpers
```
