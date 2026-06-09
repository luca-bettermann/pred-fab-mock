# pred-fab-mock

> ⚠️ **ISARC 2026 release.** This branch is the reference implementation accompanying the ISARC 2026 paper, pinned to `pred-fab@isarc-2026` for exact reproducibility. It is a frozen release — **not maintained**; for the current version and newer studies see the [default branch](https://github.com/luca-bettermann/pred-fab-mock).

A self-contained showcase of the full [PFAB](../pred-fab) journey using a simulated robotic extrusion printing process (clay / concrete AM).

## What it demonstrates

| Phase | Description |
|---|---|
| 0 — Setup | Schema, sensors, agent, calibration config (design intent fixed: design B, reinforced) |
| 1 — Baseline | 10 space-filling experiments (greedy maximin) |
| 2 — Initial Training | Fit prediction model; plot predicted vs actual |
| 3 — Exploration | 8 UCB rounds with w_explore=0.7; parameter-space plot |
| 4 — Inference | 3 rounds exploiting the model for optimal parameters (warm-started from its proposal) |

The calibration target is the two continuous process parameters (`water_ratio`, `print_speed`); inference converges to the interior optimum and the before/after path plot shows the as-printed deviation collapsing from red to green.

## Simulated process

Each experiment = one print run: **5 layers × 4 segments = 20 evaluation steps**.

- `FabricationSystem` coordinates `CameraSystem` + `EnergySensor`, iterating layer by layer
- `CameraSystem` → `PrintingFeatureModel` → `path_deviation`
- `EnergySensor` → `EnergyFeatureModel` → `energy_per_segment`
- `PathAccuracyModel` + `EnergyConsumptionModel` → `path_accuracy`, `energy_efficiency`
- `PrintingPredictionModel` (sklearn RandomForest) predicts features from process parameters

## Quick start

```bash
cd pred-fab-mock
uv venv
uv sync
uv run python main.py
```

Plots are saved to `./plots/`.

## Repository structure

```
pred-fab-mock/
├── main.py               # Entry point — a short baseline→train→explore→infer→plots narrative
├── showcase.py           # PrintingShowcase — drives the journey; hides all orchestration
├── analysis.py           # True-physics performance landscape + theoretical optimum
├── schema.py             # build_schema()
├── agent_setup.py        # build_agent(schema, camera, energy)
├── utils.py              # Shared helpers (params_from_spec, get_performance)
├── sensors/
│   ├── physics.py        # Pure deterministic physics (tilted, asymmetric response)
│   ├── camera.py         # CameraSystem
│   ├── energy.py         # EnergySensor
│   └── fabrication.py    # FabricationSystem (coordinates all sensors)
├── models/
│   ├── feature_models.py    # PrintingFeatureModel, EnergyFeatureModel
│   ├── evaluation_models.py # PathAccuracyModel, EnergyConsumptionModel
│   └── prediction_model.py  # PrintingPredictionModel
└── visualization/
    ├── plots.py          # Figures (stage prints, parameter-space topology, timeline, …)
    ├── _style.py         # Visual-identity palette, colormaps, rcParams, save helpers
    └── console.py        # Console output helpers
```
