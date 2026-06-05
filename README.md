# pred-fab-mock

> ⚠️ **Frozen snapshot — ISARC 2026.** This branch is preserved exactly as it accompanied the ISARC 2026 paper, for reproducibility, and is **not maintained**. It targets an early version of `pred-fab` and uses older terminology and APIs. For the current version and newer studies, see the [latest version on the default branch](https://github.com/luca-bettermann/pred-fab-mock).

A self-contained showcase of the full [PFAB](../pred-fab) journey using a simulated robotic extrusion printing process (clay / concrete AM).

## What it demonstrates

| Phase | Description |
|---|---|
| 0 — Setup | Schema, sensors, agent, calibration config (design intent fixed: design B, reinforced) |
| 1 — Baseline | 10 space-filling experiments (greedy maximin) |
| 2 — Initial Training | Fit prediction model; plot predicted vs actual |
| 3 — Exploration | 8 UCB rounds with w_explore=0.7; parameter-space plot |
| 4 — Inference | 3 rounds optimising performance; trajectory + before/after path plots |

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
├── main.py               # Full journey Phases 0–5
├── schema.py             # build_schema()
├── agent_setup.py        # build_agent(schema, camera, energy)
├── utils.py              # Shared helpers (params_from_spec, get_performance)
├── sensors/
│   ├── physics.py        # Pure deterministic physics
│   ├── camera.py         # CameraSystem
│   ├── energy.py         # EnergySensor
│   └── fabrication.py    # FabricationSystem (coordinates all sensors)
├── models/
│   ├── feature_models.py    # PrintingFeatureModel, EnergyFeatureModel
│   ├── evaluation_models.py # PathAccuracyModel, EnergyConsumptionModel
│   └── prediction_model.py  # PrintingPredictionModel
└── visualization/
    └── plots.py          # Per-phase plot helpers
```
