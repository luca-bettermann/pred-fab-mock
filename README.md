# pred-fab-mock

A self-contained showcase of the full [PFAB](../pred-fab) journey using a simulated robotic extrusion printing process (clay / concrete AM).

## What it demonstrates

| Phase | Description |
|---|---|
| 0 — Setup | Schema, sensors, agent, calibration config |
| 1 — Baseline | 8 space-filling experiments (greedy maximin) |
| 2 — Initial Training | Fit prediction model; plot predicted vs actual |
| 3 — Exploration | 4 UCB rounds with w_explore=0.7; parameter space plot |
| 4 — Inference | 3 rounds optimising performance; trajectory plot |
| 5 — Online Adaptation | Layer-by-layer print_speed tuning; adaptation plot |

## Simulated process

Each experiment = one print run: **5 layers × 4 segments = 20 evaluation steps**.

- `CameraSystem` → `PrintingFeatureModel` → `layer_width`, `path_deviation`
- `EnergySensor` → `EnergyFeatureModel` → `energy_per_segment`
- `PathAccuracyModel` + `EnergyConsumptionModel` → `path_accuracy`, `energy_efficiency`
- `PrintingPredictionModel` (sklearn RandomForest) predicts features from continuous params

## Quick start

```bash
cd pred-fab-mock
uv venv
uv pip install matplotlib numpy scikit-learn
uv pip install -e "../pred-fab"
python main.py
```

Plots are saved to `./plots/`.

## Repository structure

```
pred-fab-mock/
├── main.py               # Full journey Phases 0–5
├── schema.py             # build_schema()
├── agent_setup.py        # build_agent(schema, camera, energy)
├── sensors/
│   ├── physics.py        # Pure deterministic physics
│   ├── camera.py         # CameraSystem
│   └── energy.py         # EnergySensor
├── models/
│   ├── feature_models.py    # PrintingFeatureModel, EnergyFeatureModel
│   ├── evaluation_models.py # PathAccuracyModel, EnergyConsumptionModel
│   └── prediction_model.py  # PrintingPredictionModel
└── visualization/
    └── plots.py          # Per-phase plot helpers
```
