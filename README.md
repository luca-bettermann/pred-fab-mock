# pred-fab-mock

A self-contained showcase of the full [PFAB](../pred-fab) journey using a simulated robotic extrusion printing process (clay / concrete AM).

## What it demonstrates

| Phase | Description |
|---|---|
| 0 — Setup | Schema, sensors, agent, calibration bounds (water_ratio, print_speed) |
| 1 — Baseline | 4 space-filling experiments (Latin hypercube, no model) |
| 2 — Initial Training | Fit deviation + energy prediction models; validate on held-out data |
| 3 — Exploration | 6 rounds with w_explore=0.7 — model guides search toward uncertain regions |
| 4 — Inference | 3 rounds optimising performance with design intent fixed (design=B, flexible) |
| 5 — Online Adaptation | Layer-by-layer print_speed tuning based on live deviation feedback |

## Simulated process

Each experiment = one print run: **5 layers × 4 segments = 20 evaluation steps**.

**Parameters optimised:** `water_ratio` (0.30–0.50), `print_speed` (20–60 mm/s)
**Fixed per design intent:** `design` (A / B / C), `material` (standard / reinforced / flexible)

**Physics:** path deviation has a U-shaped response to print speed — too slow causes material sag, too fast causes inertia overshoot. The optimal speed varies by design complexity, material viscosity, and water ratio (via flowability). Energy increases monotonically with speed. These two objectives create a genuine trade-off.

**Sensors:**
- `CameraSystem` → `path_deviation`, `filament_width` per (layer, segment)
- `EnergySensor` → `energy_per_segment` per (layer, segment)

**Performance scores (both in [0, 1]):**
- `path_accuracy` — derived from mean path_deviation across the print
- `energy_efficiency` — derived from total energy_per_segment

## Output plots

All plots are saved to `./plots/`. The console prints a one-line description after each.

| File | Phase | What it shows |
|---|---|---|
| `path_comparison.png` | 1 | 1×5 per-layer grid — measured vs designed path, deviation fill, layer drift visible |
| `path_comparison_3d.png` | 1 | 3D tube stack — blue wireframe = designed, solid = as-printed, colour = deviation |
| `filament_volume.png` | 1 | Close-up cylindrical filament — L0 vs L4, blue ghost = designed, arrows = deviation |
| `physics_landscape.png` | 1 | U-shaped deviation vs speed — actual speed vs theoretical optimum |
| `feature_heatmaps.png` | 1 | 5×4 heatmaps of path_deviation and energy_per_segment |
| `prediction_accuracy.png` | 2 | Predicted vs actual scatter with R² for both models |
| `parameter_space.png` | 3 | water_ratio vs speed scatter — score, phase, and design encoded |
| `performance_trajectory.png` | 4 | Score history across all experiments with phase bands |
| `inference_convergence.png` | 4 | Physics score landscape with inference trajectory and optimum star |
| `adaptation.png` | 5 | Adapted speed vs counterfactual, deviation saved shown as fill |

## Quick start

```bash
cd pred-fab-mock
uv venv
uv pip install -e ".[dev]"
python main.py
```

Plots are saved to `./plots/`. The final console output includes a run summary comparing the best found parameters against the physics optimum.

## Repository structure

```
pred-fab-mock/
├── main.py               # Full journey Phases 0–5
├── schema.py             # build_schema()
├── agent_setup.py        # build_agent(schema, camera, energy)
├── utils.py              # Shared helpers (params_from_spec, get_performance)
├── sensors/
│   ├── physics.py        # Pure deterministic physics (U-shaped deviation, energy)
│   ├── camera.py         # CameraSystem — simulates path/width measurements
│   ├── energy.py         # EnergySensor — simulates energy readings
│   └── fabrication.py    # FabricationSystem — coordinates all sensors
├── models/
│   ├── feature_models.py    # PrintingFeatureModel, EnergyFeatureModel
│   ├── evaluation_models.py # PathAccuracyModel, EnergyConsumptionModel
│   └── prediction_model.py  # DeviationPredictionModel, EnergyPredictionModel (sklearn MLP)
└── visualization/
    ├── plots.py          # Per-phase plot helpers (9 functions + tube helper)
    └── console.py        # Terminal pretty-printing (phase banners, tables, summary)
```
