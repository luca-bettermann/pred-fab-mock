# pred-fab-mock — Project Context

## Purpose
Self-contained showcase of the full PFAB journey (baseline → exploration → inference → online adaptation) using a simulated robotic extrusion printing process. Built entirely against the local `pred-fab` package.

## Structure

| File / Folder | Description |
|---|---|
| `main.py` | Full journey script — configuration + phase orchestration |
| `workflow.py` | Workflow helpers: `JourneyState`, `run_baseline_phase`, `run_exploration_round`, `run_inference_round`, `run_adaptation_phase` |
| `schema.py` | `build_schema()` → DatasetSchema |
| `agent_setup.py` | `build_agent(schema, camera, energy)` → configured PfabAgent |
| `utils.py` | Small helpers: `params_from_spec`, `get_performance` |
| `sensors/` | Simulated sensor systems (camera, energy) and physics |
| `models/` | Feature, evaluation, and prediction model implementations |
| `visualization/` | Per-phase plotting and console output (plots.py + console.py) |
| `dev/` | Benchmark scripts: `eval_exploration.py` (L-BFGS-B vs DE vs random) |

## Key Points
- Schema v4: 2 designs (A, B) × 2 materials (clay, concrete); 3 features (path_deviation, energy_per_segment, production_rate); 3 performance attributes (path_accuracy, energy_efficiency, production_rate).
- Physics: 4 optima (A,clay≈40, B,clay≈33, A,concrete≈25, B,concrete≈20 mm/s). Layer drift makes deviation grow when speed deviates from the layer-specific optimum — makes adaptation meaningful.
- Calibration weights: path_accuracy=2, energy_efficiency=1, production_rate=1. Combined = (2·acc + eff + rate) / 4. Creates three-way trade-off: higher speed improves production_rate but worsens path_accuracy and energy_efficiency.
- Feature tensors are stored as `(n_layers, n_segments)` 2D arrays — not flat tables.
- Phase 5 (online adaptation) reads deviation directly from the feature tensor without calling `run_evaluation` on partial data, since partial evaluation has NaN rows that break `nanmean`.
- Categorical parameters must be carried over explicitly by passing `current_params` to `exploration_step` / `inference_step`.
- `design` and `material` are included in both MLP prediction models (one-hot encoded by DataModule; recognized by column prefix matching).
- `QUICK_TEST = True` at top of main.py enables fast runs (baseline n=2, exploration rounds=2, inference rounds=1).
