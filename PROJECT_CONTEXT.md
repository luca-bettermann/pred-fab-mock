# pred-fab-mock — Project Context

## Purpose
Self-contained showcase of the full PFAB journey (baseline → exploration → inference → online adaptation) using a simulated robotic extrusion printing process. Built entirely against the local `pred-fab` package.

## Structure

| File / Folder | Description |
|---|---|
| `main.py` | Full journey script — Phases 0–5 |
| `schema.py` | `build_schema()` → DatasetSchema |
| `agent_setup.py` | `build_agent(schema, camera, energy)` → configured PfabAgent |
| `sensors/` | Simulated sensor systems (camera, energy) and physics |
| `models/` | Feature, evaluation, and prediction model implementations |
| `visualization/` | Per-phase plotting helpers |

## Key Points
- Schema v3: 2 designs (A, B) × 2 materials (clay, concrete); no layer_width feature.
- Physics: 4 optima (A,clay≈40, B,clay≈33, A,concrete≈25, B,concrete≈20 mm/s). Layer drift makes deviation grow when speed deviates from the layer-specific optimum — makes adaptation meaningful.
- Feature tensors are stored as `(n_layers, n_segments)` 2D arrays — not flat tables.
- Phase 5 (online adaptation) reads deviation directly from the feature tensor without calling `run_evaluation` on partial data, since partial evaluation has NaN rows that break `nanmean`.
- Categorical parameters must be carried over explicitly by passing `current_params` to `exploration_step` / `inference_step`.
- `design` and `material` are included in both prediction models (one-hot encoded by DataModule; recognized by column prefix matching).
