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
- `design` / `material` categorical parameters are used by feature models but **excluded from the prediction model** because DataModule one-hot encodes them (`design_A`, etc.) and `_filter_batches_for_model` searches by original code name.
- Feature tensors are stored as `(n_layers, n_segments)` 2D arrays — not flat tables.
- Phase 5 (online adaptation) reads deviation directly from the feature tensor without calling `run_evaluation` on partial data, since partial evaluation has NaN rows that break `nanmean`.
- Categorical parameters must be carried over explicitly by passing `current_params` to `exploration_step` / `inference_step`.
