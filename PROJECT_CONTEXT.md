# pred-fab-mock — Project Context

## Purpose
Self-contained showcase of the full PFAB journey (baseline → training → exploration → inference) using a simulated robotic extrusion-printing process. Reproduces the ISARC 2026 paper's reference implementation; pinned to `pred-fab@isarc-2026`.

## Structure

| File / Folder | Description |
|---|---|
| `main.py` | Entry point — a ~6-line narrative; all logic lives in `PrintingShowcase` |
| `showcase.py` | `PrintingShowcase` — drives setup → baseline → train → explore → infer → plots; hides all orchestration |
| `analysis.py` | `true_performance_grid()` — the noiseless ground-truth performance landscape + theoretical optimum (for the topology plot) |
| `schema.py` | `build_schema()` → DatasetSchema (params, spatial domain, features, performance) |
| `agent_setup.py` | `build_agent(schema, camera, energy)` → configured PfabAgent |
| `utils.py` | Shared helpers (`params_from_spec`, `get_performance`) |
| `sensors/` | Simulated camera + energy sensors and the deterministic physics |
| `models/` | Feature, evaluation, and prediction model implementations |
| `visualization/` | Plotting (`plots.py`), visual-identity palette/helpers (`_style.py`), console output (`console.py`) |

## Key Points
- **The campaign is a focused 2-parameter calibration**: the design intent (`design`, `material`) is held fixed for the whole run, so the optimization plane (`water_ratio` × `print_speed`) is well covered and the surrogate is well-constrained.
- **Schema features must match the prediction model's outputs** (count + order) — the datamodule builds the target matrix from all schema features, so a feature with no prediction output shifts the columns.
- The fixed spatial-domain axes (`n_layers`, `n_segments`) are pushed into calibration via `agent.update_context_snapshot()` so candidate params (which carry no dimension axes) can size the prediction grid.
- Inference **warm-starts from the model's proposal**, so every inference print is a recommendation clustered on the optimum (parameter-space and timeline figures line up).
- `design` / `material` are used by feature models but **excluded from the prediction model** (DataModule one-hot encodes them; `_filter_batches_for_model` searches by original code name). Carry them over via `current_params` to `exploration_step` / `inference_step`.
- Figures follow the Visual Identity (Zinc/Steel/Emerald, softened RdYlGn) at 600 DPI; see `visualization/_style.py`.
