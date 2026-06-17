# pred-fab-mock (ADVEI 2026) — Project Context

## Purpose
Public, runnable simulation companion to the ADVEI 2026 PFAB study. Drives the full pred-fab active-learning loop (discovery → exploration → inference) against a **feature-level fabrication simulator** instead of real hardware. Mirrors `learning-by-printing` above the sensor layer — identical schema, evaluation models, and prediction model — so the workflow and algorithm reproduce without the lab.

## Layout

| Path | Role |
|------|------|
| `models/schema.py` | Canonical ADVEI `DatasetSchema` + all codes/constants (clean, non-rtde). 4 params, 10 features, 5 performance attrs, one layer×node domain (n_layers 9–13). |
| `models/evaluations/` | The 5 `IEvaluationModel`s — copied verbatim from lbp. Raised-cosine (`node_quality.py`: structural_integrity, material_deposition); linear cost-style (`extrusion_stability`, `energy_footprint`, `fabrication_time`, the last two with `aggregate_input="sum"`). `base.py`/`_common.py` hold shared scoring. |
| `models/predictions/structural_mlp.py` | `StructuralMLP` — lbp's MLP (hidden (32,16), 500 epochs); predicts the 4 genuinely-uncertain depth-1 features (loadcell_residual, robot_energy, node_overlap_mean, filament_width_mean). Only change vs lbp: clean param codes (no `to_rtde`). |
| `models/predictions/deterministic_duration.py` | `DeterministicDuration` — closed-form `printing_duration = path_length / print_speed`; kept out of the MLP so its large magnitude doesn't swamp the shared multi-output loss. |
| `models/features/` | 6 `IFeatureModel`s that route `FabricationSystem` output to schema feature codes (vision/aggregate/loadcell/energy/duration/environment). |
| `fabrication/physics.py` | Synthetic feature functions — smooth, deterministic, tuned to lbp's ranges/targets and Pareto-rich. |
| `fabrication/system.py` | `FabricationSystem` — caches per-(layer,node) + per-layer values; computes the per-layer `*_mean` aggregates internally. |
| `cli/agent_setup.py` | `build_schema` / `build_fab` / `build_agent` — registers models, wires the simulator as the feature source, sets `dimension_derivations` for variable n_layers. |
| `cli/session.py` | Config persistence (`local/session.json`) + env rebuild + the simulate→evaluate→save helper. |
| `cli/commands.py` / `cli/main.py` | Step implementations + argparse entry point. |
| `scripts/smoke.py` | End-to-end dev check (discovery→train→exploration→inference). |

## Key points
- **No hardware / no NocoDB / no trajectory.** Per-print parameters only (trajectory scheduling is out of ADVEI scope). The simulator stands in for lbp's entire sensor + extraction layer.
- **Single canonical schema** in `models/schema.py` — every code/constant referenced from there; the simulator and evals share naming.
- **Persistence**: experiments (features + performance) persist via pred-fab `LocalData` under `local/`; each CLI command rebuilds the agent and reloads prior experiments from disk.
- **Validated**: the loop runs against pred-fab@`advei-2026`. At ~24 discovery experiments all five predicted features reach R² ≥ 0.82 (duration exact); at the leaner ~18 default `loadcell_residual` / `robot_energy` are still converging (the two noisiest signals) while the rest are already strong.
- **Inference is first-time-right**: `inference` fabricates and scores its κ=0 proposal, reporting predicted vs measured performance and the gap to the best experiment seen — not just a parameter print.
- **`report` renders the journey plot** (`plots.journey`): combined score vs experiment count across the active loop against the passive CCF reference-grid best — the headline "fewer experiments" figure.
- **pred-fab consumption**: dep follows the branch — this ADVEI branch pins pred-fab tag `advei-2026` (frozen for reproduction); `main` tracks pred-fab `main`. Source of truth: `[tool.uv.sources]` in `pyproject.toml`.

## Dev note
This box's system Python 3.12 lacks `_ctypes`; use a uv-managed interpreter (`UV_PYTHON_PREFERENCE=only-managed uv venv`). Readers with a normal Python are unaffected.
