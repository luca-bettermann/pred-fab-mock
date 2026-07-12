# dev/ — Context

## Purpose
Progressive validation scripts, numbered in workflow order; run standalone from the repo root. Several target a pre-rename pred-fab API and are broken against the installed version — rehab is tracked on the board; the Status column is per-script.

| Script | Purpose | Status |
|---|---|---|
| `shared.py` | Common env/train/test-grid helpers for all scripts | partly current — `make_env`, `train_models`, `build_test_grid` work; `run_baseline` calls `agent.baseline_step`, which the installed pred-fab no longer exposes |
| `01_physics.py` | Ground-truth topology validation | stale — imports `plot_physics_topology` / `plot_cross_sections`, not in `visualization/` |
| `02_baseline.py` | Baseline sampling coverage check | stale — imports removed `plot_baseline_scatter`; depends on broken `shared.run_baseline` |
| `03_prediction.py` | Prediction-model topology sweep (per-feature R²) | stale — imports removed `DevMLP` (models expose `DevTransformer`) and `plot_topology_comparison` |
| `04_uncertainty.py` | Evidence-uncertainty sanity check | stale — calls `agent.predict_uncertainty`, which no longer exists |
| `06_trajectory.py` | Trajectory vs fixed-parameter comparison | stale — imports removed `plot_trajectory_comparison` |
| `07_inference.py` | Single-shot inference vs physics optimum | stale — imports removed `plot_inference_convergence` |
| `smoke_dataset_code.py` | `dataset_code` round-trip smoke test | current — imports resolve against the installed pred-fab |
| `_smoke_layer3.py` | Low-RAM exploration smoke test | stale — imports removed `DevMLP` |

## Key Points
- `data/` and `plots/` outputs are generated and gitignored.
- Status reflects static import/API checks against the installed pred-fab, not full runs.
