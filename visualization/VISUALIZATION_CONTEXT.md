# visualization/ — Context

## Purpose
Mock-specific plots only — generic PFAB plots live in `pred_fab.plotting`. Four near-independent modules; the only internal tie is `helpers.save_fig`, so there is no flow diagram to draw.

| Module | Description |
|---|---|
| `__init__.py` | Public re-exports of the plotting API |
| `helpers.py` | Ground-truth physics grid: `physics_combined_at`, `evaluate_physics_grid`, `get_physics_optimum`; re-exports `save_fig` |
| `journey.py` | `plot_journey` — combined score vs experiment count across all phases |
| `process.py` | `plot_path_comparison_3d` — 3D filament tube view (requires a live CameraSystem cache) |

## Key Points
- `helpers.py` owns the ground-truth grid and `get_physics_optimum`, which returns **(water_ratio, print_speed)** — keep that order at call sites.
- Grid aggregation mirrors the evaluation pipeline (`_LinearTargetScore`): per-segment scores are clipped to [0, 1] first, then averaged.
- Physics constants are read live, so all grids honour per-session randomization (`init-physics`).
- All modules force the `Agg` matplotlib backend — safe on headless hosts.
