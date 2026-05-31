# pred-fab-mock — ADVEI 2026

A self-contained, runnable companion to the **ADVEI 2026** Predictive Fabrication study. It drives the full [PFAB](../pred-fab) active-learning loop against a **synthetic fabrication simulator** — no hardware, no databases — so readers and reviewers can reproduce the *workflow* and inspect the *algorithm* without the physical lab.

It mirrors the real study ([`learning-by-printing`](../learning-by-printing)) exactly above the sensor layer: the **same schema, the same five evaluation models, and the same `StructuralMLP`**. Only the lowest layer differs — where the real study extracts features from cameras, loadcells and the robot, the mock synthesises them directly from the process parameters.

## Quick start

```bash
uv venv
uv sync                                   # installs pred-fab + torch (CPU)

uv run python -m cli.main discovery --n 18   # κ=1: space-filling seed experiments
uv run python -m cli.main train              # fit StructuralMLP on what's collected
uv run python -m cli.main exploration        # 0<κ<1: blend performance + evidence
uv run python -m cli.main inference          # κ=0: predicted-optimal parameters
uv run python -m cli.main summary            # session status
```

Each command persists its results, so the steps compose across invocations. `configure --kappa 0.4 --seed 1` sets defaults; `reset` clears the session.

## What it demonstrates

The κ-blend acquisition `A = (1−κ)·S + κ·ΔE`, swept across the three study phases:

| Phase | κ | What it does |
|-------|---|--------------|
| Discovery | 1 | pure evidence maximisation — space-fill the parameter box before any model exists |
| Exploration | 0 < κ < 1 | balance predicted performance against evidence gain |
| Inference | 0 | propose the performance-optimal parameters from the trained model |

## The simulated process

Curved-wall clay extrusion. Four process parameters drive five quality/cost performance attributes through ten features:

- **Parameters:** `path_offset` [1–3 mm], `layer_height` [2–3 mm] (sets layer count, 9–13), `calibration_factor` [1.8–2.2], `print_speed` [0.05–0.1 m/s].
- **Features:** per-(layer, node) `node_overlap`, `filament_width`; per-layer `loadcell_residual`, `robot_energy`, `printing_duration` + their node means; context `temperature`, `humidity`, `material_age`.
- **Performance:** `structural_integrity`, `material_deposition` (raised-cosine on the 7 mm target), `extrusion_stability`, `energy_footprint`, `fabrication_time` (cost-style, lower is better).

The synthetic physics is tuned to be **Pareto-rich**: every parameter trades one attribute against another (fast printing is quick but energy-hungry and less stable; high calibration widens the bead toward target but over-extrudes), so the acquisition loop has a genuine trade-off surface to navigate.

## Structure

See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md). In short: `models/` (schema + evals + MLP, shared with the real study), `fabrication/` (the synthetic simulator), `cli/` (the workflow driver), `scripts/smoke.py` (end-to-end dev check).

## pred-fab dependency

During development the `pred-fab` dependency is an editable path to the sibling checkout. Published study releases pin it to the matching tag (e.g. `pred-fab @ git+…@advei-2026`), per [the repo strategy](../knowledge-base).
