# pred-fab-mock — Project Context

## Purpose
Self-contained ADVEI 2026 showcase of the full PFAB journey (baseline → exploration → inference) using a simulated curved-wall clay extrusion process. Built against the local `pred-fab` package.

## Structure

| File / Folder | Description |
|---|---|
| `cli.py` | Step-by-step CLI with JSON session persistence |
| `cli_helpers.py` | Inline plot display (iTerm2 protocol) |
| `schema.py` | `build_schema()` → DatasetSchema (5 params, 7 features, 5 perf attributes) |
| `agent_setup.py` | `build_agent()` → configured PfabAgent |
| `workflow.py` | `JourneyState`, `run_and_evaluate`, `with_dimensions` |
| `utils.py` | Small helpers: `params_from_spec`, `get_performance` |
| `sensors/` | Feature-level physics engine + FabricationSystem |
| `models/` | Feature, evaluation, and prediction model implementations |
| `steps/` | CLI step modules (14 subcommands) |
| `visualization/` | Thin ADVEI-specific helpers (physics grid evaluation) |
| `dev/` | Progressive validation scripts: 01_physics → 07_inference |

## Key Points
- Schema: single design (curved wall, 25mm height) × single material (clay). 5 continuous parameters (3 static, 2 trajectory). Structural domain: 13 layers × 7 nodes.
- Physics: deterministic feature-level simulation (no raw sensors). Pareto-rich across 5 performance attributes.
- Features: node_overlap, filament_width (depth-2); extrusion_consistency, robot_energy, printing_duration (depth-1); temperature, humidity (context).
- Prediction: StructuralTransformer (causal attention, multi-depth) + DeterministicDuration (closed-form).
- All generic plots imported from `pred_fab.plotting`; visualization/ only has domain-specific helpers.
- CLI mirrors the main branch pattern: quick-start epilog, --plot flags, RawDescriptionHelpFormatter.
