# dev/ — Context

## Purpose
Development and benchmarking scripts that are NOT part of the main showcase.
Primarily used to evaluate prediction model quality and compare workflows.

## Modules

| Module | Description |
|---|---|
| `eval_exploration.py` | Learning curve benchmark: compares 4 workflows (baseline n=4, explore L-BFGS-B, explore DE, random n=20) on a fixed held-out test set. Produces `dev_plots/learning_curves.png` and `dev_plots/forward_passes.png`. |

## Key Points
- Uses the same schema, agent, sensors, and models as `main.py` — no new code.
- Test set: 32 uniform-grid experiments across all 4 design×material combos (fixed seed=999, never used for training).
- Two metrics: standard R² and performance-weighted R² (weights each test experiment by its ground-truth system performance score).
- Forward passes tracked per optimizer: accumulates `agent.calibration_system.last_opt_nfev` after each exploration step.
- Addresses ADVEI research question: "How does data efficiency of exploration-guided sampling compare to grid-search sampling as a function of dataset size?"
