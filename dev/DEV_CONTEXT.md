# dev/ — Context

## Purpose
Development and benchmarking scripts that are NOT part of the main showcase.
Primarily used to evaluate prediction model quality and compare workflows.

## Modules

| Module | Description |
|---|---|
| `eval_exploration.py` | Learning curve benchmark: compares exploration-guided vs. random sampling vs. baseline-only, evaluated on a fixed held-out test set. Produces `dev_plots/learning_curves.png`. |

## Key Points
- Uses the same schema, agent, sensors, and models as `main.py` — no new code.
- Test set: 32 uniform-grid experiments across all 4 design×material combos (fixed seed=999, never used for training).
- Three workflows compared at each training set size: exploration-guided (baseline + N explore rounds), baseline-only (LHS, n=N_BASELINE), random (LHS, n=N_BASELINE+N_EXPLORE).
- R² is computed per feature (path_deviation, energy_per_segment, production_rate) at each step.
- Addresses ADVEI research question: "How does data efficiency of exploration-guided sampling compare to grid-search sampling as a function of dataset size?"
