# dev/ — Context

## Purpose
Development and benchmarking scripts that are NOT part of the main showcase.
Primarily used to evaluate prediction model quality and compare workflows.

## Modules

| Module | Description |
|---|---|
| `eval_exploration.py` | Learning curve benchmark: compares 4 workflows (baseline n=4, explore L-BFGS-B, explore DE, random n=20) on a fixed held-out test set. Produces `dev/plots/learning_curves.png` and `dev/plots/forward_passes.png`. |

## Key Points
- Uses the same schema, agent, sensors, and models as `main.py` — no new code.
- Test set: 32 uniform-grid experiments across all 4 design×material combos (fixed seed=999, never used for training).
- Two metrics: R² (standard) and R²_adj (performance-adjusted, see `Metrics.calculate_adjusted_r2()`). R²_adj down-weights important samples so the gap (R²_adj − R²) reveals whether prediction quality concentrates where it matters for optimisation.
- Forward passes tracked per optimizer: accumulates `agent.last_opt_nfev` after each exploration step.
- Addresses ADVEI research question: "How does data efficiency of exploration-guided sampling compare to grid-search sampling as a function of dataset size?"
