# Handoff: pred-fab-mock

## Repositories

| Repo | Path | Latest commit |
|------|------|---------------|
| `pred-fab` (core library) | `/home/claude-user/projects/pred-fab/pred-fab` | `bee150e` |
| `pred-fab-mock` (this repo) | `/home/claude-user/projects/pred-fab/pred-fab-mock` | `fec8ec8` |

Always read `/home/claude-user/projects/pred-fab/CLAUDE.md` before starting.

---

## What the mock does

Five-phase showcase of the full PFAB journey using a simulated robotic extrusion printing process (5 layers × 4 segments per experiment):

| Phase | Description |
|-------|-------------|
| 0 | Setup: schema, sensors, agent, calibration config |
| 1 | Baseline: 4 LHS experiments |
| 2 | Initial training + prediction accuracy plot |
| 3 | 6 exploration rounds (`w_explore=0.7`) |
| 4 | 3 inference rounds (`w_explore=0.0`, fixed `design=B, material=flexible`) |
| 5 | Online adaptation: layer-by-layer `print_speed` tuning |

**Sensors:** `CameraSystem` (geometry) · `EnergySensor` (power draw)
**Features:** `path_deviation`, `layer_width` (camera) · `energy_per_segment` (energy)
**Evaluation:** `PathAccuracyModel` · `EnergyConsumptionModel`
**Prediction:** two sklearn `MultiOutputRegressor(RandomForestRegressor)` models

Run: `.venv/bin/python main.py` from the mock root.

---

## Current state

The full pipeline runs end-to-end (exit 0). All workarounds removed — everything goes through core PFAB paths. Exploration shows genuine diversity (designs A/C, all materials, speeds 20–55). Online adaptation correctly reduces speed each layer to counteract layer drift.

---

## Known issues to address

### 1. Inference returns identical proposals every round
All 3 inference rounds propose the same parameters (`spd=20.0, w=0.47`). With `w_explore=0.0`, L-BFGS-B finds the performance optimum on round 1 and returns it unchanged. The showcase arc should show convergent refinement, not a flat line. Root cause likely in how inference-mode calibration interacts with a frozen model — needs investigation at core level.

### 2. Exploration last two rounds are identical
Rounds 5 and 6 propose the same point. The agent converges before exhausting the exploration budget.

### 3. `water_ratio` stuck at upper bound during exploration
Every exploration proposal returns `w=0.47` (the max). With only 4 baseline points, the RF has learned a spurious monotonic relationship. The evidence model should push away from visited `w=0.47` regions but isn't overcoming the performance gradient.

---

## Key architectural context

**Evidence model bandwidth** — `pred-fab/src/pred_fab/orchestration/prediction.py`:

```
h = c · √d / √N
γ = max(1, c · √N)
```

`c=0.5` (exploration radius), `d` = active latent dims, `N` = training experiments.
The `√d` scaling was added to prevent flat uncertainty landscapes in high-dimensional spaces. Without it, in 6D with N=4 the bubbles covered ~0.024% of space and the acquisition gradient was zero everywhere.

**Firm rule**: always fix at the PFAB core level (`pred-fab/`), not with mock-specific workarounds. The mock is a showcase; pred-fab is the product.

---

## Working approach

Work iteratively: run `main.py`, inspect the terminal output and the plots saved to `./plots/`, identify what looks wrong, then fix. Do not make multiple changes blindly — one change at a time, re-run, look at results. The plots and terminal output are the ground truth for whether something is working.

---

## Startup checklist

```bash
cd /home/claude-user/projects/pred-fab/pred-fab      && git pull
cd /home/claude-user/projects/pred-fab/pred-fab-mock && git pull
cd /home/claude-user/projects/pred-fab/knowledge-base && git pull

# Run to see current state
cd /home/claude-user/projects/pred-fab/pred-fab-mock && .venv/bin/python main.py
```
