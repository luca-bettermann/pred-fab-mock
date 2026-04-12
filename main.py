"""Full PFAB journey: Baseline → Training → Exploration → Inference.

Demonstrates the functionality and capabilities of pred-fab in a minimal
but realistic extrusion printing scenario (single design, single material).

Configure the parameters below, then run: python main.py
"""

import numpy as np

from pred_fab.core import Dataset
from pred_fab import combined_score

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from workflow import (
    JourneyState, clean_artifacts, with_dimensions,
    run_and_evaluate, get_physics_optimum,
)
from utils import params_from_spec

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

QUICK_TEST = False

# Experiment counts
N_BASELINE   = 2  if QUICK_TEST else 20
N_EXPLORE    = 1  if QUICK_TEST else 10
N_INFER      = 1  if QUICK_TEST else 1       # single-shot inference

# Agent configuration (bounds default to schema min/max)
PERFORMANCE_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}
EXPLORATION_RADIUS  = 0.5
BOUNDARY_BUFFER     = (0.10, 0.8, 2.0)

# Exploration
KAPPA = 0.7

# DE optimizer
DE_MAXITER = 100
DE_POPSIZE = 10

PLOT_DIR = "./plots"


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _score_color(v: float) -> str:
    if v >= 0.70: return "\033[32m"
    if v >= 0.45: return "\033[33m"
    return "\033[31m"

def _perf_str(perf: dict, keys: list[str]) -> str:
    parts = []
    for k in keys:
        v = perf.get(k, 0.0)
        short = k[:3]
        parts.append(f"{short}={_score_color(v)}{v:.3f}\033[0m")
    return "  ".join(parts)

def _combined(perf: dict) -> float:
    return combined_score(perf, PERFORMANCE_WEIGHTS)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import os, shutil
    for d in ["./local", PLOT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(PLOT_DIR, exist_ok=True)

    state = JourneyState()
    fab   = FabricationSystem(CameraSystem(), EnergySensor())
    perf_keys = ["path_accuracy", "energy_efficiency", "production_rate"]

    # ── Phase 0: Setup ───────────────────────────────────────────────────────
    schema  = build_schema()
    agent   = build_agent(schema, fab.camera, fab.energy)
    dataset = Dataset(schema=schema)

    agent.configure(
        performance_weights=PERFORMANCE_WEIGHTS,
        exploration_radius=EXPLORATION_RADIUS,
        boundary_buffer=BOUNDARY_BUFFER,
        de_maxiter=DE_MAXITER,
        de_popsize=DE_POPSIZE,
    )

    print()

    # ── Phase 1: Baseline ────────────────────────────────────────────────────
    agent.console.print_phase_header(1, "Baseline",
                       f"{N_BASELINE} Sobol experiments — space-filling, no model")

    specs = agent.baseline_step(n=N_BASELINE)
    for i, spec in enumerate(specs):
        params = with_dimensions(params_from_spec(spec))
        exp_code = f"baseline_{i+1:02d}"
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = {k: float(v) for k, v in exp_data.performance.get_values_dict().items() if v is not None}
        state.record("baseline", exp_code, params, perf)

    best_idx = max(range(len(state.perf_history)),
                   key=lambda i: _combined(state.perf_history[i][1]))
    best_code = state.all_codes[best_idx]
    best_perf = state.perf_history[best_idx][1]
    print(f"  Best: \033[1m{best_code}\033[0m  {_perf_str(best_perf, perf_keys)}  "
          f"combined={_combined(best_perf):.3f}")

    # ── Phase 2: Training ────────────────────────────────────────────────────
    agent.console.print_phase_header(2, "Training",
                       "Fit prediction models on baseline data")

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    results = agent.train(datamodule, validate=True)

    # ── Phase 3: Exploration ─────────────────────────────────────────────────
    agent.console.print_phase_header(3, "Exploration",
                       f"{N_EXPLORE} rounds (kappa={KAPPA})")

    prev_params = with_dimensions(params_from_spec(specs[-1]))
    for i in range(N_EXPLORE):
        spec = agent.exploration_step(datamodule, kappa=KAPPA)
        proposed = params_from_spec(spec)
        params = with_dimensions({**prev_params, **proposed})
        exp_code = f"explore_{i+1:02d}"

        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = {k: float(v) for k, v in exp_data.performance.get_values_dict().items() if v is not None}
        state.record("exploration", exp_code, params, perf)
        prev_params = params

        u = agent.predict_uncertainty(params, datamodule)
        print(f"  {exp_code}  w={params['water_ratio']:.3f}  spd={params['print_speed']:.1f}  "
              f"{_perf_str(perf, perf_keys)}  u={u:.3f}")

        datamodule.update()
        agent.train(datamodule, validate=False)

    # ── Phase 4: Inference (single-shot) ─────────────────────────────────────
    agent.console.print_phase_header(4, "Inference",
                       "Single-shot first-time-right manufacturing")

    # Agent proposes the best parameters given current model
    spec = agent.exploration_step(datamodule, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**prev_params, **proposed})
    exp_code = "infer_01"

    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
    perf = {k: float(v) for k, v in exp_data.performance.get_values_dict().items() if v is not None}
    state.record("inference", exp_code, params, perf)

    print(f"  {exp_code}  w={params['water_ratio']:.3f}  spd={params['print_speed']:.1f}  "
          f"{_perf_str(perf, perf_keys)}  combined={_combined(perf):.3f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    spd_opt, w_opt = get_physics_optimum()
    print(f"\n  \033[2m{'─' * 58}\033[0m")
    print(f"  Physics optimum:   speed={spd_opt:.1f} mm/s, water={w_opt:.2f}")
    print(f"  Inference result:  speed={params['print_speed']:.1f} mm/s, "
          f"water={params['water_ratio']:.2f}  "
          f"combined={_combined(perf):.3f}")
    print(f"  \033[2m{'─' * 58}\033[0m")


if __name__ == "__main__":
    main()
