"""Full PFAB journey showcase for simulated robotic extrusion printing.

Phases:
  0 — Setup
  1 — Baseline (8 experiments)
  2 — Initial Training
  3 — Exploration (4 rounds)
  4 — Inference (3 rounds, with design intent fixed)
"""

import os
import shutil
from typing import Any, Dict, List, Tuple

from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec, get_performance
from visualization import (
    plot_path_comparison,
    plot_feature_heatmaps,
    plot_prediction_accuracy,
    plot_parameter_space,
    plot_performance_trajectory,
    print_phase_header,
    print_section,
    print_experiment_row,
    print_phase_summary,
    print_done,
)


def _run_and_evaluate(
    dataset: Dataset,
    agent: Any,
    fab: FabricationSystem,
    params: Dict[str, Any],
    exp_code: str,
) -> Any:
    """Create experiment, run fabrication, evaluate features + performance, persist to disk."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    return exp_data


def _with_dimensions(params: Dict[str, Any], fab: FabricationSystem) -> Dict[str, Any]:
    """Return params with n_layers and n_segments derived from the design choice."""
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}


def main() -> None:
    # Clean up previous run artefacts so schema registry stays consistent
    if os.path.exists("./pfab_data"):
        shutil.rmtree("./pfab_data")
    os.makedirs("./plots", exist_ok=True)

    # ── Phase 0: Setup ────────────────────────────────────────────────────────
    print_phase_header(0, "Setup")
    schema = build_schema()
    fab    = FabricationSystem(CameraSystem(), EnergySensor())
    agent  = build_agent(schema, fab.camera, fab.energy)

    agent.configure_calibration(
        bounds={
            "water_ratio": (0.30, 0.50),
            "print_speed": (20.0, 60.0),
        },
        performance_weights={"path_accuracy": 0.75, "energy_efficiency": 0.25},
    )

    dataset = Dataset(schema=schema)

    # ── Phase 1: Baseline ─────────────────────────────────────────────────────
    print_phase_header(1, "Baseline Sampling", "8 Latin-hypercube experiments")
    baseline_specs = agent.baseline_step(n=8)

    baseline_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
    baseline_exps = []
    last_params: Dict[str, Any] = {}

    for i, spec in enumerate(baseline_specs):
        params   = _with_dimensions(params_from_spec(spec), fab)
        exp_code = f"baseline_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf     = get_performance(exp_data)
        baseline_exps.append(exp_data)
        baseline_log.append((exp_code, params, perf))
        last_params = params
        print_experiment_row(exp_code, params, perf)

    print_phase_summary(baseline_log)
    plot_path_comparison(baseline_exps[-1], fab.camera, last_params)
    plot_feature_heatmaps(baseline_exps[-1])

    # ── Phase 2: Initial Training ──────────────────────────────────────────────
    print_phase_header(2, "Initial Training")
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    agent.train(datamodule, validate=True)
    print_section("Training complete — plotting prediction accuracy")
    plot_prediction_accuracy(agent, datamodule)

    # Tracking lists for Phase 3/4
    all_params: List[Dict[str, Any]] = [params_from_spec(s) for s in baseline_specs]
    all_phases: List[str]            = ["baseline"] * len(baseline_specs)
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]] = [
        (params_from_spec(s), get_performance(e))
        for s, e in zip(baseline_specs, baseline_exps)
    ]

    # ── Phase 3: Exploration ───────────────────────────────────────────────────
    print_phase_header(3, "Exploration", "4 rounds  (w_explore=0.7)")
    prev_params = _with_dimensions(params_from_spec(baseline_specs[-1]), fab)
    explore_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []

    for i in range(4):
        spec     = agent.exploration_step(datamodule, w_explore=0.7, current_params=prev_params)
        params   = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab)
        exp_code = f"explore_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        perf = get_performance(exp_data)
        all_params.append(params)
        all_phases.append("exploration")
        perf_history.append((params, perf))
        explore_log.append((exp_code, params, perf))
        prev_params = params
        print_experiment_row(exp_code, params, perf)

    print_phase_summary(explore_log)
    plot_parameter_space(all_params, all_phases)

    # ── Phase 4: Inference ─────────────────────────────────────────────────────
    DESIGN_INTENT = {"design": "B", "material": "reinforced"}
    print_phase_header(4, "Inference", f"3 rounds  ·  intent fixed: {DESIGN_INTENT}")
    agent.configure_calibration(fixed_params=DESIGN_INTENT)
    params = _with_dimensions({**prev_params, **DESIGN_INTENT}, fab)

    infer_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []

    for i in range(3):
        exp_code = f"infer_{i+1:02d}"
        exp_data = dataset.create_experiment(exp_code, parameters=params)
        fab.run_experiment(params)
        spec       = agent.inference_step(exp_data, datamodule, w_explore=0.0, current_params=params)
        next_params = _with_dimensions({**params, **params_from_spec(spec)}, fab)
        dataset.save_experiment(exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        perf = get_performance(exp_data)
        all_params.append(params)
        all_phases.append("inference")
        perf_history.append((params, perf))
        infer_log.append((exp_code, params, perf))
        print_experiment_row(exp_code, params, perf)
        params = next_params

    print_phase_summary(infer_log)
    plot_performance_trajectory(perf_history, all_phases)

    print_done()


if __name__ == "__main__":
    main()
