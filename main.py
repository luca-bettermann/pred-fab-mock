"""Full PFAB journey showcase for simulated robotic extrusion printing.

Phases:
  0 — Setup
  1 — Baseline (4 experiments)
  2 — Initial Training
  3 — Exploration (6 rounds)
  4 — Inference (3 rounds, with design intent fixed)
  5 — Online Adaptation (layer-by-layer)
"""

# ── Quick-test flag ────────────────────────────────────────────────────────────
# When True: baseline n=2, exploration rounds=2, inference rounds=1
QUICK_TEST = False

import os
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np

from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec, get_performance
from visualization import (
    plot_path_comparison_3d,
    plot_filament_volume,
    plot_prediction_accuracy,
    plot_parameter_space,
    plot_performance_trajectory,
    plot_adaptation,
    plot_inference_convergence,
    plot_acquisition_topology,
    plot_physics_topology,
    plot_baseline_scatter,
    print_phase_header,
    print_section,
    print_experiment_row,
    print_phase_summary,
    print_training_summary,
    print_adaptation_row,
    print_run_summary,
    print_done,
    print_explore_row,
    print_infer_row,
    print_optimizer_row,
)

# ── Phase output directories ───────────────────────────────────────────────────
_D1 = "./plots/phase_1_baseline"
_D2 = "./plots/phase_2_training"
_D3 = "./plots/phase_3_exploration"
_D4 = "./plots/phase_4_inference"
_D5 = "./plots/phase_5_adaptation"


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
    agent.logger.set_console_output(False)
    agent.evaluate(exp_data)
    dataset.save_experiment(exp_code)
    agent.logger.set_console_output(True)
    return exp_data


def _with_dimensions(params: Dict[str, Any], fab: FabricationSystem) -> Dict[str, Any]:
    """Return params with n_layers and n_segments derived from the design choice."""
    n_layers, n_segments = fab.get_dimensions(params["design"])
    return {**params, "n_layers": n_layers, "n_segments": n_segments}



def main() -> None:
    # Clean up previous run artefacts so schema registry stays consistent
    if os.path.exists("./pfab_data"):
        shutil.rmtree("./pfab_data")
    if os.path.exists("./plots"):
        shutil.rmtree("./plots")
    for d in [_D1, _D2, _D3, _D4, _D5]:
        os.makedirs(d, exist_ok=True)

    # ── Phase 0: Setup ────────────────────────────────────────────────────────
    print_phase_header(0, "Setup", "Configure agent, sensors, schema, and calibration bounds")
    schema = build_schema()
    fab    = FabricationSystem(CameraSystem(), EnergySensor())
    agent  = build_agent(schema, fab.camera, fab.energy)

    agent.configure(
        bounds={
            "water_ratio": (0.30, 0.50),
            "print_speed": (20.0, 60.0),
        },
        performance_weights={"path_accuracy": 2, "energy_efficiency": 1, "production_rate": 1},
    )

    dataset = Dataset(schema=schema)

    # ── Phase 1: Baseline ─────────────────────────────────────────────────────
    _n_baseline    = 2 if QUICK_TEST else 20
    _n_explore     = 2 if QUICK_TEST else 10
    _n_infer       = 1 if QUICK_TEST else 3
    print_phase_header(1, "Baseline Sampling",
                       f"{_n_baseline} Latin-hypercube experiments — no model yet, space-filling only")

    agent.logger.set_console_output(False)
    baseline_specs = agent.baseline_step(n=_n_baseline)
    agent.logger.set_console_output(True)

    baseline_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
    baseline_exps = []

    for i, spec in enumerate(baseline_specs):
        params   = _with_dimensions(params_from_spec(spec), fab)
        exp_code = f"baseline_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf     = get_performance(exp_data)
        baseline_exps.append(exp_data)
        baseline_log.append((exp_code, params, perf))
        print_experiment_row(exp_code, params, perf)

    print_phase_summary(baseline_log)
    plot_path_comparison_3d(baseline_exps[-1], fab.camera, params, save_dir=_D1)  # type: ignore[possibly-undefined]
    plot_filament_volume(baseline_exps[-1], fab.camera, params, save_dir=_D1)  # type: ignore[possibly-undefined]
    plot_physics_topology(agent, save_dir=_D1)
    plot_baseline_scatter(baseline_log, save_dir=_D1)
    print_section(f"4 plots saved to {_D1}/")

    # ── Phase 2: Initial Training ──────────────────────────────────────────────
    print_phase_header(2, "Initial Training",
                       "Fit prediction models (deviation + energy) on baseline data")
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    agent.logger.set_console_output(False)
    agent.train(datamodule, validate=True)
    agent.logger.set_console_output(True)
    r2_scores = plot_prediction_accuracy(agent, datamodule, save_dir=_D2)
    print_training_summary(r2_scores)
    print_section(f"1 plot saved to {_D2}/")

    # Tracking lists for Phase 3/4
    all_params: List[Dict[str, Any]] = [params_from_spec(s) for s in baseline_specs]
    all_phases: List[str]            = ["baseline"] * len(baseline_specs)
    all_codes:  List[str]            = [f"baseline_{i+1:02d}" for i in range(len(baseline_specs))]

    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]] = [
        (params_from_spec(s), get_performance(e))
        for s, e in zip(baseline_specs, baseline_exps)
    ]

    # ── Phase 3: Exploration ───────────────────────────────────────────────────
    print_phase_header(3, "Exploration",
                       f"{_n_explore} rounds  (w_explore=0.7) — model guides search toward uncertain regions")
    prev_params = _with_dimensions(params_from_spec(baseline_specs[-1]), fab)
    explore_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
    W_EXPLORE = 0.7

    for i in range(_n_explore):
        agent.logger.set_console_output(False)
        spec = agent.exploration_step(datamodule, w_explore=W_EXPLORE)
        agent.logger.set_console_output(True)

        _proposed_p = {**prev_params, **params_from_spec(spec), "n_layers": 5, "n_segments": 4}
        params = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab)
        exp_code = f"explore_{i+1:02d}"

        # Topology plot before running the experiment (shows what the model "sees")
        # Use proposed design+material as reference so topology shows the relevant slice
        plot_acquisition_topology(
            agent, W_EXPLORE, params_from_spec(spec),
            all_params, params_from_spec(spec), exp_code, save_dir=_D3,
        )

        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)

        agent.logger.set_console_output(False)
        datamodule.update()
        agent.train(datamodule, validate=False)
        agent.logger.set_console_output(True)

        perf = get_performance(exp_data)
        all_params.append(params)
        all_phases.append("exploration")
        all_codes.append(exp_code)
        perf_history.append((params, perf))
        explore_log.append((exp_code, params, perf))
        prev_params = params
        _u_at_proposed = agent.predict_uncertainty(_proposed_p, datamodule)
        print_explore_row(exp_code, params, perf, _u_at_proposed, agent.last_opt_score, W_EXPLORE)
        print_optimizer_row(agent.last_opt_n_starts, agent.last_opt_nfev)

    print_phase_summary(explore_log)
    plot_parameter_space(all_params, all_phases, perf_history, save_dir=_D3)
    print_section(f"7 plots saved to {_D3}/")

    # ── Phase 4: Inference ─────────────────────────────────────────────────────
    DESIGN_INTENT = {"design": "A", "material": "concrete"}
    print_phase_header(4, "Inference",
                       f"{_n_infer} rounds  ·  intent fixed: {DESIGN_INTENT}  ·  model optimises w_explore=0")
    agent.configure(fixed_params=DESIGN_INTENT)
    params = _with_dimensions({**prev_params, **DESIGN_INTENT}, fab)

    infer_log: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []

    for i in range(_n_infer):
        exp_code = f"infer_{i+1:02d}"
        exp_data = dataset.create_experiment(exp_code, parameters=params)
        fab.run_experiment(params)

        agent.logger.set_console_output(False)
        spec = agent.inference_step(exp_data, datamodule, w_explore=0.0, current_params=params)
        agent.logger.set_console_output(True)

        next_params = _with_dimensions({**params, **params_from_spec(spec)}, fab)

        # Topology for inference step
        plot_acquisition_topology(
            agent, 0.0, params_from_spec(spec),
            all_params, DESIGN_INTENT, exp_code, save_dir=_D4,
        )

        agent.logger.set_console_output(False)
        dataset.save_experiment(exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        agent.logger.set_console_output(True)

        perf = get_performance(exp_data)
        all_params.append(params)
        all_phases.append("inference")
        all_codes.append(exp_code)
        perf_history.append((params, perf))
        infer_log.append((exp_code, params, perf))
        print_infer_row(exp_code, params, perf, agent.last_opt_score)
        print_optimizer_row(agent.last_opt_n_starts, agent.last_opt_nfev)
        params = next_params

    prev_params = params
    print_phase_summary(infer_log)
    plot_performance_trajectory(perf_history, all_phases, exp_codes=all_codes, save_dir=_D4)
    plot_inference_convergence(infer_log, DESIGN_INTENT, save_dir=_D4)
    print_section(f"5 plots saved to {_D4}/")

    # ── Phase 5: Online Adaptation ─────────────────────────────────────────────
    print_phase_header(5, "Online Adaptation",
                       "print_speed adjusted after each layer based on live deviation feedback")
    agent.configure(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta={"print_speed": 5.0},
    )

    adapt_params = _with_dimensions({**prev_params, "print_speed": 40.0}, fab)
    adapt_exp    = dataset.create_experiment("adapt_01", parameters=adapt_params)
    agent.set_active_experiment(adapt_exp)

    layer_speeds:     List[float] = []
    layer_deviations: List[float] = []

    for layer_idx in range(5):
        fab.run_layer(adapt_params, layer_idx)
        start, end = adapt_exp.parameters.get_start_and_end_indices("n_layers", layer_idx)
        agent.feature_system.run_feature_extraction(  # type: ignore[union-attr]
            adapt_exp, evaluate_from=start, evaluate_to=end
        )

        speed_before = float(adapt_params["print_speed"])
        layer_speeds.append(speed_before)

        feat = adapt_exp.features.get_value("path_deviation")
        layer_dev = float(np.mean(feat[layer_idx, :]))
        layer_deviations.append(layer_dev)

        if layer_idx < 4:
            agent.logger.set_console_output(False)
            spec      = agent.adaptation_step(
                dimension="n_layers", step_index=layer_idx,
                exp_data=adapt_exp, record=True,
            )
            agent.logger.set_console_output(True)
            new_speed = float(spec.initial_params.get("print_speed", speed_before))
            adapt_params["print_speed"] = new_speed
            n_evals = agent.last_opt_nfev
            print_adaptation_row(layer_idx, speed_before, layer_dev, speed_after=new_speed, n_evals=n_evals)
        else:
            print_adaptation_row(layer_idx, speed_before, layer_dev)

    agent.logger.set_console_output(False)
    dataset.save_experiment("adapt_01")
    agent.logger.set_console_output(True)

    # Counterfactual at constant speed=40
    from sensors.physics import path_deviation as _phys_dev
    no_adapt_design   = str(adapt_params.get("design", "A"))
    no_adapt_water    = float(adapt_params.get("water_ratio", 0.36))
    no_adapt_material = str(adapt_params.get("material", "concrete"))
    fixed_dev = [
        float(
            sum(
                _phys_dev(40.0, no_adapt_design, seg_idx, no_adapt_water, no_adapt_material, layer_idx)
                for seg_idx in range(int(adapt_params["n_segments"]))
            )
            / int(adapt_params["n_segments"])
        )
        for layer_idx in range(5)
    ]

    plot_adaptation(layer_speeds, layer_deviations, no_adapt_deviations=fixed_dev, save_dir=_D5)
    print_section(f"1 plot saved to {_D5}/")

    # ── Run summary ────────────────────────────────────────────────────────────
    from sensors.physics import (  # type: ignore[import-not-found]
        DELTA, THETA, DESIGN_COMPLEXITY, MAT_SAG, W_OPTIMAL,
    )
    _di         = DESIGN_INTENT
    _complexity = DESIGN_COMPLEXITY[str(_di["design"])]
    _sag        = MAT_SAG[str(_di["material"])]
    _w_opt      = W_OPTIMAL[str(_di["material"])]
    _spd_opt    = float(np.clip(
        np.sqrt(THETA * _sag / (DELTA * _complexity)), 20.0, 60.0
    ))
    print_run_summary(
        perf_history, all_phases, all_codes, DESIGN_INTENT,
        phys_opt_speed=_spd_opt, phys_opt_water=_w_opt,
    )

    print_done()


if __name__ == "__main__":
    main()
