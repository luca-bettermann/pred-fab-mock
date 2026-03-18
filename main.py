"""Full PFAB journey showcase for simulated robotic extrusion printing.

Phases:
  0 — Setup
  1 — Baseline (8 experiments)
  2 — Initial Training
  3 — Exploration (4 rounds)
  4 — Inference (3 rounds)
  5 — Online Adaptation (layer-by-layer)
"""

import os
import shutil
from typing import Any, Dict, List, Tuple

from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor
from visualization.plots import (
    plot_feature_heatmaps,
    plot_prediction_accuracy,
    plot_parameter_space,
    plot_performance_trajectory,
    plot_adaptation,
)

# COMMENT: Create utils file where you add helpers etc. the main script should be as clean as possible.
# COMMENT: Maybe you could create classes for different stages of the process. Not sure if this makes sense, but think about it,

# ─── Helper ──────────────────────────────────────────────────────────────────

def _params_from_spec(spec: Any) -> Dict[str, Any]:
    """Extract a plain params dict from an ExperimentSpec."""
    return dict(spec.initial_params.to_dict())


def _get_performance(exp_data: Any) -> Dict[str, float]:
    """Extract available performance values from an evaluated ExperimentData."""
    perf = {}
    for name, val in exp_data.performance.get_values_dict().items():
        if val is not None:
            perf[name] = float(val)
    return perf


def _run_and_evaluate(
    dataset: Dataset,
    agent: Any,
    camera: CameraSystem,
    energy: EnergySensor,
    params: Dict[str, Any],
    exp_code: str,
) -> Any:
    """Create experiment, populate sensor caches, evaluate features + performance."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)

    # COMMENT: Instead of running the experiment for the sensors separately, I want you to create a FabrricationSystem class that has a run_experiment method which takes care of running all sensors and populating the caches. The run_experiment should iterate over the dimensions and populate gradually. This way, we will be able to add online adaptation later, where we call the agent directly during the fabrication steps.
    camera.run_experiment(params)
    energy.run_experiment(params)
    agent.evaluate(exp_data)
    return exp_data


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Clean up previous run artefacts so schema registry stays consistent
    if os.path.exists("./pfab_data"):
        shutil.rmtree("./pfab_data")
    os.makedirs("./plots", exist_ok=True)

    # ── Phase 0: Setup ────────────────────────────────────────────────────────
    print("\n[PHASE 0] Setup")
    schema = build_schema()
    camera = CameraSystem()
    energy = EnergySensor()
    agent = build_agent(schema, camera, energy)

    agent.configure_calibration(
        bounds={
            "layer_time":   (20.0, 70.0),
            "layer_height": (0.005, 0.010),
            "water_ratio":  (0.30, 0.50),
            "print_speed":  (20.0, 60.0),
        },
        performance_weights={"path_accuracy": 1.0, "energy_efficiency": 0.8},
    )

    dataset = Dataset(schema=schema)

    # ── Phase 1: Baseline ─────────────────────────────────────────────────────
    print("\n[PHASE 1] Baseline — 8 experiments")
    baseline_specs = agent.baseline_step(n=8)

    baseline_exps = []
    for i, spec in enumerate(baseline_specs):
        params = _params_from_spec(spec)
        params["n_layers"] = 5
        params["n_segments"] = 4
        exp_code = f"baseline_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, camera, energy, params, exp_code)
        baseline_exps.append(exp_data)
        perf = _get_performance(exp_data)
        print(f"  {exp_code}: {perf}")

    plot_feature_heatmaps(baseline_exps[-1])

    # ── Phase 2: Initial Training ──────────────────────────────────────────────
    print("\n[PHASE 2] Initial Training")
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    agent.train(datamodule, validate=True)

    plot_prediction_accuracy(agent, datamodule)

    # Tracking lists for Phase 3/4
    all_params: List[Dict[str, Any]] = [_params_from_spec(s) for s in baseline_specs]
    all_phases: List[str] = ["baseline"] * len(baseline_specs)
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]] = [
        (_params_from_spec(s), _get_performance(e))
        for s, e in zip(baseline_specs, baseline_exps)
    ]

    # ── Phase 3: Exploration ───────────────────────────────────────────────────
    print("\n[PHASE 3] Exploration — 4 rounds")
    # Use last baseline params as a starting point so categorical values propagate
    prev_params: Dict[str, Any] = _params_from_spec(baseline_specs[-1])
    prev_params["n_layers"] = 5
    prev_params["n_segments"] = 4
    for i in range(4):
        spec = agent.exploration_step(datamodule, w_explore=0.7, current_params=prev_params)
        params = dict(prev_params)           # inherit categoricals from prev
        params.update(_params_from_spec(spec))  # override with proposed continuous
        params["n_layers"] = 5
        params["n_segments"] = 4
        exp_code = f"explore_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, camera, energy, params, exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        perf = _get_performance(exp_data)
        all_params.append(params)
        all_phases.append("exploration")
        perf_history.append((params, perf))
        prev_params = params
        print(f"  {exp_code}: {perf}")

    plot_parameter_space(all_params, all_phases)

    # ── Phase 4: Inference ─────────────────────────────────────────────────────
    print("\n[PHASE 4] Inference — 3 rounds")
    for i in range(3):
        last_exp = list(dataset.get_all_experiments())[-1]
        spec = agent.inference_step(last_exp, datamodule, w_explore=0.0, current_params=prev_params)
        params = dict(prev_params)
        params.update(_params_from_spec(spec))
        params["n_layers"] = 5
        params["n_segments"] = 4
        exp_code = f"infer_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, camera, energy, params, exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        perf = _get_performance(exp_data)
        all_params.append(params)
        all_phases.append("inference")
        perf_history.append((params, perf))
        prev_params = params
        print(f"  {exp_code}: {perf}")

    plot_performance_trajectory(perf_history)

    # ── Phase 5: Online Adaptation ─────────────────────────────────────────────
    print("\n[PHASE 5] Online Adaptation — layer-by-layer")
    agent.configure_step_parameter("print_speed", "n_layers")
    agent.configure_calibration(
        adaptation_delta={"print_speed": 5.0},
    )

    # Start from last inferred parameters (includes categorical params via prev_params)
    adapt_params = dict(prev_params)
    adapt_params["n_layers"] = 5
    adapt_params["n_segments"] = 4

    adapt_exp = dataset.create_experiment("adapt_01", parameters=adapt_params)
    agent.set_active_experiment(adapt_exp)

    layer_speeds: List[float] = []
    layer_deviations: List[float] = []

    for layer_idx in range(5):
        # Populate sensor caches for this layer only
        camera.run_layer(adapt_params, layer_idx)
        energy.run_layer(adapt_params, layer_idx)

        # Extract features for this layer only
        start, end = adapt_exp.parameters.get_start_and_end_indices("n_layers", layer_idx)
        agent.feature_system.run_feature_extraction(  # type: ignore[union-attr]
            adapt_exp, evaluate_from=start, evaluate_to=end
        )

        # Record current speed and deviation — feature tensor shape: (n_layers, n_segments)
        layer_speeds.append(float(adapt_params["print_speed"]))
        feat_tensor = adapt_exp.features.get_value("path_deviation")
        layer_dev = float(feat_tensor[layer_idx, :].mean())  # type: ignore[index]
        layer_deviations.append(layer_dev)

        # Propose next print_speed (unless this is the last layer)
        if layer_idx < 4:
            spec = agent.adaptation_step(
                dimension="n_layers",
                step_index=layer_idx,
                exp_data=adapt_exp,
                record=True,
            )
            new_speed = float(spec.initial_params.get("print_speed", adapt_params["print_speed"]))
            adapt_params["print_speed"] = new_speed
            print(f"  Layer {layer_idx}: speed={layer_speeds[-1]:.1f} → {new_speed:.1f}, dev={layer_dev:.5f}")
        else:
            print(f"  Layer {layer_idx}: speed={layer_speeds[-1]:.1f}, dev={layer_dev:.5f}")

    plot_adaptation(layer_speeds, layer_deviations)
    print("\nDone. Plots saved to ./plots/")


if __name__ == "__main__":
    main()
