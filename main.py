"""Full PFAB journey showcase for simulated robotic extrusion printing.

Phases:
  0 — Setup
  1 — Baseline (8 experiments)
  2 — Initial Training
  3 — Exploration (4 rounds)
  4 — Inference (3 rounds, with design intent fixed)
  5 — Online Adaptation (layer-by-layer)
"""

import os
import shutil
from typing import Any, Dict, List, Tuple

from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec, get_performance
from visualization.plots import (
    plot_feature_heatmaps,
    plot_prediction_accuracy,
    plot_parameter_space,
    plot_performance_trajectory,
    plot_adaptation,
)


def _run_and_evaluate(
    dataset: Dataset,
    agent: Any,
    fab: FabricationSystem,
    params: Dict[str, Any],
    exp_code: str,
) -> Any:
    """Create experiment, run fabrication, evaluate features + performance."""
    exp_data = dataset.create_experiment(exp_code, parameters=params)
    fab.run_experiment(params)
    agent.evaluate(exp_data)
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
    print("\n[PHASE 0] Setup")
    schema = build_schema()
    fab = FabricationSystem(CameraSystem(), EnergySensor())
    agent = build_agent(schema, fab.camera, fab.energy)

    agent.configure_calibration(
        bounds={
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
        params = _with_dimensions(params_from_spec(spec), fab)
        exp_code = f"baseline_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)
        baseline_exps.append(exp_data)
        print(f"  {exp_code}: {get_performance(exp_data)}")

    plot_feature_heatmaps(baseline_exps[-1])

    # ── Phase 2: Initial Training ──────────────────────────────────────────────
    print("\n[PHASE 2] Initial Training")
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.25)
    agent.train(datamodule, validate=True)

    plot_prediction_accuracy(agent, datamodule)

    # Tracking lists for Phase 3/4
    all_params: List[Dict[str, Any]] = [params_from_spec(s) for s in baseline_specs]
    all_phases: List[str] = ["baseline"] * len(baseline_specs)
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]] = [
        (params_from_spec(s), get_performance(e))
        for s, e in zip(baseline_specs, baseline_exps)
    ]

    # ── Phase 3: Exploration ───────────────────────────────────────────────────
    # design and material are free — exploration discovers the best system configuration.
    print("\n[PHASE 3] Exploration — 4 rounds")
    prev_params: Dict[str, Any] = _with_dimensions(params_from_spec(baseline_specs[-1]), fab)
    explore_exps = []
    for i in range(4):
        spec = agent.exploration_step(datamodule, w_explore=0.7, current_params=prev_params)
        params = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab)
        exp_code = f"explore_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        perf = get_performance(exp_data)
        all_params.append(params)
        all_phases.append("exploration")
        perf_history.append((params, perf))
        explore_exps.append(exp_data)
        prev_params = params
        print(f"  {exp_code}: {perf}")

    plot_parameter_space(all_params, all_phases)

    # ── Phase 4: Inference ─────────────────────────────────────────────────────
    # Fix design intent: pick the best-performing design + material from exploration,
    # then optimise only the continuous parameters (layer_height, water_ratio, print_speed).
    print("\n[PHASE 4] Inference — 3 rounds")
    best_entry = max(perf_history, key=lambda x: x[1].get("path_accuracy", 0.0))
    design_intent = {k: best_entry[0][k] for k in ("design", "material")}
    print(f"  Design intent (fixed): {design_intent}")
    agent.configure_calibration(fixed_params=design_intent)

    for i in range(3):
        last_exp = list(dataset.get_all_experiments())[-1]
        spec = agent.inference_step(last_exp, datamodule, w_explore=0.0, current_params=prev_params)
        params = _with_dimensions({**prev_params, **params_from_spec(spec)}, fab)
        exp_code = f"infer_{i+1:02d}"
        exp_data = _run_and_evaluate(dataset, agent, fab, params, exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)
        perf = get_performance(exp_data)
        all_params.append(params)
        all_phases.append("inference")
        perf_history.append((params, perf))
        prev_params = params
        print(f"  {exp_code}: {perf}")

    plot_performance_trajectory(perf_history)

    # ── Phase 5: Online Adaptation ─────────────────────────────────────────────
    print("\n[PHASE 5] Online Adaptation — layer-by-layer")
    agent.configure_step_parameter("print_speed", "n_layers")
    agent.configure_calibration(adaptation_delta={"print_speed": 5.0})

    adapt_params: Dict[str, Any] = _with_dimensions(prev_params, fab)
    adapt_exp = dataset.create_experiment("adapt_01", parameters=adapt_params)
    agent.set_active_experiment(adapt_exp)

    layer_speeds: List[float] = []
    layer_deviations: List[float] = []

    for layer_idx in range(5):
        fab.run_layer(adapt_params, layer_idx)

        start, end = adapt_exp.parameters.get_start_and_end_indices("n_layers", layer_idx)
        agent.feature_system.run_feature_extraction(  # type: ignore[union-attr]
            adapt_exp, evaluate_from=start, evaluate_to=end
        )

        layer_speeds.append(float(adapt_params["print_speed"]))
        feat_tensor = adapt_exp.features.get_value("path_deviation")
        layer_dev = float(feat_tensor[layer_idx, :].mean())  # type: ignore[index]
        layer_deviations.append(layer_dev)

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
