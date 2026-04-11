"""CLI for the PFAB mock — run each phase as a separate command.

Usage:
    python cli.py configure --bounds '{"water_ratio":[0.30,0.50],"print_speed":[20.0,60.0]}' \\
                            --weights '{"path_accuracy":2.0,"energy_efficiency":1.0,"production_rate":1.0}' \\
                            --material clay --optimizer DE --buffer 0.10 0.8 2.0
    python cli.py baseline --n 20
    python cli.py train --val-size 0.25
    python cli.py explore --n 10 --w-explore 0.7
    python cli.py infer --n 3 --intent '{"design":"A","material":"clay"}'
    python cli.py adapt --start-speed 40.0 --delta '{"print_speed":5.0}'
    python cli.py summary
    python cli.py reset
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from pred_fab.orchestration import Optimizer
from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec, get_performance
from workflow import JourneyState, with_dimensions, run_and_evaluate

SESSION_FILE = ".pfab_session.json"
DATA_ROOT = "./pfab_data"


def _next_code(state: 'JourneyState', prefix: str) -> str:
    """Generate next experiment code like 'baseline_06' based on existing codes."""
    existing = [c for c in state.all_codes if c.startswith(prefix + "_")]
    return f"{prefix}_{len(existing) + 1:02d}"


# ── Session persistence ──────────────────────────────────────────────────────

def _save_session(config: dict[str, Any], journey: JourneyState) -> None:
    """Save session config + journey state to JSON."""
    data = {
        "config": config,
        "journey": {
            "all_params": journey.all_params,
            "all_phases": journey.all_phases,
            "all_codes": journey.all_codes,
            "perf_history": [(p, pf) for p, pf in journey.perf_history],
            "prev_params": journey.prev_params,
        },
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_session() -> tuple[dict[str, Any], JourneyState]:
    """Load session from JSON. Raises if no session exists."""
    if not os.path.exists(SESSION_FILE):
        print("No session found. Run 'python cli.py configure' first.")
        sys.exit(1)
    with open(SESSION_FILE) as f:
        data = json.load(f)
    config = data["config"]
    j = data["journey"]
    state = JourneyState()
    state.all_params = j["all_params"]
    state.all_phases = j["all_phases"]
    state.all_codes = j["all_codes"]
    state.perf_history = [(p, pf) for p, pf in j["perf_history"]]
    state.prev_params = j["prev_params"]
    return config, state


def _rebuild(
    config: dict[str, Any],
    verbose: bool = False,
) -> tuple[Any, Dataset, FabricationSystem]:
    """Reconstruct agent + dataset + fab from session config."""
    # Suppress all init logs for non-configure commands
    if not verbose:
        _real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    try:
        schema = build_schema()
        fab = FabricationSystem(CameraSystem(), EnergySensor())
        agent = build_agent(schema, fab.camera, fab.energy, verbose=verbose)

        # Apply configuration
        configure_kwargs: dict[str, Any] = {}
        if config.get("bounds"):
            configure_kwargs["bounds"] = {
                k: tuple(v) for k, v in config["bounds"].items()
            }
        if config.get("performance_weights"):
            configure_kwargs["performance_weights"] = config["performance_weights"]
        if config.get("fixed_params"):
            configure_kwargs["fixed_params"] = config["fixed_params"]
        if config.get("optimizer"):
            configure_kwargs["optimizer"] = Optimizer(config["optimizer"])
        if config.get("exploration_radius") is not None:
            configure_kwargs["exploration_radius"] = config["exploration_radius"]
        if config.get("mpc_lookahead") is not None:
            configure_kwargs["mpc_lookahead"] = config["mpc_lookahead"]
        if config.get("mpc_discount") is not None:
            configure_kwargs["mpc_discount"] = config["mpc_discount"]
        if config.get("boundary_buffer"):
            configure_kwargs["boundary_buffer"] = tuple(config["boundary_buffer"])
        if configure_kwargs:
            agent.configure(**configure_kwargs)

        dataset = Dataset(schema=schema)
        dataset.populate()
    finally:
        if not verbose:
            sys.stdout.close()
            sys.stdout = _real_stdout

    return agent, dataset, fab


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_configure(args: argparse.Namespace) -> None:
    """Set up agent configuration and save session."""
    config: dict[str, Any] = {}

    if args.bounds:
        config["bounds"] = json.loads(args.bounds)
    if args.weights:
        config["performance_weights"] = json.loads(args.weights)
    if args.material:
        config["fixed_params"] = {"material": args.material}
    if args.optimizer:
        config["optimizer"] = args.optimizer
    if args.radius is not None:
        config["exploration_radius"] = args.radius
    if args.mpc_lookahead is not None:
        config["mpc_lookahead"] = args.mpc_lookahead
    if args.mpc_discount is not None:
        config["mpc_discount"] = args.mpc_discount
    if args.buffer:
        config["boundary_buffer"] = args.buffer

    # Verify config by building agent (verbose to show state report)
    agent, dataset, fab = _rebuild(config, verbose=True)
    state = JourneyState()
    _save_session(config, state)


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run baseline experiments."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(1, "Baseline Sampling",
                                      f"{args.n} Sobol-sequence experiments")

    specs = agent.baseline_step(n=args.n)
    log = []
    for i, spec in enumerate(specs):
        params = with_dimensions(params_from_spec(spec), fab)
        exp_code = _next_code(state, "baseline")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        log.append((exp_code, params, perf))
        state.record("baseline", exp_code, params_from_spec(spec), perf)
        agent.console.print_experiment_row(exp_code, params, perf)

    state.prev_params = with_dimensions(params_from_spec(specs[-1]), fab)
    agent.console.print_phase_summary(log)
    _save_session(config, state)


def cmd_train(args: argparse.Namespace) -> None:
    """Train prediction models on current dataset."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(2, "Training",
                                      "Fit prediction models on available data")

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=args.val_size)
    agent.train(datamodule, validate=args.val_size > 0)

    _save_session(config, state)


def cmd_explore(args: argparse.Namespace) -> None:
    """Run exploration rounds."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(
        3, "Exploration",
        f"{args.n} rounds  (w_explore={args.w_explore})"
    )

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.0)
    agent.train(datamodule, validate=False)

    log = []
    for i in range(args.n):
        spec = agent.exploration_step(
            datamodule, w_explore=args.w_explore,
            n_optimization_rounds=args.opt_rounds,
        )
        proposed = params_from_spec(spec)
        params = with_dimensions({**state.prev_params, **proposed}, fab)
        exp_code = _next_code(state, "explore")

        # Compute uncertainty before retraining
        proposed_full = {**state.prev_params, **proposed, "n_layers": 5, "n_segments": 4}
        u = agent.predict_uncertainty(proposed_full, datamodule)

        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        log.append((exp_code, params, perf))
        state.record("exploration", exp_code, params, perf)

        agent.console.print_exploration_row(
            exp_code, params, perf, u, agent.last_opt_score
        )
        agent.console.print_optimizer_stats(agent.last_opt_n_starts, agent.last_opt_nfev)

        # Retrain
        datamodule.update()
        agent.train(datamodule, validate=False)

    agent.console.print_phase_summary(log)
    _save_session(config, state)


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference rounds."""
    config, state = _load_session()

    # Apply design intent
    if args.intent:
        intent = json.loads(args.intent)
        config.setdefault("fixed_params", {}).update(intent)

    agent, dataset, fab = _rebuild(config)

    intent_display = config.get("fixed_params", {})
    agent.console.print_phase_header(
        4, "Inference",
        f"{args.n} rounds  ·  intent: {intent_display}  ·  w_explore=0"
    )

    # Update prev_params with design intent
    state.prev_params = with_dimensions({**state.prev_params, **intent_display}, fab)

    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.0)
    agent.train(datamodule, validate=False)

    log = []
    for i in range(args.n):
        params = state.prev_params
        exp_code = _next_code(state, "infer")

        exp_data = dataset.create_experiment(exp_code, parameters=params)
        fab.run_experiment(params)

        spec = agent.inference_step(
            exp_data, datamodule, w_explore=0.0,
            n_optimization_rounds=args.opt_rounds, current_params=params,
        )
        next_params = with_dimensions({**params, **params_from_spec(spec)}, fab)

        dataset.save_experiment(exp_code)
        datamodule.update()
        agent.train(datamodule, validate=False)

        perf = get_performance(exp_data)
        log.append((exp_code, params, perf))
        state.record("inference", exp_code, params, perf)
        state.prev_params = next_params

        agent.console.print_inference_row(
            exp_code, params, perf, agent.last_opt_score
        )
        agent.console.print_optimizer_stats(agent.last_opt_n_starts, agent.last_opt_nfev)

    agent.console.print_phase_summary(log)
    _save_session(config, state)


def cmd_adapt(args: argparse.Namespace) -> None:
    """Run online adaptation."""
    config, state = _load_session()

    delta = json.loads(args.delta) if args.delta else {"print_speed": 5.0}
    config.setdefault("adaptation_delta", delta)

    agent, dataset, fab = _rebuild(config)
    agent.configure(
        step_parameters={"print_speed": "n_layers"},
        adaptation_delta=delta,
    )

    agent.console.print_phase_header(
        5, "Online Adaptation",
        "print_speed adjusted after each layer based on live deviation feedback"
    )

    # Train prediction models (needed for adaptation_step optimizer)
    datamodule = agent.create_datamodule(dataset)
    datamodule.prepare(val_size=0.0)
    agent.train(datamodule, validate=False)

    adapt_params = with_dimensions(
        {**state.prev_params, "print_speed": args.start_speed}, fab
    )
    adapt_code = _next_code(state, "adapt")
    adapt_exp = dataset.create_experiment(adapt_code, parameters=adapt_params)
    agent.set_active_experiment(adapt_exp)

    n_layers = int(adapt_params["n_layers"])
    speeds: list[float] = []
    deviations: list[float] = []

    for layer_idx in range(n_layers):
        fab.run_layer(adapt_params, layer_idx)
        start, end = adapt_exp.parameters.get_start_and_end_indices("n_layers", layer_idx)
        agent.feature_system.run_feature_extraction(
            adapt_exp, evaluate_from=start, evaluate_to=end,
        )

        speed = float(adapt_params["print_speed"])
        speeds.append(speed)
        feat = adapt_exp.features.get_value("path_deviation")
        dev = float(np.mean(feat[layer_idx, :]))
        deviations.append(dev)

        if layer_idx < n_layers - 1:
            spec = agent.adaptation_step(
                dimension="n_layers", step_index=layer_idx,
                exp_data=adapt_exp, record=True,
            )
            new_speed = float(spec.initial_params.get("print_speed", speed))
            adapt_params["print_speed"] = new_speed
            agent.console.print_adaptation_row(
                layer_idx, speed, dev, new_speed, agent.last_opt_nfev
            )
        else:
            agent.console.print_adaptation_row(layer_idx, speed, dev)

    dataset.save_experiment(adapt_code)
    _save_session(config, state)


def cmd_summary(args: argparse.Namespace) -> None:
    """Print final run summary."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_run_summary(
        state.perf_history, state.all_phases, state.all_codes,
    )
    agent.console.print_done()


def cmd_reset(args: argparse.Namespace) -> None:
    """Clear session state and data."""
    import shutil
    for path in [SESSION_FILE, DATA_ROOT, "./plots", "./logs"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  Removed {path}")
    print("Session reset.")


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="PFAB mock CLI — run fabrication phases step by step",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # configure
    p = sub.add_parser("configure", help="Set up agent configuration")
    p.add_argument("--bounds", type=str, help='JSON: {"param": [lo, hi], ...}')
    p.add_argument("--weights", type=str, help='JSON: {"perf_attr": weight, ...}')
    p.add_argument("--material", type=str, help="Fixed material (e.g. clay)")
    p.add_argument("--optimizer", choices=["lbfgsb", "de"], default=None)
    p.add_argument("--radius", type=float, default=None, help="Exploration radius")
    p.add_argument("--mpc-lookahead", type=int, default=None)
    p.add_argument("--mpc-discount", type=float, default=None)
    p.add_argument("--buffer", type=float, nargs=3, metavar=("EXTENT", "STRENGTH", "EXPONENT"),
                   help="Boundary buffer: extent strength exponent")
    p.set_defaults(func=cmd_configure)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline experiments")
    p.add_argument("--n", type=int, default=20, help="Number of experiments")
    p.set_defaults(func=cmd_baseline)

    # train
    p = sub.add_parser("train", help="Train prediction models")
    p.add_argument("--val-size", type=float, default=0.25, help="Validation split fraction")
    p.set_defaults(func=cmd_train)

    # explore
    p = sub.add_parser("explore", help="Run exploration rounds")
    p.add_argument("--n", type=int, default=10, help="Number of rounds")
    p.add_argument("--w-explore", type=float, default=0.7, help="Exploration weight")
    p.add_argument("--opt-rounds", type=int, default=5, help="Optimizer restarts")
    p.set_defaults(func=cmd_explore)

    # infer
    p = sub.add_parser("infer", help="Run inference rounds")
    p.add_argument("--n", type=int, default=3, help="Number of rounds")
    p.add_argument("--intent", type=str, help='JSON: {"design":"A","material":"clay"}')
    p.add_argument("--opt-rounds", type=int, default=5, help="Optimizer restarts")
    p.set_defaults(func=cmd_infer)

    # adapt
    p = sub.add_parser("adapt", help="Run online adaptation")
    p.add_argument("--start-speed", type=float, default=40.0)
    p.add_argument("--delta", type=str, help='JSON: {"print_speed": 5.0}')
    p.set_defaults(func=cmd_adapt)

    # summary
    p = sub.add_parser("summary", help="Print final run summary")
    p.set_defaults(func=cmd_summary)

    # reset
    p = sub.add_parser("reset", help="Clear session state and data")
    p.set_defaults(func=cmd_reset)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
