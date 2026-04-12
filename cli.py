"""CLI for the PFAB mock — run each phase as a separate command.

Usage:
    python cli.py configure --bounds '{"water_ratio":[0.30,0.50],"print_speed":[20.0,60.0]}'
    python cli.py baseline --n 20
    python cli.py train --val-size 0.25
    python cli.py explore --n 10 --kappa 0.7
    python cli.py infer
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


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


SESSION_FILE = ".pfab_session.json"


def _next_code(state: JourneyState, prefix: str) -> str:
    existing = [c for c in state.all_codes if c.startswith(prefix + "_")]
    return f"{prefix}_{len(existing) + 1:02d}"


# ── Session persistence ──────────────────────────────────────────────────────

def _save_session(config: dict[str, Any], journey: JourneyState) -> None:
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
        json.dump(_to_native(data), f, indent=2)


def _load_session() -> tuple[dict[str, Any], JourneyState]:
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


def _rebuild(config: dict[str, Any], verbose: bool = False) -> tuple[Any, Dataset, FabricationSystem]:
    """Reconstruct agent + dataset + fab from session config."""
    if not verbose:
        _real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    try:
        schema = build_schema()
        fab = FabricationSystem(CameraSystem(), EnergySensor())
        agent = build_agent(schema, fab.camera, fab.energy, verbose=verbose)

        configure_kwargs: dict[str, Any] = {}
        if config.get("bounds"):
            configure_kwargs["bounds"] = {k: tuple(v) for k, v in config["bounds"].items()}
        if config.get("performance_weights"):
            configure_kwargs["performance_weights"] = config["performance_weights"]
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
        if config.get("de_maxiter") is not None:
            configure_kwargs["de_maxiter"] = config["de_maxiter"]
        if config.get("de_popsize") is not None:
            configure_kwargs["de_popsize"] = config["de_popsize"]
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
    if args.de_maxiter is not None:
        config["de_maxiter"] = args.de_maxiter
    if args.de_popsize is not None:
        config["de_popsize"] = args.de_popsize

    agent, dataset, fab = _rebuild(config, verbose=True)
    state = JourneyState()
    _save_session(config, state)


def cmd_baseline(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(1, "Baseline", f"{args.n} Sobol experiments")
    specs = agent.baseline_step(n=args.n)

    for spec in specs:
        params = with_dimensions(params_from_spec(spec))
        exp_code = _next_code(state, "baseline")
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("baseline", exp_code, params, perf)
        agent.console.print_experiment_row(exp_code, params, perf)

    state.prev_params = with_dimensions(params_from_spec(specs[-1]))
    _save_session(config, state)


def cmd_train(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(2, "Training", "Fit prediction models")
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=args.val_size)
    results = agent.train(dm, validate=args.val_size > 0)

    if results:
        agent.console.print_training_summary(results)
    _save_session(config, state)


def cmd_explore(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(3, "Exploration",
                                      f"{args.n} rounds (kappa={args.kappa})")
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    for i in range(args.n):
        spec = agent.exploration_step(dm, kappa=args.kappa)
        proposed = params_from_spec(spec)
        params = with_dimensions({**state.prev_params, **proposed})
        exp_code = _next_code(state, "explore")

        u = agent.predict_uncertainty(params, dm)
        exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
        perf = get_performance(exp_data)
        state.record("exploration", exp_code, params, perf)

        agent.console.print_exploration_row(exp_code, params, perf, u, agent.last_opt_score)

        dm.update()
        agent.train(dm, validate=False)

    _save_session(config, state)


def cmd_infer(args: argparse.Namespace) -> None:
    """Single-shot inference — first-time-right manufacturing."""
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)

    agent.console.print_phase_header(4, "Inference", "Single-shot first-time-right")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.0)
    agent.train(dm, validate=False)

    spec = agent.exploration_step(dm, kappa=0.0)
    proposed = params_from_spec(spec)
    params = with_dimensions({**state.prev_params, **proposed})
    exp_code = _next_code(state, "infer")

    exp_data = run_and_evaluate(dataset, agent, fab, params, exp_code)
    perf = get_performance(exp_data)
    state.record("inference", exp_code, params, perf)

    agent.console.print_inference_row(exp_code, params, perf, agent.last_opt_score)
    _save_session(config, state)


def cmd_summary(args: argparse.Namespace) -> None:
    config, state = _load_session()
    agent, dataset, fab = _rebuild(config)
    agent.console.print_run_summary(state.perf_history, state.all_phases, state.all_codes)
    agent.console.print_done()


def cmd_reset(args: argparse.Namespace) -> None:
    import shutil
    for path in [SESSION_FILE, "./local", "./plots", "./logs"]:
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
    p.add_argument("--optimizer", choices=["lbfgsb", "de"], default=None)
    p.add_argument("--radius", type=float, default=None, help="Exploration radius")
    p.add_argument("--mpc-lookahead", type=int, default=None)
    p.add_argument("--mpc-discount", type=float, default=None)
    p.add_argument("--buffer", type=float, nargs=3, metavar=("EXTENT", "STRENGTH", "EXP"),
                   help="Boundary buffer: extent strength exponent")
    p.add_argument("--de-maxiter", type=int, default=None, help="DE max generations")
    p.add_argument("--de-popsize", type=int, default=None, help="DE population size")
    p.set_defaults(func=cmd_configure)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline experiments")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=cmd_baseline)

    # train
    p = sub.add_parser("train", help="Train prediction models")
    p.add_argument("--val-size", type=float, default=0.25)
    p.set_defaults(func=cmd_train)

    # explore
    p = sub.add_parser("explore", help="Run exploration rounds")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--kappa", type=float, default=0.7, help="Exploration weight (0=exploit, 1=explore)")
    p.set_defaults(func=cmd_explore)

    # infer
    p = sub.add_parser("infer", help="Single-shot inference")
    p.set_defaults(func=cmd_infer)

    # summary
    p = sub.add_parser("summary", help="Print run summary")
    p.set_defaults(func=cmd_summary)

    # reset
    p = sub.add_parser("reset", help="Clear session state and data")
    p.set_defaults(func=cmd_reset)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
