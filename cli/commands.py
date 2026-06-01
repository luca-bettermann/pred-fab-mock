"""CLI command implementations for the ADVEI 2026 mock.

Each command rebuilds the environment from disk, runs one PFAB step, and
persists results. Because the mock fabricates synthetically, discovery and
exploration simulate + evaluate in-process (no external "run the print" gap).
``--plot`` renders pred-fab showcase plots inline (saved to ``plots/``).
"""
from __future__ import annotations

import shutil
from typing import Any

import numpy as np

from cli.session import (
    build_env, load_config, save_config, next_code, simulate_and_evaluate,
    params_from_spec, perf_dict, fmt_perf, SESSION_DIR,
)
from cli import plots

_G = "\033[32m"; _C = "\033[36m"; _R = "\033[0m"


def _round(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in d.items()}


def _train(agent: Any, dataset: Any, *, val_size: float = 0.0) -> Any:
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=val_size)
    agent.train(dm, validate=val_size > 0)
    return dm


def _mean_perf(dataset: Any, prefix: str = "discovery") -> dict[str, float] | None:
    codes = [c for c in dataset.get_experiment_codes() if c.startswith(prefix + "/")]
    if not codes:
        return None
    acc: dict[str, list[float]] = {}
    for c in codes:
        for k, v in perf_dict(dataset.get_experiment(c)).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _best_experiment(dataset: Any) -> Any:
    best, best_score = None, -1.0
    for c in dataset.get_experiment_codes():
        exp = dataset.get_experiment(c)
        p = perf_dict(exp)
        s = sum(p.values()) / len(p) if p else -1.0
        if s > best_score:
            best, best_score = exp, s
    return best


def _try_plot(fn, *a, **k) -> None:
    try:
        plots.show_inline(fn(*a, **k))
    except Exception as e:  # plotting must never break the workflow
        print(f"  ! plot skipped: {e}")


def discovery(args: Any) -> None:
    agent, fab, dataset, config = build_env(verbose=args.verbose)
    specs = agent.discovery_step(n=args.n)
    print(f"\n  {_C}Discovery{_R} — {len(specs)} space-filling experiments (κ=1)\n")
    for spec in specs:
        code = next_code(dataset, "discovery")
        exp = simulate_and_evaluate(agent, fab, dataset, params_from_spec(spec), code, "discovery")
        print(f"  {code:<16s}  {fmt_perf(perf_dict(exp))}")
    print(f"\n  {_G}✓{_R} {len(specs)} discovery experiments saved.\n")
    if getattr(args, "plot", False):
        _try_plot(plots.acquisition_topology, agent, dataset, None, 1.0, "discovery")


def train(args: Any) -> None:
    agent, fab, dataset, config = build_env(verbose=args.verbose)
    n = len(dataset.get_experiment_codes())
    if n == 0:
        print("  ! No experiments yet — run 'discovery' first.")
        return
    print(f"\n  {_C}Train{_R} StructuralMLP on {n} experiments\n")
    _train(agent, dataset, val_size=0.25)
    print(f"\n  {_G}✓{_R} trained.\n")


def exploration(args: Any) -> None:
    agent, fab, dataset, config = build_env(verbose=args.verbose)
    if not dataset.get_experiment_codes():
        print("  ! No experiments yet — run 'discovery' first.")
        return
    dm = _train(agent, dataset, val_size=0.0)
    kappa = args.kappa if args.kappa is not None else float(config.get("kappa", 0.5))
    spec = agent.exploration_step(dm, kappa=kappa)
    proposal = params_from_spec(spec)
    code = next_code(dataset, "exploration")
    exp = simulate_and_evaluate(agent, fab, dataset, proposal, code, "exploration")
    print(f"\n  {_C}Exploration{_R} (κ={kappa}) → {code}")
    print(f"  proposal:  {_round(proposal)}")
    print(f"  measured:  {fmt_perf(perf_dict(exp))}")
    print(f"\n  {_G}✓{_R} {code} saved.\n")
    if getattr(args, "plot", False):
        _try_plot(plots.acquisition_topology, agent, dataset, proposal, kappa, "exploration")
        _try_plot(plots.performance_radar, perf_dict(exp), _mean_perf(dataset), "exploration", code)


def inference(args: Any) -> None:
    agent, fab, dataset, config = build_env(verbose=args.verbose)
    if not dataset.get_experiment_codes():
        print("  ! No experiments yet — run 'discovery' + 'train' first.")
        return
    dm = _train(agent, dataset, val_size=0.0)
    spec = agent.acquisition_step(dm, kappa=0.0)
    proposal = params_from_spec(spec)
    print(f"\n  {_C}Inference{_R} (κ=0) — predicted-optimal parameters")
    print(f"  proposal:  {_round(proposal)}\n")
    if getattr(args, "plot", False):
        _try_plot(plots.acquisition_topology, agent, dataset, proposal, 0.0, "inference")


def report(args: Any) -> None:
    """Train on the session, then render the showcase plots."""
    agent, fab, dataset, config = build_env(verbose=False)
    if not dataset.get_experiment_codes():
        print("  ! No experiments yet — run 'discovery' first.")
        return
    dm = _train(agent, dataset, val_size=0.0)
    kappa = float(config.get("kappa", 0.5))
    print(f"\n  {_C}Report{_R} — acquisition topology (κ={kappa}) + best-experiment radar\n")
    _try_plot(plots.acquisition_topology, agent, dataset, None, kappa, "report")
    best = _best_experiment(dataset)
    if best is not None:
        _try_plot(plots.performance_radar, perf_dict(best), _mean_perf(dataset), "report", f"best: {best.code}")
    print()


def configure(args: Any) -> None:
    config = load_config()
    if args.kappa is not None:
        config["kappa"] = args.kappa
    if args.seed is not None:
        config["seed"] = args.seed
    save_config(config)
    print(f"\n  {_C}Config{_R}")
    for k, v in config.items():
        print(f"  {k:<10s} {v}")
    print()


def summary(args: Any) -> None:
    agent, fab, dataset, config = build_env(verbose=False)
    codes = sorted(dataset.get_experiment_codes())
    print(f"\n  {_C}Session summary{_R} — {len(codes)} experiments  (κ={config.get('kappa')}, seed={config.get('seed')})\n")
    for ds in ("discovery", "exploration", "inference"):
        members = [c for c in codes if c.startswith(ds + "/")]
        if members:
            print(f"  {ds:<12s} {len(members):>2d}  ({members[0]} … {members[-1]})")
    print()


def reset(args: Any) -> None:
    import os
    removed = []
    for d in (SESSION_DIR, "logs", "local", "plots"):
        if os.path.isdir(d):
            shutil.rmtree(d)
            removed.append(d)
    print(f"  {_G}✓{_R} reset" + (f" (removed {', '.join(sorted(set(removed)))})" if removed else " (nothing to remove)"))
