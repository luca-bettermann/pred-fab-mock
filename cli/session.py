"""Session helpers for the ADVEI 2026 mock CLI.

Lean compared to learning-by-printing's: the mock has no NocoDB/external data,
no trust regions, no trajectory. State persists two ways — a small `session.json`
for config, and pred-fab's `LocalData` for the experiments themselves (features +
performance), so each command rebuilds the agent and reloads prior experiments
from disk.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from typing import Any

from pred_fab.core import Dataset

from models.schema import build_advei_dataset_schema, derive_n_layers, N_NODES
from cli.agent_setup import build_agent, build_fab

SESSION_DIR = "local"
SESSION_FILE = os.path.join(SESSION_DIR, "session.json")
DATASETS = ("reference", "test", "grid", "discovery", "exploration", "inference")

DEFAULT_CONFIG: dict[str, Any] = {"kappa": 0.5, "seed": 0, "weights": None}


@contextmanager
def _quiet(enabled: bool):
    """Suppress stdout for the noisy systems-init unless verbose."""
    if not enabled:
        yield
        return
    real = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = real


def load_config() -> dict[str, Any]:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return dict(DEFAULT_CONFIG)


def save_config(config: dict[str, Any]) -> None:
    os.makedirs(SESSION_DIR, exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(config, f, indent=2)


def apply_config(agent: Any, config: dict[str, Any]) -> None:
    if config.get("weights"):
        agent.configure_performance(weights=config["weights"])
    if config.get("kappa") is not None:
        agent.configure_exploration(kappa=float(config["kappa"]))


def build_env(verbose: bool = False):
    """Build (agent, fab, dataset, config) with prior experiments reloaded from disk."""
    config = load_config()
    schema = build_advei_dataset_schema(root_folder=".")
    fab = build_fab(random_seed=config.get("seed"))
    with _quiet(not verbose):
        agent = build_agent(schema, fab, verbose=verbose)
    apply_config(agent, config)
    if config.get("seed") is not None:
        import torch
        torch.manual_seed(int(config["seed"]))
        agent.calibration_system.random_seed = int(config["seed"])
    dataset = Dataset(schema=schema)
    for ds in DATASETS:
        try:
            dataset.populate(dataset=ds)
        except Exception:
            pass
    return agent, fab, dataset, config


def with_dims(params: dict[str, Any]) -> dict[str, Any]:
    """Add the per-experiment tensor dimensions (variable n_layers from layer_height)."""
    return {**params, "n_layers": derive_n_layers(float(params["layer_height"])), "n_nodes": N_NODES}


def params_from_spec(spec: Any) -> dict[str, Any]:
    return dict(spec.initial_params.values)


def next_code(dataset: Dataset, prefix: str) -> str:
    existing = [c for c in dataset.get_experiment_codes() if c.startswith(prefix + "/")]
    return f"{prefix}/{len(existing):03d}"


def simulate_and_evaluate(agent: Any, fab: Any, dataset: Dataset,
                          params: dict[str, Any], code: str, dataset_code: str) -> Any:
    """Create one experiment, simulate it, score it, and persist it."""
    params = with_dims(params)
    exp = dataset.create_experiment(code, parameters=params, dataset_code=dataset_code)
    fab.run_experiment(params)
    agent.evaluate(exp)
    dataset.save_experiment(code)
    return exp


def perf_dict(exp: Any) -> dict[str, float]:
    return {k: float(v) for k, v in exp.performance.get_values_dict().items() if v is not None}


def fmt_perf(p: dict[str, float]) -> str:
    return "  ".join(f"{k.split('_')[0][:4]}={p.get(k, float('nan')):.2f}" for k in sorted(p))
