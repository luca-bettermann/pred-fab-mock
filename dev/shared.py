"""Shared helpers for dev test scripts.

Every dev/0X_*.py file imports from here to avoid duplicating
agent setup, experiment running, and test-set generation.
"""

import os
import sys
import shutil
from typing import Any

import numpy as np

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.core import Dataset
from pred_fab.orchestration import PfabAgent

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from utils import params_from_spec
from workflow import run_and_evaluate, with_dimensions

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def make_env(
    data_tag: str = "default",
    verbose: bool = True,
) -> tuple[PfabAgent, FabricationSystem, Dataset]:
    """Build a fresh agent + fabrication system + dataset.

    Each call creates a clean data directory under dev/data/{data_tag}.
    """
    root = os.path.join(DATA_DIR, data_tag)
    if os.path.exists(root):
        shutil.rmtree(root)
    schema = build_schema(root_folder=root)
    fab    = FabricationSystem(CameraSystem(), EnergySensor())
    agent  = build_agent(schema, fab.camera, fab.energy, verbose=verbose)
    dataset = Dataset(schema=schema)
    return agent, fab, dataset


def run_baseline(
    agent: PfabAgent,
    fab: FabricationSystem,
    dataset: Dataset,
    n: int,
) -> list[dict[str, Any]]:
    """Execute N baseline experiments (Sobol). Returns list of param dicts."""
    specs = agent.baseline_step(n=n)
    all_params = []
    for i, spec in enumerate(specs):
        params = with_dimensions(params_from_spec(spec))
        run_and_evaluate(dataset, agent, fab, params, f"baseline_{i+1:02d}")
        all_params.append(params)
    return all_params


def train_models(agent: PfabAgent, dataset: Dataset, val_size: float = 0.25):
    """Create datamodule, prepare, train. Returns (datamodule, train_results)."""
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=val_size)
    results = agent.train(dm, validate=val_size > 0)
    return dm, results


def build_test_grid(
    n_water: int = 16,
    n_speed: int = 16,
    water_range: tuple[float, float] = (0.31, 0.49),
    speed_range: tuple[float, float] = (21.0, 59.0),
) -> list[dict[str, Any]]:
    """Build a proper 2D grid of test parameters (independent axes)."""
    waters = np.linspace(water_range[0], water_range[1], n_water)
    speeds = np.linspace(speed_range[0], speed_range[1], n_speed)
    test_params = []
    for w in waters:
        for spd in speeds:
            test_params.append(with_dimensions({
                "water_ratio": float(w),
                "print_speed": float(spd),
            }))
    return test_params


def clean_plots(*subdirs: str) -> None:
    """Ensure plot directories exist, clearing any existing contents."""
    for sub in subdirs:
        path = os.path.join(PLOT_DIR, sub)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def ensure_plot_dir() -> str:
    """Ensure the base plot directory exists and return its path."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR
