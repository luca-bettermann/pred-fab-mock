"""Layer 3 smoke — exploration step with vectorised DE.

Tuned for low-RAM hosts: short training (200 epochs), small DE budget
(popsize=4, maxiter=25), 3 baseline experiments, single exploration call.
We're verifying the path runs end-to-end, not optimisation quality.

To enable hot-path profiling, run with ``PFAB_PROFILE=1`` or pass ``--profile``::

    PFAB_PROFILE=1 python dev/_smoke_layer3.py
    python dev/_smoke_layer3.py --profile
"""

import argparse
import gc
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.utils import profiler

from models.prediction_model import DevMLP, EnergyMLP
from shared import make_env, run_baseline, train_models, with_dims
from utils import params_from_spec

PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}


def main(profile: bool = False) -> None:
    warnings.filterwarnings("ignore")
    print("\n  Layer 3 smoke (batched autoreg + vectorised DE)")

    if profile:
        profiler.enable()
        print("  profiling: enabled (PFAB hot-path sections)")

    # Shorten training — smoke verifies the path, not convergence quality
    DevMLP.EPOCHS = 200
    EnergyMLP.EPOCHS = 200

    agent, fab, dataset = make_env("smoke_layer3", verbose=False)
    agent.configure_performance(weights=PERF_WEIGHTS)
    # Tight DE budget — popsize=4 × D=4 = 16 candidates, 25 iter
    agent.configure_optimizer(de_popsize=4, de_maxiter=25)

    run_baseline(agent, fab, dataset, 3)
    dm, _ = train_models(agent, dataset, val_size=0.0)

    sample = with_dims({"water_ratio": 0.4, "print_speed": 35.0})
    _ = agent.predict_performance(sample)

    gc.collect()

    # Reset profiler so only the exploration step is measured (excludes
    # baseline / training / first-call torch.compile cost).
    if profile:
        profiler.reset()

    t0 = time.perf_counter()
    spec = agent.exploration_step(dm, kappa=0.5)
    t_explore = time.perf_counter() - t0

    print(f"\n  exploration step (κ=0.5): {t_explore:6.2f}s")
    print(f"  proposed:                  {params_from_spec(spec)}")
    print("\n  ✓ Layer 3 intact")

    if profile:
        print("\n  Profile (sorted by total time):")
        print(profiler.report(sort_by="total"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 3 smoke (with optional profiling)")
    p.add_argument("--profile", action="store_true",
                   help="Enable PFAB profiler and print breakdown after run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Honour env var as a second activation path
    if os.environ.get("PFAB_PROFILE", "").lower() in ("1", "true", "yes"):
        args.profile = True
    main(profile=args.profile)
