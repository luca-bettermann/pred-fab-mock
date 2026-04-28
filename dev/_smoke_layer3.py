"""Layer 3 smoke — exploration step with vectorised DE.

Tuned for low-RAM hosts: short training (200 epochs), small DE budget
(popsize=4, maxiter=25), 3 baseline experiments, single exploration call.
We're verifying the path runs end-to-end, not optimisation quality.
"""

import gc
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.prediction_model import DevMLP, EnergyMLP
from shared import make_env, run_baseline, train_models, with_dims
from utils import params_from_spec

PERF_WEIGHTS = {"path_accuracy": 2.0, "energy_efficiency": 1.0, "production_rate": 1.0}


def main():
    warnings.filterwarnings("ignore")
    print("\n  Layer 3 smoke (batched autoreg + vectorised DE)")

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

    t0 = time.perf_counter()
    spec = agent.exploration_step(dm, kappa=0.5)
    t_explore = time.perf_counter() - t0

    print(f"\n  exploration step (κ=0.5): {t_explore:6.2f}s")
    print(f"  proposed:                  {params_from_spec(spec)}")
    print("\n  ✓ Layer 3 intact")


if __name__ == "__main__":
    main()
