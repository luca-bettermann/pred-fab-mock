"""Strategy D commit 5+ smoke — gradient optimiser vs DE for baseline_step.

Runs the same N=3 baseline twice (DE then GRADIENT) on the mock, prints
wall time + final acquisition score for each, plus the proposed params.

Use to confirm the gradient path runs end-to-end and to track relative
optimisation quality / cost. Expectation: gradient is faster (~10-30×
fewer evals) at comparable score quality on smooth landscapes.
"""

import gc
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.orchestration import Optimizer
from pred_fab.utils import profiler

from models.prediction_model import DevMLP, EnergyMLP
from shared import make_env
from utils import params_from_spec


def _run_baseline(label: str, optimizer: Optimizer, profile: bool = False) -> tuple[float, list]:
    DevMLP.EPOCHS = 200
    EnergyMLP.EPOCHS = 200

    agent, fab, dataset = make_env(f"smoke_gradient_{label}", verbose=False)
    agent.configure_optimizer(
        backend=optimizer,
        de_popsize=4, de_maxiter=25,
        gradient_n_starts=4, gradient_n_iters=40, gradient_method="adam",
    )

    if profile:
        profiler.reset()
    t0 = time.perf_counter()
    specs = agent.baseline_step(n=3)
    elapsed = time.perf_counter() - t0
    return elapsed, specs


def main(profile: bool = False) -> None:
    warnings.filterwarnings("ignore")
    print("\n  Strategy D smoke — GRADIENT vs DE on baseline_step(n=3)")
    print("  Note: only the Process phase (continuous-only) will route to gradient;")
    print("  Domain phase (integer / domain dims) always uses DE.\n")

    if profile:
        profiler.enable()

    print("  ── DE ──")
    t_de, specs_de = _run_baseline("de", Optimizer.DE)
    print(f"    wall: {t_de:.2f}s")
    for i, s in enumerate(specs_de):
        print(f"    exp_{i}: {params_from_spec(s)}")
    gc.collect()

    print("\n  ── GRADIENT ──")
    t_grad, specs_grad = _run_baseline("grad", Optimizer.GRADIENT)
    print(f"    wall: {t_grad:.2f}s")
    for i, s in enumerate(specs_grad):
        print(f"    exp_{i}: {params_from_spec(s)}")

    ratio = t_de / t_grad if t_grad > 0 else float("nan")
    if ratio >= 1.0:
        print(f"\n  GRADIENT is {ratio:.2f}× faster than DE on this problem.")
    else:
        print(f"\n  DE is {1/ratio:.2f}× faster than GRADIENT on this problem (small-D regime).")
    if profile:
        print("\n  Profile (sorted by total time):")
        print(profiler.report(sort_by="total"))


if __name__ == "__main__":
    main(profile="--profile" in sys.argv)
