"""02 — Baseline Sampling Validation.

Verify that baseline_step(n) (LHS) spreads experiments well across
the 5D parameter space.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import make_env, ensure_plot_dir
from utils import params_from_spec
from schema import PARAM_BOUNDS

N_BASELINE = 20


def main():
    ensure_plot_dir()
    agent, _fab, _dataset = make_env("02_baseline", verbose=False)

    specs = agent.baseline_step(n=N_BASELINE)
    points = [params_from_spec(s) for s in specs]

    print(f"  Baseline: {N_BASELINE} experiments (Sobol space-filling)")
    for code, lo, hi in PARAM_BOUNDS:
        vals = [p[code] for p in points]
        print(f"  {code:<20s}: [{min(vals):.4f}, {max(vals):.4f}]  (schema: [{lo}, {hi}])")


if __name__ == "__main__":
    main()
