"""02 — Baseline Sampling Validation.

Verify that baseline_step(n) (Sobol) spreads experiments well across
the 2D parameter space (water_ratio x print_speed).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization import plot_baseline_scatter
from shared import make_env, ensure_plot_dir
from utils import params_from_spec

N_BASELINE = 20
BOUNDS = {"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)}


def main():
    plot_dir = ensure_plot_dir()
    agent, fab, dataset = make_env("02_baseline", verbose=False)
    agent.configure(bounds=BOUNDS)

    specs = agent.baseline_step(n=N_BASELINE)
    points = [params_from_spec(s) for s in specs]

    waters = [p["water_ratio"] for p in points]
    speeds = [p["print_speed"] for p in points]
    print(f"  Baseline: {N_BASELINE} experiments (Sobol sequence)")
    print(f"  Water ratio: [{min(waters):.3f}, {max(waters):.3f}]  (bounds: {BOUNDS['water_ratio']})")
    print(f"  Print speed: [{min(speeds):.1f}, {max(speeds):.1f}]  (bounds: {BOUNDS['print_speed']})")

    out = os.path.join(plot_dir, "02_baseline_coverage.png")
    plot_baseline_scatter(out, points)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
