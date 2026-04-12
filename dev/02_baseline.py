"""02 — Baseline Sampling Validation.

Verify that baseline_step(n) (LHS) spreads experiments well across
the 2D parameter space (water_ratio x print_speed).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization import plot_baseline_scatter
from shared import make_env, ensure_plot_dir
from utils import params_from_spec

N_BASELINE = 20


def main():
    plot_dir = ensure_plot_dir()
    agent, fab, dataset = make_env("02_baseline", verbose=False)

    specs = agent.baseline_step(n=N_BASELINE)
    points = [params_from_spec(s) for s in specs]

    waters = [p["water_ratio"] for p in points]
    speeds = [p["print_speed"] for p in points]
    print(f"  Baseline: {N_BASELINE} experiments (\u03ba=1, pure evidence)")
    print(f"  Water ratio: [{min(waters):.3f}, {max(waters):.3f}]  (schema: [0.30, 0.50])")
    print(f"  Print speed: [{min(speeds):.1f}, {max(speeds):.1f}]  (schema: [20.0, 60.0])")

    out = os.path.join(plot_dir, "02_baseline_coverage.png")
    plot_baseline_scatter(out, points)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
