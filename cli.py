"""CLI for the ADVEI 2026 mock — step-by-step predictive-fabrication workflow.

Each command delegates to a standalone step module. Steps can also be run
directly: ``python steps/baseline.py --n 5``.
"""

import argparse

from steps import (
    analyse, baseline, configure, explore, grid, inference,
    init_agent, init_physics, init_schema, report, reset, summary, test_set,
)


_EPILOG = """\
Quick start:
  # Setup
  uv run cli.py reset
  uv run cli.py init-schema
  uv run cli.py init-agent
  uv run cli.py init-physics --seed 42 --plot
  uv run cli.py configure \\
      --weights '{"structural_integrity":2,"material_deposition":1, \\
                  "extrusion_stability":1,"energy_footprint":1,"fabrication_time":1}' \\
      --schedule print_speed:n_layers --schedule slowdown_factor:n_layers

  # Static grids
  uv run cli.py grid --dataset-code reference --low-pct 0.25 --high-pct 0.75 \\
      --fractional-x 1 --half-face-centers --n-center 1
  uv run cli.py test-set --n-center 3

  # System discovery
  uv run cli.py baseline --n 5 --plot
  uv run cli.py explore --n 5 --kappa 0.5 --plot
  uv run cli.py analyse --plot --test-set 20

  # Production
  uv run cli.py inference --design-intent '{"layer_height":2.5}' --plot
  uv run cli.py report baseline_01 --plot
  uv run cli.py summary
"""


def _wide_formatter(prog: str) -> argparse.HelpFormatter:
    return argparse.HelpFormatter(prog, max_help_position=36, width=110)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="ADVEI 2026 mock CLI — predictive fabrication workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # reset
    p = sub.add_parser("reset", help="Clear session state and data",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=reset.run)

    # init-schema
    p = sub.add_parser("init-schema", help="Show the problem schema",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=init_schema.run)

    # init-agent
    p = sub.add_parser("init-agent", help="Initialise the agent",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=init_agent.run)

    # init-physics
    p = sub.add_parser("init-physics",
                       help="Randomize physics constants and show topology",
                       formatter_class=_wide_formatter)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--plot", action="store_true",
                   help="Show plots inline in terminal")
    p.set_defaults(func=init_physics.run)

    # configure
    p = sub.add_parser("configure",
                       help="Set weights / exploration / optimiser / schedule",
                       formatter_class=_wide_formatter)
    p.add_argument("--show", action="store_true",
                   help="Show all current configuration values")
    p.add_argument("--weights", type=str, default=None, metavar="JSON",
                   help="Performance weights JSON")
    p.add_argument("--radius", type=float, default=None,
                   help="Evidence decay radius (default: 0.09)")
    p.add_argument("--sigma", type=float, default=None,
                   help="Direct sigma override (bypasses radius scaling)")
    p.add_argument("--trust-regions", type=str, default=None, metavar="JSON",
                   help="Per-param max change per step")
    p.add_argument("--schedule", action="append", default=[], metavar="PARAM:DIM",
                   help="Schedule param over dimension (repeatable)")
    p.add_argument("--de-maxiter", type=int, default=None,
                   help="DE max generations (default: 30)")
    p.add_argument("--de-popsize", type=int, default=None,
                   help="DE population size (default: 64)")
    p.set_defaults(func=configure.run)

    # baseline
    p = sub.add_parser("baseline",
                       help="Run baseline (Sobol) experiments",
                       formatter_class=_wide_formatter)
    p.add_argument("--n", type=int, default=5, help="Number of experiments")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.add_argument("--schedule", action="append", default=[], metavar="PARAM:DIM",
                   help="Override configured schedule (repeatable)")
    p.set_defaults(func=baseline.run)

    # grid
    p = sub.add_parser("grid", help="Run a CCF static-grid dataset",
                       formatter_class=_wide_formatter)
    p.add_argument("--dataset-code", type=str, default="grid",
                   help="Tag for dataset grouping (default: grid)")
    p.add_argument("--low-pct", type=float, default=0.25,
                   help="Low-level percentile (default: 0.25)")
    p.add_argument("--high-pct", type=float, default=0.75,
                   help="High-level percentile (default: 0.75)")
    p.add_argument("--fractional-x", type=int, default=0,
                   help="Fractional design generator index")
    p.add_argument("--half-face-centers", action="store_true",
                   help="Use half-fraction face centers")
    p.add_argument("--n-center", type=int, default=1,
                   help="Number of center-point replicates")
    p.set_defaults(func=grid.run)

    # test-set
    p = sub.add_parser("test-set",
                       help="Run ADVEI test dataset (full CCF, 0.15/0.85)",
                       formatter_class=_wide_formatter)
    p.add_argument("--n-center", type=int, default=3,
                   help="Center-point replicates (default: 3)")
    p.set_defaults(func=test_set.run)

    # explore
    p = sub.add_parser("explore",
                       help="Run model-guided exploration rounds",
                       formatter_class=_wide_formatter)
    p.add_argument("--n", type=int, default=5, help="Number of rounds")
    p.add_argument("--kappa", type=float, default=0.5,
                   help="Exploration weight (0=exploit, 1=explore)")
    p.add_argument("--plot", action="store_true", help="Show per-round plots")
    p.add_argument("--schedule", action="append", default=[], metavar="PARAM:DIM",
                   help="Override configured schedule (repeatable)")
    p.set_defaults(func=explore.run)

    # inference
    p = sub.add_parser("inference",
                       help="First-time-right inference (kappa=0)",
                       formatter_class=_wide_formatter)
    p.add_argument("--design-intent", type=str, default=None, metavar="JSON",
                   help="Fix parameters for inference")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.add_argument("--schedule", action="append", default=[], metavar="PARAM:DIM",
                   help="Override configured schedule (repeatable)")
    p.set_defaults(func=inference.run)

    # analyse
    p = sub.add_parser("analyse",
                       help="Compare ground truth vs. prediction + MAE",
                       formatter_class=_wide_formatter)
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.add_argument("--test-set", type=int, default=0, dest="test_set",
                   help="Generate N test experiments and report MAE/R²")
    p.set_defaults(func=analyse.run)

    # summary
    p = sub.add_parser("summary",
                       help="Print run summary across all phases",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=summary.run)

    # report
    p = sub.add_parser("report",
                       help="Generate visual report for an experiment",
                       formatter_class=_wide_formatter)
    p.add_argument("exp_code", type=str,
                   help="Experiment code (e.g. baseline_01)")
    p.add_argument("--plot", action="store_true",
                   help="Show plots inline in terminal")
    p.set_defaults(func=report.run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
