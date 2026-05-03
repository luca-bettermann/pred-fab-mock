"""CLI for the ADVEI 2026 mock — step-by-step predictive-fabrication workflow.

Each command delegates to a standalone step module. Steps can also be run
directly: ``python steps/baseline.py --n 5``.

Quick start:

    uv run cli.py reset
    uv run cli.py init-schema
    uv run cli.py init-agent
    uv run cli.py configure --weights '{"structural_integrity":2,"material_deposition":1,
        "extrusion_stability":1,"energy_footprint":1,"fabrication_time":1}' \\
        --schedule print_speed:n_layers --schedule slowdown_factor:n_layers

    # Static grids
    uv run cli.py grid --dataset-code reference --low-pct 0.25 --high-pct 0.75 \\
        --fractional-x 1 --half-face-centers --n-center 1   # 22 runs
    uv run cli.py test-set --n-center 3                      # 45 runs (full CCF, 0.15/0.85)

    # System discovery
    uv run cli.py baseline --n 5
    uv run cli.py explore --n 5 --kappa 0.5

    # Production
    uv run cli.py inference --design-intent '{"layer_height":2.5}'
    uv run cli.py summary
"""

import argparse

from steps import (
    baseline, configure, explore, grid, inference,
    init_agent, init_schema, reset, summary, test_set,
)


def _wide_formatter(prog: str) -> argparse.HelpFormatter:
    return argparse.HelpFormatter(prog, max_help_position=36, width=110)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="ADVEI 2026 mock CLI — predictive fabrication workflow",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reset", help="Clear session state and data", formatter_class=_wide_formatter)
    p.set_defaults(func=reset.run)

    p = sub.add_parser("init-schema", help="Show the problem schema", formatter_class=_wide_formatter)
    p.set_defaults(func=init_schema.run)

    p = sub.add_parser("init-agent", help="Initialise the agent", formatter_class=_wide_formatter)
    p.set_defaults(func=init_agent.run)

    p = sub.add_parser("configure", help="Set weights / exploration / optimiser / schedule",
                       formatter_class=_wide_formatter)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--trust-regions", type=str, default=None)
    p.add_argument("--schedule", action="append", default=[])
    p.add_argument("--de-maxiter", type=int, default=None)
    p.add_argument("--de-popsize", type=int, default=None)
    p.set_defaults(func=configure.run)

    p = sub.add_parser("baseline", help="Run baseline (Sobol) experiments",
                       formatter_class=_wide_formatter)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--schedule", action="append", default=[])
    p.set_defaults(func=baseline.run)

    p = sub.add_parser("grid", help="Run a CCF static-grid dataset",
                       formatter_class=_wide_formatter)
    p.add_argument("--dataset-code", type=str, default="grid")
    p.add_argument("--low-pct", type=float, default=0.25)
    p.add_argument("--high-pct", type=float, default=0.75)
    p.add_argument("--fractional-x", type=int, default=0)
    p.add_argument("--half-face-centers", action="store_true")
    p.add_argument("--n-center", type=int, default=1)
    p.set_defaults(func=grid.run)

    p = sub.add_parser("test-set", help="Run ADVEI's full-CCF test dataset (0.15/0.85, 45 runs by default)",
                       formatter_class=_wide_formatter)
    p.add_argument("--n-center", type=int, default=3)
    p.set_defaults(func=test_set.run)

    p = sub.add_parser("explore", help="Run model-guided exploration rounds",
                       formatter_class=_wide_formatter)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--schedule", action="append", default=[])
    p.set_defaults(func=explore.run)

    p = sub.add_parser("inference", help="First-time-right inference call",
                       formatter_class=_wide_formatter)
    p.add_argument("--design-intent", type=str, default=None)
    p.add_argument("--schedule", action="append", default=[])
    p.set_defaults(func=inference.run)

    p = sub.add_parser("summary", help="Print run summary across all phases",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=summary.run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
