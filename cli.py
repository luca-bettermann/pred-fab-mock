"""CLI for the PFAB mock — step-by-step predictive fabrication workflow.

Each command delegates to a standalone step file in steps/.
Steps can also be run directly: python steps/baseline.py --n 5

Quick start:
    # Setup
    uv run cli.py reset
    uv run cli.py init-schema
    uv run cli.py init-agent
    uv run cli.py init-physics --seed 42 --plot
    uv run cli.py configure \\
        --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}' \\
        --schedule print_speed:n_layers \\
        --trust-regions '{"print_speed":5.0}'

    # System discovery
    uv run cli.py baseline --n 5 --plot
    uv run cli.py report baseline_01 --plot
    uv run cli.py explore --n 5 --kappa 0.5
    uv run cli.py analyse --plot --test-set 20

    # Production
    uv run cli.py inference --design-intent '{"n_layers":5}' --plot
    uv run cli.py adapt --design-intent '{"n_layers":5}'
    uv run cli.py summary
"""

import argparse

from steps import (
    reset, init_schema, init_agent, init_physics, configure,
    baseline, explore, analyse, inference,
    adapt, summary, report,
)


def _wide_formatter(prog: str) -> argparse.HelpFormatter:
    """Argparse formatter with a wider help column to avoid mid-sentence wraps."""
    return argparse.HelpFormatter(prog, max_help_position=36, width=110)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="PFAB mock CLI — predictive fabrication workflow step by step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  # Setup
  uv run cli.py reset
  uv run cli.py init-schema
  uv run cli.py init-agent
  uv run cli.py init-physics --seed 42 --plot
  uv run cli.py configure \\
      --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}' \\
      --schedule print_speed:n_layers \\
      --trust-regions '{"print_speed":5.0}'

  # System discovery
  uv run cli.py baseline --n 5 --plot
  uv run cli.py report baseline_01 --plot
  uv run cli.py explore --n 5 --kappa 0.5
  uv run cli.py analyse --plot --test-set 20

  # Production
  uv run cli.py inference --design-intent '{"n_layers":5}' --plot
  uv run cli.py adapt --design-intent '{"n_layers":5}'
  uv run cli.py summary
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # reset
    p = sub.add_parser("reset", help="Clear all session state and data",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=reset.run)

    # init-schema
    p = sub.add_parser("init-schema", help="Show the problem schema",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=init_schema.run)

    # init-agent
    p = sub.add_parser("init-agent", help="Initialize the agent",
                       formatter_class=_wide_formatter)
    p.add_argument("--model", choices=["mlp", "rf"], default="mlp",
                   help="Prediction model type (default: mlp)")
    p.set_defaults(func=init_agent.run)

    # init-physics
    p = sub.add_parser("init-physics", help="Randomize physics constants and show topology",
                       formatter_class=_wide_formatter)
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--plot", action="store_true", help="Show plots inline in terminal")
    p.set_defaults(func=init_physics.run)

    # configure
    p = sub.add_parser("configure", help="Set agent configuration",
                       formatter_class=_wide_formatter)
    p.add_argument("--show", action="store_true",
                   help="Show all current configuration values")
    p.add_argument("--weights", type=str, default=None, metavar="JSON",
                   help='Performance weights — e.g. \'{"path_accuracy":2,"energy_efficiency":1}\'')
    p.add_argument("--bounds", type=str, default=None, metavar="JSON",
                   help='Parameter bounds — e.g. \'{"water_ratio":[0.35,0.45]}\'')
    p.add_argument("--trust-regions", type=str, default=None, metavar="JSON", dest="trust_regions",
                   help='Per-param max change per step (used by both schedule and adaptation). '
                        'Default = bounds_span / 10. Override e.g. \'{"print_speed":5.0}\'')
    p.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                   help="Default schedule (e.g. print_speed:n_layers). Repeatable. "
                        "Per-command --schedule overrides.")
    p.add_argument("--radius", type=float, default=None,
                   help="Evidence decay radius (default: 0.09). σ = radius · √n_active_dims")
    p.add_argument("--sigma", type=float, default=None,
                   help="Direct σ override (bypasses radius × √D scaling)")
    p.add_argument("--mc-exp-offset", type=float, default=None, dest="mc_exp_offset",
                   help="Sobol MC exponent offset; M = round(2^(D + offset)), default 3.0")
    p.add_argument("--de-maxiter", type=int, default=None,
                   help="Phase 1 DE max generations (default: 30)")
    p.add_argument("--de-popsize", type=int, default=None,
                   help="Phase 1 DE population size (default: 64)")
    p.set_defaults(func=configure.run)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline experiments (space-filling)",
                       formatter_class=_wide_formatter)
    p.add_argument("--n", type=int, default=5, help="Number of experiments")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                   help="Override the configured schedule (e.g. print_speed:n_layers). Repeatable.")
    p.add_argument("--design-intent", type=str, default=None,
                   help="JSON: fix parameters (required for schedule). Example: '{\"n_layers\":5}'")
    p.add_argument("--iterations", type=int, default=None, help="DE max iterations for this run")
    p.set_defaults(func=baseline.run)

    # explore
    p = sub.add_parser("explore", help="Run exploration rounds (incremental)",
                       formatter_class=_wide_formatter)
    p.add_argument("--n", type=int, default=5, help="Number of rounds to add")
    p.add_argument("--kappa", type=float, default=0.5,
                   help="Exploration weight (0=exploit, 1=explore)")
    p.add_argument("--plot", action="store_true", help="Show per-round plots inline")
    p.add_argument("--validate", action="store_true", help="Validate model during training")
    p.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                   help="Override the configured schedule. Repeatable.")
    p.add_argument("--design-intent", type=str, default=None,
                   help="JSON: fix parameters for schedule mode")
    p.add_argument("--iterations", type=int, default=None, help="DE max iterations for this run")
    p.set_defaults(func=explore.run)

    # analyse — folds the old standalone test-set step
    p = sub.add_parser("analyse",
                       help="Compare ground truth vs. prediction; optional numerical evaluation on test set",
                       formatter_class=_wide_formatter)
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.add_argument("--test-set", type=int, default=0, dest="test_set",
                   help="Generate N test experiments inline and report MAE / R² (default: 0 = visual only)")
    p.set_defaults(func=analyse.run)

    # inference
    p = sub.add_parser("inference", help="Single-shot first-time-right manufacturing",
                       formatter_class=_wide_formatter)
    p.add_argument("--design-intent", type=str, default=None,
                   help="JSON: fix parameters for inference. Example: '{\"n_layers\":5}'")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                   help="Override the configured schedule. Repeatable.")
    p.add_argument("--iterations", type=int, default=None, help="DE max iterations for this run")
    p.set_defaults(func=inference.run)

    # ── Advanced commands ──

    # adapt — uses configured trust regions; no --schedule (adaptation IS scheduling at runtime)
    p = sub.add_parser("adapt", help="Online inference with layer-by-layer adaptation",
                       formatter_class=_wide_formatter)
    p.add_argument("--design-intent", type=str, default=None,
                   help="JSON: fix parameters. Example: '{\"n_layers\":5}'")
    p.set_defaults(func=adapt.run)

    # summary
    p = sub.add_parser("summary", help="Show run summary across all phases",
                       formatter_class=_wide_formatter)
    p.set_defaults(func=summary.run)

    # report
    p = sub.add_parser("report", help="Generate visual report for an experiment",
                       formatter_class=_wide_formatter)
    p.add_argument("exp_code", type=str, help="Experiment code (e.g. base_01, explore_03)")
    p.add_argument("--plot", action="store_true", help="Show plots inline in terminal")
    p.set_defaults(func=report.run)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
