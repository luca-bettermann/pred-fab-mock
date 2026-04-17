"""CLI for the PFAB mock — step-by-step predictive fabrication workflow.

Each command delegates to a standalone step file in steps/.
Steps can also be run directly: python steps/baseline.py --n 10

Quick start:
    uv run cli.py reset
    uv run cli.py init-schema
    uv run cli.py init-agent
    uv run cli.py configure --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'
    uv run cli.py init-physics --seed 42 --plot
    uv run cli.py baseline --n 10 --plot
    uv run cli.py explore --n 5 --kappa 0.5 --plot
    uv run cli.py test-set --n 20
    uv run cli.py analyse --plot
    uv run cli.py inference --design-intent '{"n_layers":5}' --plot
    uv run cli.py summary
"""

import argparse

from steps import (
    reset, init_schema, init_agent, init_physics, configure,
    baseline, explore, test_set, analyse, inference,
    explore_trajectory, adapt, summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="PFAB mock CLI — predictive fabrication workflow step by step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  uv run cli.py reset
  uv run cli.py init-schema
  uv run cli.py init-agent
  uv run cli.py configure --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'
  uv run cli.py init-physics --seed 42 --plot
  uv run cli.py baseline --n 10 --plot
  uv run cli.py explore --n 5 --kappa 0.5 --plot
  uv run cli.py test-set --n 20
  uv run cli.py analyse --plot
  uv run cli.py inference --design-intent '{"n_layers":5}' --plot
  uv run cli.py summary
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # reset
    p = sub.add_parser("reset", help="Clear all session state and data")
    p.set_defaults(func=reset.run)

    # init-schema
    p = sub.add_parser("init-schema", help="Show the problem schema")
    p.set_defaults(func=init_schema.run)

    # init-agent
    p = sub.add_parser("init-agent", help="Initialize the agent")
    p.add_argument("--model", choices=["mlp", "rf"], default="mlp", help="Prediction model type (default: mlp)")
    p.set_defaults(func=init_agent.run)

    # init-physics
    p = sub.add_parser("init-physics", help="Randomize physics constants and show topology")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--plot", action="store_true", help="Show plots inline in terminal")
    p.set_defaults(func=init_physics.run)

    # configure
    p = sub.add_parser("configure", help="Set agent configuration",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="""
Configuration groups:

  Performance:
    --weights JSON           Performance attribute weights
                             Example: '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}'

  Exploration:
    --radius FLOAT           KDE exploration radius (default: 0.15)
    --buffer FLOAT           Normalization buffer for perf/unc (default: 0.5)
    --decay-exp FLOAT        Bandwidth decay exponent (default: 0.5)

  Optimizer:
    --optimizer {de,lbfgsb}  Backend (default: de)
    --de-maxiter INT         DE max generations (default: 100)
    --de-popsize INT         DE population size (default: 10)

  Bounds:
    --bounds JSON            Parameter search bounds override
                             Example: '{"water_ratio":[0.35,0.45]}'
""")
    p.add_argument("--bounds", type=str, help="JSON: parameter bounds override")
    p.add_argument("--weights", type=str, help="JSON: performance attribute weights")
    p.add_argument("--optimizer", choices=["lbfgsb", "de"], default=None)
    p.add_argument("--radius", type=float, default=None, help="Exploration radius")
    p.add_argument("--buffer", type=float, default=None, help="Normalization buffer (default: 0.5)")
    p.add_argument("--decay-exp", type=float, default=None, help="Bandwidth decay exponent (default: 0.5)")
    p.add_argument("--de-maxiter", type=int, default=None)
    p.add_argument("--de-popsize", type=int, default=None)
    p.set_defaults(func=configure.run)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline experiments (space-filling)")
    p.add_argument("--n", type=int, default=10, help="Number of experiments")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=baseline.run)

    # explore
    p = sub.add_parser("explore", help="Run exploration rounds (incremental)")
    p.add_argument("--n", type=int, default=5, help="Number of rounds to add")
    p.add_argument("--kappa", type=float, default=0.5, help="Exploration weight (0=exploit, 1=explore)")
    p.add_argument("--plot", action="store_true", help="Show per-round plots inline")
    p.add_argument("--validate", action="store_true", help="Validate model during training")
    p.set_defaults(func=explore.run)

    # test-set
    p = sub.add_parser("test-set", help="Create held-out test experiments")
    p.add_argument("--n", type=int, default=20, help="Number of test experiments")
    p.set_defaults(func=test_set.run)

    # analyse
    p = sub.add_parser("analyse", help="Evaluate model on test set + sensitivity analysis")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=analyse.run)

    # inference
    p = sub.add_parser("inference", help="Single-shot first-time-right manufacturing")
    p.add_argument("--design-intent", type=str, default=None,
                   help="JSON: fix parameters for inference. Example: '{\"n_layers\":5}'")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=inference.run)

    # ── Advanced commands ──

    # explore-trajectory
    p = sub.add_parser("explore-trajectory", help="Trajectory exploration: per-layer speed optimization")
    p.add_argument("--n", type=int, default=3, help="Number of trajectory rounds")
    p.add_argument("--kappa", type=float, default=0.5, help="Exploration weight")
    p.add_argument("--delta", type=float, default=5.0, help="Trust region half-width for speed (mm/s)")
    p.add_argument("--smoothing", type=float, default=0.25, help="Smoothing penalty (0=off, 0.3=strong)")
    p.add_argument("--lookahead", type=int, default=2, help="MPC lookahead steps")
    p.add_argument("--discount", type=float, default=0.9, help="MPC discount factor")
    p.add_argument("--design-intent", type=str, default=None, help="JSON: fix parameters")
    p.add_argument("--plot", action="store_true", help="Show plots inline")
    p.set_defaults(func=explore_trajectory.run)

    # adapt
    p = sub.add_parser("adapt", help="Online inference with layer-by-layer adaptation")
    p.add_argument("--delta", type=float, default=5.0, help="Trust region half-width for speed (mm/s)")
    p.add_argument("--design-intent", type=str, default=None, help="JSON: fix parameters")
    p.set_defaults(func=adapt.run)

    # summary
    p = sub.add_parser("summary", help="Show run summary across all phases")
    p.set_defaults(func=summary.run)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
