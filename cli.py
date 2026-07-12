"""CLI for the PFAB mock — step-by-step predictive fabrication workflow.

Each command delegates to a standalone step module in steps/; the step owns its
argument spec (add_arguments) and can also run directly from the repo root:
python -m steps.baseline --n 5

Quick start:
    # Setup
    uv run cli.py reset
    uv run cli.py init-schema
    uv run cli.py init-agent
    uv run cli.py configure \\
        --weights '{"path_accuracy":2,"energy_efficiency":1,"production_rate":1}' \\
        --schedule print_speed:n_layers \\
        --trust-regions '{"print_speed":5.0}'
    uv run cli.py init-physics --seed 42 --plot

    # System discovery
    uv run cli.py baseline --n 5 --plot
    uv run cli.py report baseline_01 --plot
    uv run cli.py explore --n 5 --kappa 0.5 --design-intent '{"n_layers":5}'
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

# Command registry: each step module owns its description (module docstring)
# and argument spec (add_arguments); the CLI only wires them to subcommands.
COMMANDS = [
    ("reset", reset),
    ("init-schema", init_schema),
    ("init-agent", init_agent),
    ("init-physics", init_physics),
    ("configure", configure),
    ("baseline", baseline),
    ("explore", explore),
    ("analyse", analyse),
    ("inference", inference),
    ("adapt", adapt),
    ("summary", summary),
    ("report", report),
]


def _wide_formatter(prog: str) -> argparse.HelpFormatter:
    """Argparse formatter with a wider help column to avoid mid-sentence wraps."""
    return argparse.HelpFormatter(prog, max_help_position=36, width=110)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfab-mock",
        description="PFAB mock CLI — predictive fabrication workflow step by step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module in COMMANDS:
        help_line = (module.__doc__ or "").strip().splitlines()[0].rstrip(".")
        p = sub.add_parser(name, help=help_line, formatter_class=_wide_formatter)
        module.add_arguments(p)
        p.set_defaults(func=module.run)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
