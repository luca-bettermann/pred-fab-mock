"""CLI entry point for the ADVEI 2026 mock — drives the PFAB loop on synthetic fabrication.

Run from the repo root:  python -m cli.main <command>
"""
import argparse

from cli import commands

_EPILOG = """\
Quick start:
  python -m cli.main reset
  python -m cli.main discovery --n 18
  python -m cli.main train
  python -m cli.main exploration --kappa 0.5
  python -m cli.main inference
  python -m cli.main summary
"""


def _add_verbose(p: argparse.ArgumentParser) -> None:
    p.add_argument("--verbose", action="store_true", help="Show systems-init output")


def _add_plot(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plot", action="store_true", help="Render showcase plots inline + save to plots/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mock",
        description="ADVEI 2026 — pred-fab-mock CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("discovery", help="Propose + simulate N space-filling experiments (κ=1)")
    p.add_argument("--n", type=int, default=18, help="Number of experiments")
    _add_plot(p)
    _add_verbose(p)
    p.set_defaults(func=commands.discovery)

    p = sub.add_parser("train", help="Train StructuralMLP on collected experiments")
    _add_verbose(p)
    p.set_defaults(func=commands.train)

    p = sub.add_parser("exploration", help="Propose + simulate one exploration experiment")
    p.add_argument("--kappa", type=float, default=None, help="Exploration weight (default: session config)")
    _add_plot(p)
    _add_verbose(p)
    p.set_defaults(func=commands.exploration)

    p = sub.add_parser("inference", help="Propose the predicted-optimal parameters (κ=0)")
    _add_plot(p)
    _add_verbose(p)
    p.set_defaults(func=commands.inference)

    p = sub.add_parser("report", help="Train + render showcase plots (acquisition topology + radar)")
    p.set_defaults(func=commands.report)

    p = sub.add_parser("configure", help="Set + persist session config (kappa, seed)")
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=commands.configure)

    p = sub.add_parser("summary", help="Show session status")
    p.set_defaults(func=commands.summary)

    p = sub.add_parser("reset", help="Clear session + local data")
    p.set_defaults(func=commands.reset)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
