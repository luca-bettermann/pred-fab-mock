"""Configure agent settings (weights, exploration, optimizer, trust regions)."""
import argparse
import json
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps._common import load_session, save_session


def run(args: argparse.Namespace) -> None:
    config, state = load_session()

    if args.weights:
        config["performance_weights"] = json.loads(args.weights)
    if args.radius is not None:
        config["exploration_radius"] = float(args.radius)
    if args.sigma is not None:
        config["sigma"] = float(args.sigma)
    if args.trust_regions:
        config["trust_regions"] = json.loads(args.trust_regions)
    if args.schedule:
        config["default_schedule"] = list(args.schedule)
    if args.de_maxiter is not None:
        config["de_maxiter"] = int(args.de_maxiter)
    if args.de_popsize is not None:
        config["de_popsize"] = int(args.de_popsize)

    print("\n  Configuration updated.")
    for key in [
        "performance_weights", "exploration_radius", "sigma",
        "trust_regions", "default_schedule", "de_maxiter", "de_popsize",
    ]:
        if key in config and config[key] is not None:
            print(f"    {key}: {config[key]}")

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure agent settings")
    parser.add_argument("--weights", type=str, default=None,
                        help='JSON map: code → weight, e.g. \'{"structural_integrity":2,"fabrication_time":1}\'')
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--trust-regions", type=str, default=None,
                        help="JSON map of per-runtime-param trust deltas")
    parser.add_argument("--schedule", action="append", default=[],
                        help="Default trajectory schedules; repeatable PARAM:DIM")
    parser.add_argument("--de-maxiter", type=int, default=None)
    parser.add_argument("--de-popsize", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
