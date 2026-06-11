"""Set agent configuration."""
import argparse
import json

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import load_session, save_session, print_config_set, print_config_show


def run(args: argparse.Namespace) -> None:
    config, state = load_session()

    if args.show:
        print_config_show(config)
        return

    if args.weights:
        new = json.loads(args.weights)
        print_config_set("Weights", config.get("performance_weights"), new)
        config["performance_weights"] = new
    if args.bounds:
        new = json.loads(args.bounds)
        print_config_set("Bounds", config.get("bounds"), new)
        config["bounds"] = new
    if getattr(args, "trust_regions", None):
        new = json.loads(args.trust_regions)
        print_config_set("Trust regions", config.get("trust_regions"), new)
        config["trust_regions"] = new
    if getattr(args, "schedule", None):
        print_config_set("Default schedule", config.get("default_schedule"), args.schedule)
        config["default_schedule"] = list(args.schedule)
    if getattr(args, "sigma", None) is not None:
        print_config_set("Sigma", config.get("sigma"), args.sigma)
        config["sigma"] = args.sigma
    if getattr(args, "kappa", None) is not None:
        print_config_set("Kappa default", config.get("kappa"), args.kappa)
        config["kappa"] = args.kappa
    if getattr(args, "n_starts", None) is not None:
        print_config_set("Multi-start count", config.get("n_starts"), args.n_starts)
        config["n_starts"] = args.n_starts
    if getattr(args, "n_sobol", None) is not None:
        print_config_set("Sobol candidates", config.get("n_sobol"), args.n_sobol)
        config["n_sobol"] = args.n_sobol
    if getattr(args, "lr", None) is not None:
        print_config_set("Learning rate", config.get("lr"), args.lr)
        config["lr"] = args.lr
    print()
    save_session(config, state)


if __name__ == "__main__":
    raise SystemExit("Run via cli.py: uv run cli.py configure ...")
