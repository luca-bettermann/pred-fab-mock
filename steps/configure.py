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
    if args.radius is not None:
        print_config_set("Exploration radius", config.get("exploration_radius"), args.radius)
        config["exploration_radius"] = args.radius
    if getattr(args, "sigma", None) is not None:
        print_config_set("Sigma override", config.get("sigma"), args.sigma)
        config["sigma"] = args.sigma
    if getattr(args, "mc_exp_offset", None) is not None:
        print_config_set("MC exponent offset", config.get("mc_exponent_offset"), args.mc_exp_offset)
        config["mc_exponent_offset"] = args.mc_exp_offset
    if args.de_maxiter is not None:
        print_config_set("DE max iterations", config.get("de_maxiter"), args.de_maxiter)
        config["de_maxiter"] = args.de_maxiter
    if args.de_popsize is not None:
        print_config_set("DE population size", config.get("de_popsize"), args.de_popsize)
        config["de_popsize"] = args.de_popsize
    if getattr(args, "smoothing", None) is not None:
        print_config_set("Smoothing", config.get("schedule_smoothing"), args.smoothing)
        config["schedule_smoothing"] = args.smoothing
    print()
    save_session(config, state)


if __name__ == "__main__":
    raise SystemExit("Run via cli.py: uv run cli.py configure ...")
