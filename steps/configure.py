"""Set agent configuration."""
import argparse
import json

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import load_session, save_session, print_config_set


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    if args.weights:
        config["performance_weights"] = json.loads(args.weights)
        print_config_set("Weights", config["performance_weights"])
    if args.bounds:
        config["bounds"] = json.loads(args.bounds)
        print_config_set("Bounds", config["bounds"])
    if args.optimizer:
        config["optimizer"] = args.optimizer
        print_config_set("Optimizer", args.optimizer)
    if args.radius is not None:
        config["exploration_radius"] = args.radius
        print_config_set("Exploration radius", args.radius)
    if args.buffer is not None:
        config["buffer"] = args.buffer
        print_config_set("Buffer", args.buffer)
    if args.decay_exp is not None:
        config["decay_exp"] = args.decay_exp
        print_config_set("Decay exponent", args.decay_exp)
    if args.de_maxiter is not None:
        config["de_maxiter"] = args.de_maxiter
        print_config_set("DE max iterations", args.de_maxiter)
    if args.de_popsize is not None:
        config["de_popsize"] = args.de_popsize
        print_config_set("DE population size", args.de_popsize)
    print()
    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set agent configuration")
    parser.add_argument("--bounds", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--optimizer", choices=["lbfgsb", "de"], default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--buffer", type=float, default=None)
    parser.add_argument("--decay-exp", type=float, default=None)
    parser.add_argument("--de-maxiter", type=int, default=None)
    parser.add_argument("--de-popsize", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
