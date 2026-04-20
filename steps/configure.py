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
    if args.optimizer:
        print_config_set("Optimizer", config.get("optimizer"), args.optimizer)
        config["optimizer"] = args.optimizer
    if args.radius is not None:
        print_config_set("Exploration radius", config.get("exploration_radius"), args.radius)
        config["exploration_radius"] = args.radius
    if args.buffer is not None:
        print_config_set("Buffer", config.get("buffer"), args.buffer)
        config["buffer"] = args.buffer
    if args.decay_exp is not None:
        print_config_set("Decay exponent", config.get("decay_exp"), args.decay_exp)
        config["decay_exp"] = args.decay_exp
    if args.de_maxiter is not None:
        print_config_set("DE max iterations", config.get("de_maxiter"), args.de_maxiter)
        config["de_maxiter"] = args.de_maxiter
    if args.de_popsize is not None:
        print_config_set("DE population size", config.get("de_popsize"), args.de_popsize)
        config["de_popsize"] = args.de_popsize
    if getattr(args, 'schedule_smoothing', None) is not None:
        print_config_set("Schedule smoothing", config.get("schedule_smoothing"), args.schedule_smoothing)
        config["schedule_smoothing"] = args.schedule_smoothing
    if getattr(args, 'schedule_delta', None) is not None:
        print_config_set("Schedule delta", config.get("schedule_delta"), args.schedule_delta)
        config["schedule_delta"] = args.schedule_delta
    print()
    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set agent configuration")
    parser.add_argument("--show", action="store_true", help="Show all current configuration values")
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
