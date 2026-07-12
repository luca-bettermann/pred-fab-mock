"""Set agent configuration."""
import argparse
import json

from steps._common import load_session, save_session, print_config_set, print_config_show, run_step


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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--show", action="store_true",
                        help="Show all current configuration values")
    parser.add_argument("--weights", type=str, default=None, metavar="JSON",
                        help='Performance weights — e.g. \'{"path_accuracy":2,"energy_efficiency":1}\'')
    parser.add_argument("--bounds", type=str, default=None, metavar="JSON",
                        help='Parameter bounds — e.g. \'{"water_ratio":[0.35,0.45]}\'')
    parser.add_argument("--trust-regions", type=str, default=None, metavar="JSON", dest="trust_regions",
                        help='Per-param max change per step (used by both schedule and adaptation). '
                             'Default = bounds_span / 10. Override e.g. \'{"print_speed":5.0}\'')
    parser.add_argument("--schedule", action="append", metavar="PARAM:DIM",
                        help="Default schedule (e.g. print_speed:n_layers). Repeatable. "
                             "Per-command --schedule overrides.")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Evidence kernel σ (per normalized dimension)")
    parser.add_argument("--kappa", type=float, default=None,
                        help="Default exploration weight κ (0=exploit, 1=explore)")
    parser.add_argument("--n-starts", type=int, default=None, dest="n_starts",
                        help="Optimizer multi-start count")
    parser.add_argument("--n-sobol", type=int, default=None, dest="n_sobol",
                        help="Optimizer Sobol candidate count")
    parser.add_argument("--lr", type=float, default=None,
                        help="Optimizer learning rate")


if __name__ == "__main__":
    run_step(__doc__, add_arguments, run)
