"""Randomize physics constants and show the ground truth topology."""
import argparse
import os

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import (
    load_session, save_session, ensure_plot_dir, show_plot,
    randomize_physics, apply_physics_config, PHYSICS_CONFIG_KEY,
    X_AXIS, Y_AXIS, FIXED_DIMS,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    plot_dir = ensure_plot_dir()
    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "\u2501" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 0.3{_R}{_B} \u25b8 Physics{_R}")
    seed_str = f"seed={args.seed}" if args.seed is not None else "random"
    print(f"  {_D}Randomize ground truth ({seed_str}){_R}")
    print(f"{_B}{_C}{bar}{_R}")
    seed = args.seed
    physics = randomize_physics(seed)
    config[PHYSICS_CONFIG_KEY] = physics
    apply_physics_config(physics)
    print(f"\n  Physics constants:")
    for key, val in physics.items():
        if isinstance(val, list):
            print(f"    {key:<25s} = [{', '.join(f'{v:.3f}' for v in val)}]")
        else:
            print(f"    {key:<25s} = {val:.6f}")

    from pred_fab.plotting import plot_metric_topology
    from visualization.helpers import evaluate_physics_grid

    perf_weights = config.get("performance_weights")
    waters, speeds, metrics = evaluate_physics_grid(50, perf_weights)
    # Separate individual metrics from combined
    metric_names = list(metrics.keys())
    individual = {k: metrics[k] for k in metric_names[:-1]}
    combined = metrics[metric_names[-1]]

    path = os.path.join(plot_dir, "00_physics_topology.png")
    plot_metric_topology(path, X_AXIS, Y_AXIS, waters, speeds,
                          individual, combined,
                          combined_label=metric_names[-1],
                          weights=perf_weights,
                          title="Physics Performance Topology",
                          fixed_params=FIXED_DIMS)
    show_plot(path, inline=args.plot)
    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomize physics constants and show topology")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
