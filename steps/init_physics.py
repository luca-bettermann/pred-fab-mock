"""Randomize physics constants and show the ground truth topology."""
import argparse
import os

from pred_fab.plotting import plot_metric_topology
from visualization.helpers import evaluate_physics_grid
from steps._common import (
    run_step,
    load_session, save_session, ensure_plot_dir, show_plot_with_header,
    randomize_physics, apply_physics_config, PHYSICS_CONFIG_KEY,
    X_AXIS, Y_AXIS, FIXED_DIMS, print_phase_banner,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    plot_dir = ensure_plot_dir()
    seed_str = f"seed={args.seed}" if args.seed is not None else "random"
    print_phase_banner("0.3", "Physics", f"Randomize ground truth ({seed_str})")
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
                          fixed_params=FIXED_DIMS)
    show_plot_with_header(path, "Physics: Ground Truth Topology", inline=args.plot)
    save_session(config, state)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", help="Show plots inline in terminal")


if __name__ == "__main__":
    run_step(__doc__, add_arguments, run)
