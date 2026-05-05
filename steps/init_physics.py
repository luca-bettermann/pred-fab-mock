"""Show the ADVEI physics ground truth and optionally plot the topology."""

import argparse
import os

from steps._common import (
    load_session, save_session, ensure_plot_dir, show_plot_with_header,
    effective_weights, SPEED_AXIS, CALIB_AXIS, DEFAULT_FIXED,
)
from sensors.physics import (
    COMPONENT_HEIGHT_MM, MAX_N_LAYERS, PATH_LENGTH_PER_LAYER_M,
    TARGET_FILAMENT_WIDTH_MM, TARGET_NODE_OVERLAP_MM,
    ROBOT_NOMINAL_POWER_W,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    plot_dir = ensure_plot_dir()

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 0.3{_R}{_B} ▸ Physics{_R}")
    print(f"  {_D}ADVEI 2026 ground truth (deterministic){_R}")
    print(f"{_B}{_C}{bar}{_R}")

    print(f"\n  Physics constants:")
    print(f"    {'component_height':<28s} = {COMPONENT_HEIGHT_MM} mm")
    print(f"    {'max_n_layers':<28s} = {MAX_N_LAYERS}")
    print(f"    {'n_nodes':<28s} = 7")
    print(f"    {'path_length_per_layer':<28s} = {PATH_LENGTH_PER_LAYER_M} m")
    print(f"    {'target_node_overlap':<28s} = {TARGET_NODE_OVERLAP_MM} mm")
    print(f"    {'target_filament_width':<28s} = {TARGET_FILAMENT_WIDTH_MM} mm")
    print(f"    {'robot_nominal_power':<28s} = {ROBOT_NOMINAL_POWER_W} W")

    if args.plot:
        from visualization.helpers import evaluate_physics_grid
        from pred_fab.plotting import plot_metric_topology

        perf_weights = effective_weights(config)
        speeds, calibs, metrics = evaluate_physics_grid(20, perf_weights)
        individual = {k: v for k, v in metrics.items() if k != "combined"}

        path = os.path.join(plot_dir, "00_physics_topology.png")
        plot_metric_topology(
            path, SPEED_AXIS, CALIB_AXIS, speeds, calibs,
            individual, metrics["combined"],
            combined_label="combined",
            weights=perf_weights,
            fixed_params={k: v for k, v in DEFAULT_FIXED.items()
                          if k not in ("print_speed", "calibration_factor")},
        )
        show_plot_with_header(path, "Physics: Ground Truth Topology", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show ADVEI physics topology")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
