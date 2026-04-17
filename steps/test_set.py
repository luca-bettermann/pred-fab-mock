"""Create a held-out test set for model evaluation."""
import argparse

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import (
    load_session, save_session, rebuild, load_physics_from_session,
    generate_test_params, with_dimensions, run_and_evaluate,
)


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    load_physics_from_session(config)

    test_params = generate_test_params(args.n)
    print(f"\n  Creating {len(test_params)} test experiments...")

    for i, params in enumerate(test_params):
        exp_code = f"test_{i+1:02d}"
        if dataset.has_experiment(exp_code):
            continue
        params = with_dimensions(params)
        run_and_evaluate(dataset, agent, fab, params, exp_code)

    print(f"  Test set: {len(test_params)} experiments (test_01..test_{len(test_params):02d})")
    config["test_set_n"] = len(test_params)
    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create held-out test experiments")
    parser.add_argument("--n", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
