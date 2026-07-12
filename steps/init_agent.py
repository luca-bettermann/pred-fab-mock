"""Initialize the agent and show its state."""
import argparse

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import load_session, save_session, rebuild, print_phase_banner


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    print_phase_banner("0.2", "Agent")
    agent, _, _ = rebuild(config)
    agent.state_report()
    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the agent")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
