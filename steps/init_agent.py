"""Initialize the agent and show its state."""
import argparse

from steps._common import load_session, save_session, rebuild, print_phase_banner, run_step


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    print_phase_banner("0.2", "Agent")
    agent, _, _ = rebuild(config)
    agent.state_report()
    save_session(config, state)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """No arguments."""


if __name__ == "__main__":
    run_step(__doc__, add_arguments, run)
