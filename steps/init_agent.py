"""Initialize the agent and show its state."""
import argparse

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import load_session, save_session, rebuild


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 0.2{_R}{_B} ▸ Agent{_R}")
    print(f"{_B}{_C}{bar}{_R}")
    agent, _, _ = rebuild(config)
    agent.state_report()
    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the agent")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
