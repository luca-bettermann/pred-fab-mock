"""Clear all session state, data, and plots."""
import argparse
import os
import shutil

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import SESSION_FILE


def run(args: argparse.Namespace) -> None:
    for path in [SESSION_FILE, "./local", "./plots", "./logs"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  Removed {path}")
    print("  Session reset.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear all session state and data")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
