"""Clear all session state, data, and plots."""
import argparse
import os
import shutil

from schema import LOCAL_DIR, PLOT_DIR, LOGS_DIR
from steps._common import SESSION_FILE, run_step


def run(args: argparse.Namespace) -> None:
    for path in [SESSION_FILE, LOCAL_DIR, PLOT_DIR, LOGS_DIR]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  Removed {path}")
    print("  Session reset.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """No arguments."""


if __name__ == "__main__":
    run_step(__doc__, add_arguments, run)
