"""Generate a held-out test dataset (sub-CCF design) — convenience wrapper."""
import argparse
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps import grid as _grid


def run(args: argparse.Namespace) -> None:
    """Pipe through to grid.run with ADVEI's test-set defaults."""
    grid_args = argparse.Namespace(
        dataset_code="test",
        low_pct=0.15,
        high_pct=0.85,
        fractional_x=0,
        half_face_centers=False,
        n_center=args.n_center,
    )
    _grid.run(grid_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the ADVEI test dataset (45-run CCF)")
    parser.add_argument("--n-center", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
