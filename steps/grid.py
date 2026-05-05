"""Generate a static grid dataset (CCF design) and execute every run.

Replicates the Central-Composite-Face design generator used by
``learning-by-printing/studies/grid_designs.py``. The same function
produces both the ADVEI 2026 reference set (22 runs) and test set (45
runs); pass ``--low-pct`` / ``--high-pct`` / ``--fractional-x`` /
``--n-center`` / ``--half-face-centers`` to vary which.

Each run is fabricated and evaluated like any other experiment, tagged
with the chosen ``--dataset-code`` (default: ``grid``) so the splits can
be assembled later via ``DataModule.set_split_dataset``.
"""

from __future__ import annotations

import argparse
import itertools
import sys as _sys
from pathlib import Path
from typing import Sequence

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import load_session, rebuild, save_session, ensure_plot_dir, get_performance, effective_weights
from workflow import with_dimensions, run_and_evaluate
from schema import PARAM_BOUNDS


# ─── CCF design generator (port of learning-by-printing.studies.grid_designs) ─

def _level_value(low_abs: float, high_abs: float, level: int, low_pct: float, high_pct: float) -> float:
    """Encode a coded level (-1 / 0 / +1) as the absolute parameter value."""
    span = high_abs - low_abs
    if level == -1:
        return low_abs + low_pct * span
    if level == 0:
        return low_abs + 0.5 * span
    return low_abs + high_pct * span


def _full_corners(k: int) -> list[tuple[int, ...]]:
    """All 2^k corner points of a k-dim hypercube at ±1."""
    return list(itertools.product((-1, 1), repeat=k))


def _half_fraction_corners_k5(x: int) -> list[tuple[int, ...]]:
    """Half-fraction (Resolution V) for k=5, x=1 → 16 corners.

    Defining relation: I = ABCDE → run only when product of levels = +1.
    """
    if x != 1:
        raise NotImplementedError("Only k=5, x=1 fractional design implemented in mock.")
    out = []
    for corner in _full_corners(5):
        if (corner[0] * corner[1] * corner[2] * corner[3] * corner[4]) == 1:
            out.append(corner)
    return out


def _face_centers(k: int, half: bool) -> list[tuple[int, ...]]:
    """Face-center points: ±1 on one axis, 0 elsewhere.

    Full set: 2k points (both ± per axis).
    Half set: k points (alternating + / − per axis index).
    """
    out = []
    for axis in range(k):
        if half:
            level = 1 if axis % 2 == 0 else -1
            point = [0] * k
            point[axis] = level
            out.append(tuple(point))
        else:
            for level in (-1, 1):
                point = [0] * k
                point[axis] = level
                out.append(tuple(point))
    return out


def generate_ccf_design(
    *,
    bounds: Sequence[tuple[str, float, float]],
    low_pct: float,
    high_pct: float,
    fractional_x: int = 0,
    half_face_centers: bool = False,
    n_center: int = 1,
) -> list[dict[str, float]]:
    """Build a CCF design as a list of param dicts.

    Args mirror ``learning-by-printing.studies.grid_designs.generate_ccf_design``.
    """
    k = len(bounds)
    if fractional_x > 0:
        corners = _half_fraction_corners_k5(fractional_x)
    else:
        corners = _full_corners(k)
    faces = _face_centers(k, half=half_face_centers)
    centers = [(0,) * k] * n_center
    coded_points = corners + faces + centers

    runs: list[dict[str, float]] = []
    for coded in coded_points:
        run = {}
        for (code, lo, hi), level in zip(bounds, coded):
            run[code] = _level_value(lo, hi, level, low_pct, high_pct)
        runs.append(run)
    return runs


# ─── CLI ─────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    ensure_plot_dir()
    perf_weights = effective_weights(config)

    runs = generate_ccf_design(
        bounds=PARAM_BOUNDS,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        fractional_x=args.fractional_x,
        half_face_centers=args.half_face_centers,
        n_center=args.n_center,
    )

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  GRID{_R}{_B} ▸ {args.dataset_code}{_R}")
    print(f"  {_D}{len(runs)}-point CCF (low={args.low_pct}, high={args.high_pct}){_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

    log: list[tuple[str, dict, dict]] = []
    for i, run_params in enumerate(runs):
        code = f"{args.dataset_code}_{i + 1:02d}"
        params = with_dimensions(run_params)
        if dataset.has_experiment(code):
            print(f"  {code}: already exists — skipping")
            continue
        exp_data = run_and_evaluate(
            dataset, agent, fab, params, code, dataset_code=args.dataset_code,
        )
        perf = get_performance(exp_data)
        state.record(args.dataset_code, code, params, perf)
        log.append((code, params, perf))
        score = combined_score(perf, perf_weights or {})
        print(f"  {code:<14s}  combined={score:.3f}")

    if log:
        scores = [combined_score(p, perf_weights or {}) for _, _, p in log]
        print(f"\n  {_D}{'─' * 40}{_R}")
        print(f"  {len(log)} runs  best={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CCF static grid dataset and execute every run",
    )
    parser.add_argument("--dataset-code", type=str, default="grid",
                        help="Tag for the produced experiments (e.g. 'reference', 'test', 'grid')")
    parser.add_argument("--low-pct", type=float, default=0.25,
                        help="Coded -1 level as fraction of param range")
    parser.add_argument("--high-pct", type=float, default=0.75,
                        help="Coded +1 level as fraction of param range")
    parser.add_argument("--fractional-x", type=int, default=0,
                        help="0 = full 2^k corners; 1 = Resolution-V half (k=5 only)")
    parser.add_argument("--half-face-centers", action="store_true",
                        help="Use the alternating-direction half (k face points instead of 2k)")
    parser.add_argument("--n-center", type=int, default=1,
                        help="Number of grand-center replicates")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
