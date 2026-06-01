"""Central-Composite-Face (CCF) static-grid generator.

Ports learning-by-printing's grid design (``studies/grid_designs.py``): corner
points (full 2^k factorial or a Resolution-V half-fraction) + face-center points
(full 2k or an alternating-direction half) + grand-center replicates. Coded
levels (-1 / 0 / +1) map to absolute parameter values via ``low_pct`` / ``high_pct``
of each parameter's range.

This produces the **passive** comparison datasets — the fixed ``reference`` grid
and held-out ``test`` grid (no model, no active learning) — that the active
discovery/exploration loop is measured against. Generalised to any k (the
learning-by-printing original hard-coded k=5; the mock has 4 parameters).
"""
from __future__ import annotations

import itertools
from typing import Sequence


def _level_value(low: float, high: float, level: int, low_pct: float, high_pct: float) -> float:
    """Encode a coded level (-1 / 0 / +1) as an absolute parameter value."""
    span = high - low
    if level == -1:
        return low + low_pct * span
    if level == 0:
        return low + 0.5 * span
    return low + high_pct * span


def _full_corners(k: int) -> list[tuple[int, ...]]:
    """All 2^k corner points of a k-dim hypercube at ±1."""
    return list(itertools.product((-1, 1), repeat=k))


def _half_fraction_corners(k: int) -> list[tuple[int, ...]]:
    """Half-fraction corners (defining relation I = product of all factors = +1) → 2^(k-1)."""
    out = []
    for corner in _full_corners(k):
        prod = 1
        for v in corner:
            prod *= v
        if prod == 1:
            out.append(corner)
    return out


def _face_centers(k: int, half: bool) -> list[tuple[int, ...]]:
    """Face-center points: ±1 on one axis, 0 elsewhere. Full = 2k; half = k (alternating)."""
    out = []
    for axis in range(k):
        levels = [1 if axis % 2 == 0 else -1] if half else [-1, 1]
        for level in levels:
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
    """Build a CCF design as a list of ``{param_code: value}`` dicts."""
    k = len(bounds)
    corners = _half_fraction_corners(k) if fractional_x > 0 else _full_corners(k)
    faces = _face_centers(k, half=half_face_centers)
    centers = [(0,) * k] * n_center

    runs: list[dict[str, float]] = []
    for coded in corners + faces + centers:
        runs.append({
            code: _level_value(lo, hi, level, low_pct, high_pct)
            for (code, lo, hi), level in zip(bounds, coded)
        })
    return runs
