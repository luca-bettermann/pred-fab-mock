"""Shared utilities for evaluation models."""
from __future__ import annotations


def normalize_to_unit(
    value: float, *, low: float, high: float, invert: bool = False,
) -> float:
    """Map `value` from `[low, high]` into `[0, 1]`, clamping outside.

    `invert=True` flips so that `low → 1.0` and `high → 0.0` — useful for
    cost attributes where smaller is better.
    """
    if high <= low:
        return 0.5
    raw = (value - low) / (high - low)
    raw = max(0.0, min(1.0, raw))
    return 1.0 - raw if invert else raw


def proximity_to_target(
    value: float, *, target: float, tolerance_fraction: float = 0.5,
) -> float:
    """Score ∈ [0, 1] — 1.0 when `value == target`, drops linearly with
    distance, hits 0 at `target × tolerance_fraction` deviation.
    """
    if target == 0:
        return 1.0 if value == 0 else 0.0
    deviation = abs(value - target) / (abs(target) * tolerance_fraction)
    return max(0.0, 1.0 - deviation)


def triangular_score(value: float, *, target: float) -> float:
    """Two-segment linear (triangular) score peaked at ``target``.

    Anchors:
      - ``score(0) = 0``
      - ``score(target) = 1``
      - ``score(2 × target) = 0``

    Linear between anchors; clamped to 0 outside ``[0, 2 × target]``.
    Used by StructuralIntegrity where overlap = filament_width is the
    target, no-overlap and full-bead-doubled-overlap are equally bad.
    """
    if target <= 0:
        return 0.0
    if value <= 0 or value >= 2 * target:
        return 0.0
    if value <= target:
        return value / target
    return (2 * target - value) / target
