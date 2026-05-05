"""Terminal pretty-printing for the ADVEI 2026 mock CLI flow."""

import math
from typing import Any, Dict, List, Optional, Tuple

from pred_fab.utils.metrics import combined_score

# ANSI codes
_R  = "\033[0m"   # reset
_B  = "\033[1m"   # bold
_D  = "\033[2m"   # dim
_G  = "\033[32m"  # green
_Y  = "\033[33m"  # yellow
_RD = "\033[31m"  # red
_C  = "\033[36m"  # cyan
_W  = 58          # banner width


def _score_color(v: float) -> str:
    if v >= 0.70:
        return _G
    if v >= 0.45:
        return _Y
    return _RD


def print_phase_header(num: str, title: str, subtitle: str = "") -> None:
    """Print a phase banner to stdout."""
    bar = "━" * _W
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE {num}{_R}{_B} ▸ {title}{_R}")
    if subtitle:
        print(f"  {_D}{subtitle}{_R}")
    print(f"{_B}{_C}{bar}{_R}")


def print_section(title: str) -> None:
    print(f"\n  {_B}▸ {title}{_R}")


def print_experiment_row(
    exp_code: str,
    params: Dict[str, Any],
    perf: Dict[str, float],
    perf_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Print one formatted experiment result row (ADVEI schema)."""
    spd = params.get("print_speed", 0.0)
    sf  = params.get("slowdown_factor", 0.0)
    lh  = params.get("layer_height", 0.0)
    cf  = params.get("calibration_factor", 0.0)
    po  = params.get("path_offset", 0.0)

    meta = f"spd={spd:.4f}  sf={sf:.2f}  lh={lh:.1f}  cf={cf:.2f}  po={po:.1f}"

    perf_parts: List[str] = []
    for key in ["structural_integrity", "material_deposition", "extrusion_stability",
                "energy_footprint", "fabrication_time"]:
        v = perf.get(key, float("nan"))
        if not math.isnan(v):
            short = _perf_short(key)
            perf_parts.append(f"{short}={_score_color(v)}{v:.3f}{_R}")

    comb = combined_score(perf, perf_weights) if perf_weights else _default_combined(perf)
    comb_str = f"comb={_score_color(comb)}{comb:.3f}{_R}"

    print(
        f"  {_B}{exp_code:<14}{_R}"
        f"{_D}{meta}{_R}"
        f"  {' '.join(perf_parts)}  {comb_str}"
    )


def print_explore_row(
    exp_code: str,
    params: Dict[str, Any],
    perf: Dict[str, float],
    perf_weights: Optional[Dict[str, float]] = None,
    kappa: Optional[float] = None,
) -> None:
    """One-line exploration experiment result."""
    spd = params.get("print_speed", 0.0)
    sf  = params.get("slowdown_factor", 0.0)
    lh  = params.get("layer_height", 0.0)

    meta = f"spd={spd:.4f}  sf={sf:.2f}  lh={lh:.1f}"

    comb = combined_score(perf, perf_weights) if perf_weights else _default_combined(perf)
    comb_str = f"comb={_score_color(comb)}{comb:.3f}{_R}"
    kappa_str = f"  κ={kappa:.2f}" if kappa is not None else ""

    perf_parts: List[str] = []
    for key in ["structural_integrity", "material_deposition", "extrusion_stability",
                "energy_footprint", "fabrication_time"]:
        v = perf.get(key, float("nan"))
        if not math.isnan(v):
            short = _perf_short(key)
            perf_parts.append(f"{short}={_score_color(v)}{v:.3f}{_R}")

    print(
        f"  {_B}{exp_code:<14}{_R}"
        f"{_D}{meta}{_R}"
        f"  {' '.join(perf_parts)}  {comb_str}{kappa_str}"
    )


def print_infer_row(
    exp_code: str,
    params: Dict[str, Any],
    perf: Dict[str, float],
    perf_weights: Optional[Dict[str, float]] = None,
) -> None:
    """One-line inference experiment result."""
    print_explore_row(exp_code, params, perf, perf_weights)


def print_training_summary(r2_scores: Dict[str, float]) -> None:
    """Print R² scores for each prediction model output."""
    parts = "  ".join(
        f"{name}: R²={_score_color(max(0.0, r2))}{r2:.3f}{_R}"
        for name, r2 in r2_scores.items()
    )
    print(f"\n  {_B}Model quality{_R}  {parts}")


def print_phase_summary(
    experiments: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
    perf_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Print a one-line best-result summary at the end of a phase."""
    if not experiments:
        return
    best_code, _, best_perf = max(
        experiments,
        key=lambda x: combined_score(x[2], perf_weights) if perf_weights else _default_combined(x[2]),
    )
    comb = combined_score(best_perf, perf_weights) if perf_weights else _default_combined(best_perf)
    perf_parts = "  ".join(
        f"{_perf_short(k)}={v:.3f}"
        for k, v in best_perf.items()
        if not math.isnan(v)
    )
    print(
        f"\n  {_G}✓{_R} Best: {_B}{best_code}{_R} "
        f"— {perf_parts}  comb={_G}{comb:.3f}{_R}"
    )


def print_run_summary(
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]],
    phases: List[str],
    exp_codes: List[str],
    perf_weights: Optional[Dict[str, float]] = None,
) -> None:
    """Print a final table showing best per phase."""
    bar = "─" * _W
    print(f"\n  {_B}Run Summary{_R}")
    print(f"  {_D}{bar}{_R}")
    print(f"  {'Phase':<15s}  {'Experiments':>11s}  {'Best Combined':>14s}")
    print(f"  {_D}{bar}{_R}")

    for phase in ["baseline", "grid", "exploration", "test", "inference"]:
        indices = [i for i, p in enumerate(phases) if p == phase]
        if not indices:
            continue
        scores = [
            combined_score(perf_history[i][1], perf_weights) if perf_weights
            else _default_combined(perf_history[i][1])
            for i in indices
        ]
        best = max(scores)
        print(f"  {phase:<15s}  {len(indices):>11d}  {_score_color(best)}{best:>14.3f}{_R}")

    print(f"  {_D}{bar}{_R}")
    total = len(perf_history)
    print(f"  Total: {total} experiments")
    print()


def print_done() -> None:
    bar = "━" * _W
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  Done.  Plots saved to ./plots/{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")


# ── Internal helpers ─────────────────────────────────────────────────────────

_PERF_SHORT = {
    "structural_integrity": "SI",
    "material_deposition": "MD",
    "extrusion_stability": "ES",
    "energy_footprint": "EF",
    "fabrication_time": "FT",
}


def _perf_short(key: str) -> str:
    return _PERF_SHORT.get(key, key[:4])


def _default_combined(perf: Dict[str, float]) -> float:
    """Equal-weight combined score (fallback when weights not provided)."""
    vals = [v for v in perf.values() if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else 0.0
