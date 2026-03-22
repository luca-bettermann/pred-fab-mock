"""Terminal pretty-printing for the PFAB showcase flow."""

from typing import Dict, Any, List, Optional, Tuple

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


def print_phase_header(num: int, title: str, subtitle: str = "") -> None:
    """Print a phase banner to stdout."""
    bar = "━" * _W
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE {num}{_R}{_B} ▸ {title}{_R}")
    if subtitle:
        print(f"  {_D}{subtitle}{_R}")
    print(f"{_B}{_C}{bar}{_R}")


def print_section(title: str) -> None:
    """Print a lightweight section label (for sub-steps)."""
    print(f"\n  {_B}▸ {title}{_R}")


def print_experiment_row(
    exp_code: str,
    params: Dict[str, Any],
    perf: Dict[str, float],
) -> None:
    """Print one formatted experiment result row."""
    design   = params.get("design",      "?")
    material = str(params.get("material", "?"))[:3]
    water    = params.get("water_ratio",  0.0)
    speed    = params.get("print_speed",  0.0)

    acc = perf.get("path_accuracy",    float("nan"))
    eff = perf.get("energy_efficiency", float("nan"))

    acc_str = f"{_score_color(acc)}{acc:.3f}{_R}"
    eff_str = f"{_score_color(eff)}{eff:.3f}{_R}"

    meta = f"design={design}  mat={material}  w={water:.2f}  spd={speed:.1f}"
    print(
        f"  {_B}{exp_code:<14}{_R}"
        f"{_D}{meta}{_R}"
        f"  acc={acc_str}  eff={eff_str}"
    )


def print_phase_summary(
    experiments: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
) -> None:
    """Print a one-line best-result summary at the end of a phase."""
    if not experiments:
        return
    best_code, best_params, best_perf = max(
        experiments, key=lambda x: x[2].get("path_accuracy", 0.0)
    )
    acc = best_perf.get("path_accuracy",    float("nan"))
    eff = best_perf.get("energy_efficiency", float("nan"))
    design   = best_params.get("design",   "?")
    material = best_params.get("material", "?")
    print(
        f"\n  {_G}✓{_R} Best: {_B}{best_code}{_R} "
        f"— acc={_G}{acc:.3f}{_R}  eff={eff:.3f}  "
        f"{_D}(design={design}, {material}){_R}"
    )


def print_adaptation_row(
    layer_idx: int,
    speed_before: float,
    deviation: float,
    speed_after: Optional[float] = None,
) -> None:
    """Print one layer's adaptation step result."""
    dev_color = _score_color(max(0.0, 1.0 - deviation / 0.003))
    dev_str   = f"{dev_color}{deviation:.5f}{_R}"

    if speed_after is not None:
        spd_str = f"{speed_before:.1f} → {_B}{speed_after:.1f}{_R}"
    else:
        spd_str = f"{speed_before:.1f}"

    print(f"  Layer {layer_idx}  │  speed={spd_str}  │  dev={dev_str}")


def print_done() -> None:
    bar = "━" * _W
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  Done.  Plots saved to ./plots/{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")
