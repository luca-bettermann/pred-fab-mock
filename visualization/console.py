"""Terminal pretty-printing for the PFAB showcase flow."""

import math
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

    acc  = perf.get("path_accuracy",    float("nan"))
    eff  = perf.get("energy_efficiency", float("nan"))
    rate = perf.get("production_rate",   float("nan"))

    acc_str  = f"{_score_color(acc)}{acc:.3f}{_R}"
    eff_str  = f"{_score_color(eff)}{eff:.3f}{_R}"

    meta = f"design={design}  mat={material}  w={water:.2f}  spd={speed:.1f}"
    rate_part = ""
    if not math.isnan(rate):
        rate_str = f"{_score_color(rate)}{rate:.3f}{_R}"
        rate_part = f"  rate={rate_str}"
    print(
        f"  {_B}{exp_code:<14}{_R}"
        f"{_D}{meta}{_R}"
        f"  acc={acc_str}  eff={eff_str}{rate_part}"
    )


def print_explore_row(
    exp_code: str,
    params: Dict[str, Any],
    perf: Dict[str, float],
    u: float,
    obj: float,
    w_explore: float,
) -> None:
    """One-line exploration experiment result: params (dim) + perf scores + u + obj."""
    design   = params.get("design",      "?")
    material = str(params.get("material", "?"))[:3]
    water    = params.get("water_ratio",  0.0)
    speed    = params.get("print_speed",  0.0)

    acc  = perf.get("path_accuracy",    float("nan"))
    eff  = perf.get("energy_efficiency", float("nan"))
    rate = perf.get("production_rate",   float("nan"))

    meta   = f"design={design}  mat={material}  w={water:.2f}  spd={speed:.1f}"
    perf_s = (
        f"acc={_score_color(acc)}{acc:.3f}{_R}  "
        f"eff={_score_color(eff)}{eff:.3f}{_R}  "
        f"rate={_score_color(rate)}{rate:.3f}{_R}"
    )
    u_s   = f"{_score_color(u)}{u:.3f}{_R}"
    obj_s = f"{_score_color(obj)}{obj:.3f}{_R}"
    print(
        f"  {_B}{exp_code:<14}{_R}"
        f"{_D}{meta}{_R}"
        f"  {perf_s}  "
        f"u={u_s}  obj={obj_s}"
    )


def print_infer_row(
    exp_code: str,
    params: Dict[str, Any],
    perf: Dict[str, float],
    obj: float,
) -> None:
    """One-line inference experiment result: params (dim) + perf scores + obj."""
    design   = params.get("design",      "?")
    material = str(params.get("material", "?"))[:3]
    water    = params.get("water_ratio",  0.0)
    speed    = params.get("print_speed",  0.0)

    acc  = perf.get("path_accuracy",    float("nan"))
    eff  = perf.get("energy_efficiency", float("nan"))
    rate = perf.get("production_rate",   float("nan"))

    meta   = f"design={design}  mat={material}  w={water:.2f}  spd={speed:.1f}"
    perf_s = (
        f"acc={_score_color(acc)}{acc:.3f}{_R}  "
        f"eff={_score_color(eff)}{eff:.3f}{_R}  "
        f"rate={_score_color(rate)}{rate:.3f}{_R}"
    )
    obj_s = f"{_score_color(obj)}{obj:.3f}{_R}"
    print(
        f"  {_B}{exp_code:<14}{_R}"
        f"{_D}{meta}{_R}"
        f"  {perf_s}  "
        f"obj={obj_s}"
    )


def print_optimizer_row(n_starts: int, n_evals: int) -> None:
    """Dim optimizer summary line."""
    print(f"  {_D}    {n_starts} starts · {n_evals} evals{_R}")


def _combined_score(perf: Dict[str, float]) -> float:
    """2:1:1 weighted combined score across path_accuracy, energy_efficiency, production_rate."""
    return (
        2 * perf.get("path_accuracy",    0.0)
        +   perf.get("energy_efficiency", 0.0)
        +   perf.get("production_rate",   0.0)
    ) / 4


def print_phase_summary(
    experiments: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
) -> None:
    """Print a one-line best-result summary at the end of a phase."""
    if not experiments:
        return
    best_code, best_params, best_perf = max(
        experiments,
        key=lambda x: _combined_score(x[2]),
    )
    acc  = best_perf.get("path_accuracy",    float("nan"))
    eff  = best_perf.get("energy_efficiency", float("nan"))
    rate = best_perf.get("production_rate",   float("nan"))
    design   = best_params.get("design",   "?")
    material = best_params.get("material", "?")
    rate_part = f"  rate={rate:.3f}" if not math.isnan(rate) else ""
    print(
        f"\n  {_G}✓{_R} Best: {_B}{best_code}{_R} "
        f"— acc={_G}{acc:.3f}{_R}  eff={eff:.3f}{rate_part}  "
        f"{_D}(design={design}, {material}){_R}"
    )


def print_training_summary(r2_scores: Dict[str, float]) -> None:
    """Print R² scores for each prediction model output."""
    parts = "  ".join(
        f"{name}: R²={_score_color(max(0.0, r2))}{r2:.3f}{_R}"
        for name, r2 in r2_scores.items()
    )
    print(f"\n  {_B}Model quality{_R}  {parts}")


def print_adaptation_row(
    layer_idx: int,
    speed_before: float,
    deviation: float,
    speed_after: Optional[float] = None,
    n_evals: Optional[int] = None,
) -> None:
    """Print one layer's adaptation step result."""
    dev_color = _score_color(max(0.0, 1.0 - deviation / 0.003))
    dev_str   = f"{dev_color}{deviation:.5f}{_R}"

    evals_str = f"  {_D}({n_evals} evals){_R}" if n_evals is not None else ""

    if speed_after is not None:
        spd_str = f"{speed_before:.1f} → {_B}{speed_after:.1f}{_R} mm/s"
        print(f"  Layer {layer_idx}  │  speed={spd_str}  │  dev={dev_str}{evals_str}")
    else:
        # Last layer — no adaptation, align columns with a fixed-width speed field
        spd_str = f"{speed_before:.1f} mm/s{' ' * 13}"
        print(f"  Layer {layer_idx}  │  speed={spd_str}  │  dev={dev_str}")


def print_run_summary(
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]],
    phases: List[str],
    exp_codes: List[str],
    design_intent: Dict[str, Any],
    phys_opt_speed: float,
    phys_opt_water: float,
) -> None:
    """Print a final table comparing best found params against the physics optimum."""
    # Best overall and best inference
    all_scored = [(code, params, perf, phase)
                  for (params, perf), code, phase
                  in zip(perf_history, exp_codes, phases)]
    best        = max(all_scored, key=lambda x: _combined_score(x[2]))
    infer_items = [x for x in all_scored if x[3] == "inference"]
    best_infer  = max(infer_items, key=lambda x: _combined_score(x[2])) if infer_items else None

    bar = "─" * _W
    print(f"\n  {_B}Run Summary{_R}")
    print(f"  {_D}{bar}{_R}")

    header = f"  {'Experiment':<16} {'design':<8} {'material':<11} {'w':>5} {'spd':>6}  {'acc':>6}  {'eff':>6}  {'rate':>6}  {'combined':>8}"
    print(f"  {_D}{header}{_R}")
    print(f"  {_D}{bar}{_R}")

    def _row(label: str, code: str, params: Dict[str, Any], perf: Dict[str, float]) -> None:
        acc  = perf.get("path_accuracy",    float("nan"))
        eff  = perf.get("energy_efficiency", float("nan"))
        rate = perf.get("production_rate",   float("nan"))
        comb = _combined_score(perf)
        w    = params.get("water_ratio", float("nan"))
        spd  = params.get("print_speed",  float("nan"))
        des  = str(params.get("design",   "?"))
        mat  = str(params.get("material", "?"))
        rate_s = f"{_score_color(rate)}{rate:>6.3f}{_R}" if not math.isnan(rate) else f"{'n/a':>6}"
        print(
            f"  {_B}{label:<16}{_R}"
            f"{des:<8}{mat:<11}"
            f"{w:>5.2f} {spd:>6.1f}  "
            f"{_score_color(acc)}{acc:>6.3f}{_R}  "
            f"{_score_color(eff)}{eff:>6.3f}{_R}  "
            f"{rate_s}  "
            f"{_score_color(comb)}{comb:>8.3f}{_R}"
        )

    _row("Best overall", best[0], best[1], best[2])
    if best_infer:
        _row("Best inference", best_infer[0], best_infer[1], best_infer[2])

    # Physics optimum row (no perf scores — computed analytically)
    des_i = str(design_intent.get("design",   "?"))
    mat_i = str(design_intent.get("material", "?"))
    print(
        f"  {_B}{'Physics optimum':<16}{_R}"
        f"{des_i:<8}{mat_i:<11}"
        f"{phys_opt_water:>5.2f} {phys_opt_speed:>6.1f}  "
        f"{_D}(theoretical minimum deviation){_R}"
    )
    print(f"  {_D}{bar}{_R}")


def print_done() -> None:
    bar = "━" * _W
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  Done.  Plots saved to ./plots/{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")
