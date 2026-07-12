"""CLI helper functions: inline plot display, physics randomization, test set generation."""

import base64
import sys
from typing import Any

import numpy as np

import sensors.physics as phys
from sensors.physics import N_LAYERS, N_SEGMENTS, DEFAULT_SEGMENT_CURVATURE
from pred_fab.utils.metrics import combined_score
from schema import WATER_RATIO_BOUNDS, PRINT_SPEED_BOUNDS


# ── Inline plot display ───────────────────────────────────────────────────────

def show_plot(path: str, inline: bool = True) -> None:
    """Display a plot: save path always printed, inline image if requested.

    Uses the iTerm2 inline image protocol (works in iTerm2, WezTerm, Kitty).
    Install iTerm2: ``brew install --cask iterm2``
    """
    if not inline:
        print(f"  Plot: {path}")
        return

    print()
    with open(path, "rb") as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode("ascii")
    sys.stdout.write(f"\033]1337;File=inline=1;size={len(img_data)}:{b64}\a\n")
    sys.stdout.flush()
    print(f"  Plot: {path}")


# ── Physics randomization ─────────────────────────────────────────────────────

PHYSICS_CONFIG_KEY = "physics_config"

# Randomization ranges (fraction of default value, or absolute bounds)
PHYSICS_RANGES = {
    "W_OPTIMAL": (0.36, 0.46),
    "W_ENERGY_OPT": (0.33, 0.43),
    "DELTA": (0.000005, 0.000020),
    "THETA": (0.006, 0.018),
    "SAG": (1.0, 2.5),
    "COMPLEXITY": (0.6, 1.5),
    "LAYER_SPD_SHIFT": (0.2, 0.8),
    "W_SLIP": (0.40, 0.48),
}


def randomize_physics(seed: int | None = None) -> dict[str, Any]:
    """Generate randomized physics constants within sensible ranges."""
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {}

    for key, (lo, hi) in PHYSICS_RANGES.items():
        config[key] = float(rng.uniform(lo, hi))

    # Randomize segment curvature: shuffle and slightly perturb
    base_curv = np.array(DEFAULT_SEGMENT_CURVATURE)
    perturb = rng.uniform(-0.05, 0.05, size=4)
    config["SEGMENT_CURVATURE"] = (base_curv + perturb).tolist()
    rng.shuffle(config["SEGMENT_CURVATURE"])

    return config


def apply_physics_config(config: dict[str, Any]) -> None:
    """Apply physics config to the sensors.physics module at runtime."""
    known_keys = set(PHYSICS_RANGES) | {"SEGMENT_CURVATURE"}
    for key, val in config.items():
        if key == "SEGMENT_CURVATURE":
            phys.SEGMENT_CURVATURE = list(val)
        elif key in known_keys:
            setattr(phys, key, val)
        else:
            print(f"  Warning: unknown physics key '{key}' ignored")


def load_physics_from_session(session_config: dict[str, Any]) -> None:
    """Load and apply physics config from session if present."""
    physics = session_config.get(PHYSICS_CONFIG_KEY)
    if physics:
        apply_physics_config(physics)


# ── Test set generation ───────────────────────────────────────────────────────

def generate_test_params(n: int, seed: int = 99) -> list[dict[str, Any]]:
    """Generate n test parameter sets on a stratified grid.

    Uses a separate seed from training to ensure independence. The grid spans
    the schema parameter bounds, inset slightly to stay off the edges.
    """
    rng = np.random.default_rng(seed)
    inset_frac = 0.05
    grid_size = max(int(np.ceil(np.sqrt(n))), 2)

    w_margin = (WATER_RATIO_BOUNDS[1] - WATER_RATIO_BOUNDS[0]) * inset_frac
    s_margin = (PRINT_SPEED_BOUNDS[1] - PRINT_SPEED_BOUNDS[0]) * inset_frac
    waters = np.linspace(WATER_RATIO_BOUNDS[0] + w_margin, WATER_RATIO_BOUNDS[1] - w_margin, grid_size)
    speeds = np.linspace(PRINT_SPEED_BOUNDS[0] + s_margin, PRINT_SPEED_BOUNDS[1] - s_margin, grid_size)

    candidates = []
    for w in waters:
        for s in speeds:
            candidates.append({
                "water_ratio": float(w),
                "print_speed": float(s),
                "n_layers": N_LAYERS,
                "n_segments": N_SEGMENTS,
            })

    rng.shuffle(candidates)
    return candidates[:n]


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def compute_local_sensitivity(
    agent: Any,
    params: dict[str, Any],
    param_codes: list[str],
    perf_weights: dict[str, float],
    delta_frac: float = 0.02,
) -> dict[str, float]:
    """Compute local sensitivity |∂combined/∂param| at a point via finite differences."""
    sensitivities: dict[str, float] = {}
    for code in param_codes:
        val = params.get(code)
        if val is None or not isinstance(val, (int, float)):
            continue

        # Finite difference step
        delta = abs(float(val)) * delta_frac if abs(float(val)) > 1e-6 else delta_frac
        params_plus = {**params, code: float(val) + delta}
        params_minus = {**params, code: float(val) - delta}

        try:
            perf_plus = agent.predict_performance(params_plus)
            perf_minus = agent.predict_performance(params_minus)
            score_plus = combined_score(perf_plus, perf_weights)
            score_minus = combined_score(perf_minus, perf_weights)
            sensitivities[code] = abs(score_plus - score_minus) / (2 * delta)
        except (ValueError, RuntimeError, KeyError) as exc:
            print(f"  Warning: sensitivity for '{code}' failed ({exc}); recording nan")
            sensitivities[code] = float("nan")

    return sensitivities


