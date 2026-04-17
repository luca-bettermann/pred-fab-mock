"""CLI helper functions: inline plot display, physics randomization, test set generation."""

import base64
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np


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

# Default physics constants (from sensors/physics.py)
DEFAULT_PHYSICS = {
    "W_OPTIMAL": 0.42,
    "W_ENERGY_OPT": 0.38,
    "DELTA": 0.000011,
    "THETA": 0.011,
    "SAG": 1.6,
    "COMPLEXITY": 1.0,
    "LAYER_SPD_SHIFT": 0.40,
    "W_SLIP": 0.45,
    "SEGMENT_CURVATURE": [0.85, 1.15, 0.95, 1.05],
}

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
    base_curv = np.array([0.85, 1.15, 0.95, 1.05])
    perturb = rng.uniform(-0.05, 0.05, size=4)
    config["SEGMENT_CURVATURE"] = (base_curv + perturb).tolist()
    rng.shuffle(config["SEGMENT_CURVATURE"])

    return config


def apply_physics_config(config: dict[str, Any]) -> None:
    """Apply physics config to the sensors.physics module at runtime."""
    import sensors.physics as phys

    for key, val in config.items():
        if key == "SEGMENT_CURVATURE":
            phys.SEGMENT_CURVATURE = list(val)
        elif hasattr(phys, key):
            setattr(phys, key, val)


def load_physics_from_session(session_config: dict[str, Any]) -> None:
    """Load and apply physics config from session if present."""
    physics = session_config.get(PHYSICS_CONFIG_KEY)
    if physics:
        apply_physics_config(physics)


# ── Test set generation ───────────────────────────────────────────────────────

def generate_test_params(n: int, seed: int = 99) -> list[dict[str, Any]]:
    """Generate n test parameter sets on a stratified grid.

    Uses a separate seed from training to ensure independence.
    Reads dimension bounds from the physics constants to respect schema constraints.
    """
    rng = np.random.default_rng(seed)

    from sensors.physics import N_LAYERS, N_SEGMENTS

    waters = np.linspace(0.31, 0.49, max(int(np.ceil(np.sqrt(n))), 2))
    speeds = np.linspace(21.0, 59.0, max(int(np.ceil(np.sqrt(n))), 2))

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
    from pred_fab import combined_score

    base_perf = agent.predict_performance(params)
    base_score = combined_score(base_perf, perf_weights)

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
        except Exception:
            sensitivities[code] = 0.0

    return sensitivities


def plot_sensitivity(
    save_path: str,
    sensitivities: dict[str, float],
    title: str = "Local Sensitivity Analysis",
) -> None:
    """Delegate to pred_fab.plotting.plot_sensitivity."""
    from pred_fab.plotting import plot_sensitivity as _plot
    _plot(save_path, sensitivities, title=title)
