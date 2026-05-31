"""Synthetic ADVEI 2026 physics — feature values computed directly from parameters.

The mock simulates at the *feature level*: each ``feature_*`` function returns a
plausible synthetic measurement (one node_overlap, one filament_width, …) as a
smooth, deterministic function of the process parameters and the (layer, node)
coordinate. The real-fabrication counterpart in ``learning-by-printing`` extracts
the same features from real sensors.

Design goals (mirrors the lbp study so the workflow is faithful):

- **Pareto-rich.** The five performance attributes are each reachable but not
  jointly maximisable — every parameter trades one attribute against another
  (e.g. high ``print_speed`` is fast and cheap on time but worse on extrusion
  stability and energy; high ``calibration_factor`` widens the filament toward
  target but over-extrudes the loadcell trace).
- **Range-matched.** Magnitudes land in the lbp evaluators' bounds: node_overlap
  and filament_width around the 7 mm target; loadcell_residual in [0, 800] g²;
  per-experiment robot_energy in ~70–185 kJ and printing_duration in ~110–300 s.
- **Smooth + deterministic.** No discontinuities; small per-layer/per-node drift
  keeps predictions non-trivial. ``slowdown_factor`` is fixed at 0 for ADVEI, so
  the corner-slowdown axis is absent here.
"""
from __future__ import annotations

import math

from models.schema import PATH_LENGTH_M, TARGET_FILAMENT_WIDTH_MM

# Parameter mid-points (centres of the lbp bounds) — used to centre the physics.
_CALIB_MID = 2.0      # calibration_factor ∈ [1.8, 2.2]
_OFFSET_MID = 2.0     # path_offset ∈ [1, 3] mm
_SPEED_MID = 0.075    # print_speed ∈ [0.05, 0.1] m/s
_SPEED_HALF = 0.025   # half-range of print_speed
_LH_MID = 2.5         # layer_height ∈ [2, 3] mm

# Robot power reference (W) at mid-speed / mid-load — sets the energy scale.
_POWER_REF_W = 760.0

# Context ambient terms (deterministic drift + bounded oscillation over layers).
_TEMP_AMBIENT_C = 22.0
_HUMID_AMBIENT_PCT = 45.0
_MATERIAL_AGE_BASE_H = 3.0

# Curved-wall node-curvature profile: corners (ends) curve more than the middle.
_NODE_CURVATURE = (1.20, 1.05, 0.95, 0.90, 0.95, 1.05, 1.20)


def _node_curvature(node_idx: int, n_nodes: int = 7) -> float:
    """Curvature multiplier at ``node_idx`` — high at the corners, low mid-span."""
    if n_nodes == len(_NODE_CURVATURE):
        return _NODE_CURVATURE[node_idx]
    if n_nodes <= 1:
        return 1.0
    t = node_idx / (n_nodes - 1)
    return 0.95 + 0.25 * (1.0 - math.sin(math.pi * t))


# === Per-(layer, node) quality features ======================================

def feature_node_overlap(*, path_offset_mm: float, calibration_factor: float,
                         layer_idx: int, node_idx: int, n_nodes: int = 7) -> float:
    """Per-(layer, node) overlap between adjacent toolpaths [mm]. Target ≈ 7 mm.

    More material (``calibration_factor``↑) widens the overlap; a larger
    ``path_offset`` spreads the paths apart and shrinks it; corners accumulate
    a little more; slow clay creep adds a small drift over height.
    """
    curv = _node_curvature(node_idx, n_nodes)
    base = (TARGET_FILAMENT_WIDTH_MM
            + 9.0 * (calibration_factor - _CALIB_MID)
            - 1.6 * (path_offset_mm - _OFFSET_MID))
    corner = 1.0 + 0.10 * (curv - 1.0)
    creep = 0.04 * layer_idx
    return max(0.0, base * corner + creep)


def feature_filament_width(*, calibration_factor: float, print_speed_mps: float,
                           layer_height_mm: float, layer_idx: int, node_idx: int,
                           n_nodes: int = 7) -> float:
    """Per-(layer, node) deposited filament width [mm]. Target ≈ 7 mm (tight ±1 mm).

    More flow (``calibration_factor``↑) widens; faster printing stretches the
    bead thinner; a taller layer squashes less and narrows it. Corners get a
    small thickening from local accumulation.
    """
    curv = _node_curvature(node_idx, n_nodes)
    width = (TARGET_FILAMENT_WIDTH_MM
             + 3.0 * (calibration_factor - _CALIB_MID)
             - 10.0 * (print_speed_mps - _SPEED_MID)
             - 0.6 * (layer_height_mm - _LH_MID)
             + 0.25 * (curv - 1.0))
    width += 0.05 * math.sin(0.4 * layer_idx + 0.3 * node_idx)
    return max(1.0, width)


# === Per-layer features ======================================================

def feature_loadcell_residual(*, print_speed_mps: float, calibration_factor: float,
                              layer_idx: int) -> float:
    """Per-layer extrusion-stability residual [g²] — template-fit MSE, lower is better.

    Smoothest at mid speed; both too-slow (creep) and too-fast (slip) raise the
    residual. Over-extrusion (``calibration_factor`` above mid) drips and adds
    residual. Later layers benefit slightly from thermal warm-up.
    """
    speed_norm = (print_speed_mps - _SPEED_MID) / _SPEED_HALF      # 0 at centre, ±1 at edges
    over_extrusion = max(0.0, (calibration_factor - _CALIB_MID) / 0.2)
    residual = (80.0
                + 360.0 * speed_norm ** 2
                + 300.0 * over_extrusion ** 2
                - 3.0 * layer_idx)
    return max(0.0, residual)


def feature_printing_duration(*, print_speed_mps: float) -> float:
    """Per-layer printing duration [s] = toolpath length / print speed."""
    return PATH_LENGTH_M / max(print_speed_mps, 1e-6)


def feature_robot_energy(*, print_speed_mps: float, layer_height_mm: float,
                         layer_idx: int) -> float:
    """Per-layer robot energy [J] = power · duration.

    Power rises steeply with speed (kinetic + acceleration), while duration falls
    as 1/speed — net energy *increases* with speed, so fast printing is quick but
    energy-hungry (the speed↔time vs. speed↔energy tension). Taller layers carry
    more deposited mass and cost a little more.
    """
    duration_s = feature_printing_duration(print_speed_mps=print_speed_mps)
    power_w = _POWER_REF_W * (print_speed_mps / _SPEED_MID) ** 1.9
    mass_factor = 1.0 + 0.15 * (layer_height_mm - _LH_MID)
    warmup = 1.0 - 0.004 * layer_idx
    return power_w * mass_factor * warmup * duration_s


# === Context (uncontrollable) features =======================================

def feature_temperature(layer_idx: int) -> float:
    """Per-layer mean ambient temperature [°C] — slow drift + bounded oscillation."""
    return _TEMP_AMBIENT_C + 0.05 * layer_idx + 0.4 * math.sin(0.7 * layer_idx + 1.1)


def feature_humidity(layer_idx: int) -> float:
    """Per-layer mean ambient humidity [%] — slow drift + bounded oscillation."""
    return _HUMID_AMBIENT_PCT - 0.10 * layer_idx + 1.5 * math.cos(0.5 * layer_idx + 0.4)


def feature_material_age(layer_idx: int) -> float:
    """Per-layer clay batch age [h] — ages slowly as the print proceeds."""
    return _MATERIAL_AGE_BASE_H + 0.04 * layer_idx
