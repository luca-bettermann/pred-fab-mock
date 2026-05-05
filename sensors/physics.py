"""Synthetic ADVEI 2026 physics — produces feature values directly.

The mock simulates fabrication at the *feature level*: each call to a
``feature_*`` function returns a plausible synthetic measurement (one
node_overlap, one filament_width, one extrusion_consistency, etc.) as a
function of the process parameters and the (layer, node) coordinate.

Design goals:

- Pareto-rich. Five performance attributes (three quality, two cost) all
  reachable but not jointly maximisable — every parameter has an effect
  axis that competes with another.
- Smooth + deterministic. No discontinuities, no random surprises across
  the parameter grid; small per-layer noise terms model fab variability.
- Physically flavoured. Numerical magnitudes land in roughly the right
  units (mm for geometry, A for motor current, s for duration), so log
  output reads sensibly.

Constants exposed at the top of the file (component height, supply voltage,
etc.) — anything that's part of the physical setup but not on the
optimisation surface.
"""

from __future__ import annotations

import math


# === Hardware / geometric constants ==========================================

COMPONENT_HEIGHT_MM = 30.0       # curved-wall height (mm). n_layers = round(height / layer_height)
MAX_N_LAYERS = 15                # fixed tensor size — layer_height=2.0 → 15 real layers (the max)
PATH_LENGTH_PER_LAYER_M = 0.50   # nominal toolpath length per layer (m)
TARGET_FILAMENT_WIDTH_MM = 11.0  # target deposition width (mm) — drives material_deposition
TARGET_NODE_OVERLAP_MM = 1.5     # target overlap at the corner nodes (mm)
SUPPLY_VOLTAGE_V = 3.0           # extruder motor supply voltage (V) — used by energy footprint

# Per-axis ambient noise terms (deterministic, seeded by coordinates)
TEMP_AMBIENT_C = 22.0
TEMP_DRIFT_PER_LAYER = 0.05
TEMP_NOISE_AMP = 0.4

HUMID_AMBIENT_PCT = 45.0
HUMID_DRIFT_PER_LAYER = -0.10
HUMID_NOISE_AMP = 1.5


# === Curved-wall node-curvature profile ======================================
# Symmetric profile with stronger curvature at the wall corners. Treated as
# fixed geometry — n_nodes is canonical at 7 (mirrors learning-by-printing).
NODE_CURVATURE_PROFILE = [1.20, 1.05, 0.95, 0.90, 0.95, 1.05, 1.20]


def _node_curvature(node_idx: int, n_nodes: int = 7) -> float:
    """Return curvature multiplier at ``node_idx``. Falls back to a smooth
    cosine profile when ``n_nodes`` differs from the canonical 7.
    """
    if n_nodes == len(NODE_CURVATURE_PROFILE):
        return NODE_CURVATURE_PROFILE[node_idx]
    # Cosine profile in [0.9, 1.2]: high at endpoints, low in the middle.
    if n_nodes <= 1:
        return 1.0
    t = node_idx / (n_nodes - 1)  # 0 → 1
    return 0.95 + 0.25 * (1.0 - math.sin(math.pi * t))


def n_layers_for_height(layer_height_mm: float) -> int:
    """Derive layer count from per-print layer height (round to nearest int)."""
    return max(1, int(round(COMPONENT_HEIGHT_MM / layer_height_mm)))


# === Per-(layer, node) features ==============================================

def feature_node_overlap(
    *,
    path_offset_mm: float,
    layer_height_mm: float,
    calibration_factor: float,
    print_speed_mps: float,
    slowdown_factor: float,
    layer_idx: int,
    node_idx: int,
    n_nodes: int = 7,
) -> float:
    """Per-(layer, node) overlap between adjacent toolpaths [mm].

    Drivers:
      - ``calibration_factor``: more material → more overlap (linear).
      - ``path_offset_mm``: larger offset → less overlap (linear penalty).
      - ``slowdown_factor``: higher slowdown → cleaner corners → slightly
        less accumulated overlap.
      - ``layer_idx``: clay creep adds a small overlap drift over height.
      - ``node_idx``: corners (high curvature) accumulate more overlap.
    """
    curv = _node_curvature(node_idx, n_nodes)
    base = (calibration_factor - 1.6) * 1.4 - path_offset_mm * 0.55
    creep = 0.05 * layer_idx
    corner_factor = 0.90 + 0.25 * (curv - 1.0)
    slowdown_penalty = 1.0 - 0.18 * slowdown_factor
    overlap = max(0.0, base + creep) * corner_factor * slowdown_penalty
    # Bias toward the typical 0–3 mm range.
    return 0.20 + overlap


def feature_filament_width(
    *,
    path_offset_mm: float,
    layer_height_mm: float,
    calibration_factor: float,
    print_speed_mps: float,
    slowdown_factor: float,
    layer_idx: int,
    node_idx: int,
    n_nodes: int = 7,
) -> float:
    """Per-(layer, node) filament width [mm].

    Drivers:
      - ``calibration_factor``: more flow → wider filament.
      - ``print_speed_mps``: faster → stretched thinner.
      - ``layer_height_mm``: taller layer → less squash → narrower.
      - ``slowdown_factor``: at corners, slowdown thickens the filament.
      - ``node_idx``: corners get a small thickness bump from slow-down +
        local accumulation.
    """
    curv = _node_curvature(node_idx, n_nodes)
    width = (
        9.5
        + 4.5 * (calibration_factor - 1.6) / 0.6      # +0..4.5 mm via calibration
        - 0.35 * (print_speed_mps - 0.004) * 1000.0   # -0..1.4 mm via speed
        - 1.5 * (layer_height_mm - 2.0)               # -0..1.5 mm via layer height
        + 0.6 * slowdown_factor * (curv - 1.0)        # corner bump
    )
    # Tiny per-layer drift to keep predictions non-trivial.
    width += 0.05 * math.sin(0.4 * layer_idx + 0.3 * node_idx)
    return max(2.0, width)


# === Per-layer features ======================================================

def feature_extrusion_consistency(
    *,
    print_speed_mps: float,
    slowdown_factor: float,
    calibration_factor: float,
    layer_idx: int,
) -> float:
    """Per-layer extrusion stability proxy ∈ (0, 1] (R² of cumulative-weight line fit).

    Drivers:
      - ``print_speed_mps``: smoothest in the middle of the speed band; both
        too-slow (creep) and too-fast (slip) degrade R².
      - ``slowdown_factor``: heavy slowdown introduces stop-go transients.
      - ``calibration_factor``: extreme over-extrusion drips.
      - ``layer_idx``: late layers benefit from thermal warm-up (small
        improvement).
    """
    speed_norm = (print_speed_mps - 0.006) / 0.002       # centred at 0.006 m/s
    speed_pen = 0.18 * speed_norm ** 2
    slowdown_pen = 0.22 * slowdown_factor ** 2
    calib_pen = 0.06 * max(0.0, calibration_factor - 2.0)
    warmup = -0.01 * layer_idx                            # later layers slightly steadier
    raw = 1.0 - speed_pen - slowdown_pen - calib_pen + warmup
    return max(0.30, min(1.0, raw))


def feature_current_mean_feeder(
    *,
    calibration_factor: float,
    layer_height_mm: float,
    print_speed_mps: float,
    slowdown_factor: float,
    layer_idx: int,
) -> float:
    """Per-layer mean feeder-motor current [A].

    Drivers:
      - ``calibration_factor``: higher flow demand → higher current (dominant).
      - ``layer_height_mm``: thicker layer → more material per unit time.
      - ``print_speed_mps``: faster → faster pump → higher current.
      - ``slowdown_factor``: slowdown reduces effective speed → small dip.
    """
    effective_speed = print_speed_mps * (1.0 - 0.45 * slowdown_factor)
    current = (
        0.55
        + 0.40 * (calibration_factor - 1.6) / 0.6        # 0–0.4 A via calibration
        + 0.30 * (layer_height_mm - 2.0)                  # 0–0.3 A via layer height
        + 90.0 * effective_speed                          # 0.36–0.72 A via speed
    )
    # Small thermal drift up over height.
    current += 0.01 * layer_idx
    return max(0.20, current)


def feature_current_mean_nozzle(
    *,
    calibration_factor: float,
    print_speed_mps: float,
    slowdown_factor: float,
    layer_idx: int,
) -> float:
    """Per-layer mean nozzle-screw-motor current [A]. Smaller than feeder,
    more sensitive to slowdown (corner deceleration unloads the screw).
    """
    effective_speed = print_speed_mps * (1.0 - 0.45 * slowdown_factor)
    current = (
        0.40
        + 0.30 * (calibration_factor - 1.6) / 0.6
        + 75.0 * effective_speed
        - 0.06 * slowdown_factor                          # screw unloads on slowdown
    )
    current += 0.008 * layer_idx
    return max(0.15, current)


def feature_printing_duration(
    *,
    print_speed_mps: float,
    slowdown_factor: float,
) -> float:
    """Per-layer printing duration [s]. Deterministic from speed + slowdown.

    Same formula used by ``DeterministicDuration`` predictor at offline-
    planning time — fab-side simulation (this function) is the ground truth
    that the predictor mirrors.
    """
    effective_speed = print_speed_mps * (1.0 - 0.45 * slowdown_factor)
    if effective_speed <= 1e-6:
        return 1e6
    return PATH_LENGTH_PER_LAYER_M / effective_speed


# === Context (uncontrollable) features =======================================

def feature_temperature(layer_idx: int) -> float:
    """Per-layer mean ambient temperature [°C]. Slow drift + small noise."""
    return (
        TEMP_AMBIENT_C
        + TEMP_DRIFT_PER_LAYER * layer_idx
        + TEMP_NOISE_AMP * math.sin(0.7 * layer_idx + 1.1)
    )


def feature_humidity(layer_idx: int) -> float:
    """Per-layer mean ambient humidity [%]. Slow drift + small noise."""
    return (
        HUMID_AMBIENT_PCT
        + HUMID_DRIFT_PER_LAYER * layer_idx
        + HUMID_NOISE_AMP * math.cos(0.5 * layer_idx + 0.4)
    )
