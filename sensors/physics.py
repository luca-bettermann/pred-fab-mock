"""Pure, deterministic physics functions for the extrusion printing simulation."""

# Design path complexity coefficients
DESIGN_COMPLEXITY = {"A": 1.0, "B": 1.5, "C": 2.2}
DESIGN_SCALE = {"A": 0.9, "B": 1.0, "C": 1.1}

# Material viscosity coefficients
MATERIAL_VISCOSITY = {"standard": 1.0, "reinforced": 1.4, "flexible": 0.7}
MATERIAL_FACTOR = {"standard": 1.0, "reinforced": 1.05, "flexible": 0.95}

# Physics constants — filament width
ALPHA_OFFSET = 0.0035  # constant width offset
BETA = 0.008           # water_ratio contribution to width
GAMMA = 0.0002         # inverse print_speed contribution to width

# Physics constants — path deviation (U-shaped speed response)
# Two competing error sources:
#   deviation_speed = DELTA * print_speed * complexity * curvature   (inertia / overshoot)
#   deviation_sag   = THETA * viscosity / (print_speed * flow) * curvature  (low-speed droop)
# Optimal speed = sqrt(THETA * viscosity / (DELTA * complexity * flow)), varies by design+material.
DELTA = 0.000011       # high-speed inertia coefficient [m / (mm/s · complexity)]
THETA = 0.029          # low-speed sag coefficient [m · mm/s / viscosity]
LAYER_DRIFT = 0.000150 # m/layer — gradual path drift from material settling

# Flowability: how well the material flows at a given water_ratio.
# flow = max(0.1, 1 − KAPPA · (water_ratio − w_opt)²)
# Too little or too much water degrades flow; the optimum varies by material.
KAPPA = 20.0
W_OPTIMAL_WATER = {
    "flexible":   0.44,   # low viscosity — benefits from more water
    "standard":   0.40,   # balanced
    "reinforced": 0.36,   # high viscosity — too much water hurts stiffness
}

# Physics constants — energy
ETA = 1.0    # base energy per segment [J]
PHI = 0.35   # print_speed contribution to energy [J / (mm/s)]


def segment_curvature(segment_idx: int) -> float:
    """Return a curvature factor for a given segment index (0-based)."""
    return 0.8 + 0.15 * segment_idx


def filament_width(
    water_ratio: float,
    print_speed: float,
    design: str,
    material: str,
) -> float:
    """Deterministic filament width [m]."""
    ds = DESIGN_SCALE[design]
    mf = MATERIAL_FACTOR[material]
    return ALPHA_OFFSET + BETA * water_ratio - GAMMA / print_speed + ds * mf * 0.001


def path_deviation(
    print_speed: float,
    design: str,
    segment_idx: int,
    water_ratio: float,
    material: str,
    layer_idx: int = 0,
) -> float:
    """Deterministic lateral path deviation [m].

    Two competing error sources create a U-shaped speed response:
    - High-speed inertia error: grows with print_speed × design complexity.
    - Low-speed sag error: material droops when extruded slowly; grows with
      material viscosity / (print_speed × flowability).

    The optimal speed is in the interior of the operating range and varies by
    design complexity, material viscosity, and water_ratio (via flowability).
    Using too little or too much water degrades flowability, raising deviation
    at all speeds and shifting the optimal speed upward.

    Optimal speed formula (ignoring curvature):
        spd_opt = sqrt(THETA × viscosity / (DELTA × complexity × flow))
    """
    complexity = DESIGN_COMPLEXITY[design]
    viscosity = MATERIAL_VISCOSITY[material]
    curv = segment_curvature(segment_idx)
    w_opt = W_OPTIMAL_WATER[material]
    flow = max(0.1, 1.0 - KAPPA * (water_ratio - w_opt) ** 2)
    deviation_speed = DELTA * print_speed * complexity * curv
    deviation_sag = THETA * viscosity / (print_speed * flow) * curv
    return deviation_speed + deviation_sag + LAYER_DRIFT * layer_idx


def energy_per_segment(
    print_speed: float,
    material: str,
    segment_idx: int = 0,
    layer_idx: int = 0,
) -> float:
    """Deterministic energy consumed per segment [J].

    Base energy scales with print speed and material viscosity.
    Upper layers and curved segments consume slightly more energy.
    """
    viscosity = MATERIAL_VISCOSITY[material]
    AVG_CURVATURE = 1.025  # mean of curvature(0..3)
    pos_scale = (segment_curvature(segment_idx) / AVG_CURVATURE) * (1.0 + 0.03 * layer_idx)
    return (ETA + PHI * print_speed) * viscosity * pos_scale
