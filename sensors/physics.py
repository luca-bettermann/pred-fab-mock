"""Pure, deterministic physics functions for the extrusion printing simulation."""

# Design path complexity coefficients
DESIGN_COMPLEXITY = {"A": 1.0, "B": 1.5, "C": 2.2}
DESIGN_SCALE = {"A": 0.9, "B": 1.0, "C": 1.1}

# Material viscosity coefficients
MATERIAL_VISCOSITY = {"standard": 1.0, "reinforced": 1.4, "flexible": 0.7}
MATERIAL_FACTOR = {"standard": 1.0, "reinforced": 1.05, "flexible": 0.95}

# Physics constants
ALPHA_OFFSET = 0.0035  # constant width offset (was ALPHA * fixed_layer_height)
BETA = 0.008        # water_ratio contribution to width
GAMMA = 0.0002      # inverse print_speed contribution to width
DELTA = 0.000030    # print_speed contribution to deviation [m / (mm/s)]
ZETA = 0.000400     # design complexity contribution to deviation [m per unit]
LAYER_DRIFT = 0.000150  # m/layer — path drift from material settling per layer
ETA = 1.0           # base energy per segment [J]
PHI = 0.35          # print_speed contribution to energy [J / (mm/s)]


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
    layer_idx: int = 0,
) -> float:
    """Deterministic lateral path deviation [m].

    Deviation scales linearly with print_speed and design complexity.
    LAYER_DRIFT adds a per-layer offset simulating gradual material settling.
    """
    complexity = DESIGN_COMPLEXITY[design]
    curv = segment_curvature(segment_idx)
    base = (DELTA * print_speed + ZETA * complexity) * curv
    return base + LAYER_DRIFT * layer_idx


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
