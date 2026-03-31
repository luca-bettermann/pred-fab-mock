"""Pure, deterministic physics functions for the extrusion printing simulation."""

# Design path complexity coefficients
DESIGN_COMPLEXITY = {"A": 1.0, "B": 1.5, "C": 2.2}
DESIGN_SCALE = {"A": 0.9, "B": 1.0, "C": 1.1}

# Material viscosity coefficients
MATERIAL_VISCOSITY = {"standard": 1.0, "reinforced": 1.4, "flexible": 0.7}
MATERIAL_FACTOR = {"standard": 1.0, "reinforced": 1.05, "flexible": 0.95}

# Physics constants
ALPHA = 0.5       # layer_height contribution to width
BETA = 0.008      # water_ratio contribution to width
GAMMA = 0.0002    # inverse print_speed contribution to width
DELTA = 0.0003    # print_speed/layer_time contribution to deviation
ZETA = 0.0005     # design_complexity contribution to deviation


def segment_curvature(segment_idx: int) -> float:
    """Return a curvature factor for a given segment index (0-based)."""
    # Segments 0,1,2,3 → curvature increases slightly
    return 0.8 + 0.15 * segment_idx


def filament_width(
    layer_height: float,
    water_ratio: float,
    print_speed: float,
    design: str,
    material: str,
) -> float:
    """Deterministic filament width [m]."""
    ds = DESIGN_SCALE[design]
    mf = MATERIAL_FACTOR[material]
    return ALPHA * layer_height + BETA * water_ratio - GAMMA / print_speed + ds * mf * 0.001


def path_deviation(
    print_speed: float,
    layer_time: float,
    design: str,
    segment_idx: int,
) -> float:
    """Deterministic lateral path deviation [m]."""
    complexity = DESIGN_COMPLEXITY[design]
    curv = segment_curvature(segment_idx)
    return (DELTA * print_speed / layer_time + ZETA * complexity) * curv


def energy_per_segment(
    print_speed: float,
    layer_height: float,
    material: str,
    layer_time: float,
) -> float:
    """Deterministic energy consumed per segment [J]."""
    viscosity = MATERIAL_VISCOSITY[material]
    return (print_speed * layer_height * viscosity) / layer_time * 1200.0
