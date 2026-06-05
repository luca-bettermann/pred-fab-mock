"""Pure, deterministic physics functions for the extrusion printing simulation."""

# Design path complexity coefficients
DESIGN_COMPLEXITY = {"A": 1.0, "B": 1.5, "C": 2.2}
DESIGN_SCALE = {"A": 0.9, "B": 1.0, "C": 1.1}

# Material viscosity coefficients
MATERIAL_VISCOSITY = {"standard": 1.0, "reinforced": 1.4, "flexible": 0.7}
MATERIAL_FACTOR = {"standard": 1.0, "reinforced": 1.05, "flexible": 0.95}

# Filament-width physics constants
ALPHA = 0.5       # layer_height contribution to width
BETA = 0.008      # water_ratio contribution to width
GAMMA = 0.0002    # inverse print_speed contribution to width

# Path-deviation physics. Deviation is minimal at a material/design-specific
# operating point that lies in the interior of the calibration bounds, so the
# best parameters must be *discovered* rather than read off a boundary.
WATER_OPT = {"standard": 0.40, "reinforced": 0.44, "flexible": 0.37}  # within [0.30, 0.50]
SPEED_OPT = {"A": 32.0, "B": 40.0, "C": 50.0}                         # within [20, 60] mm/s
DEV_FLOOR = 0.00005   # m, residual deviation at the operating point
K_WATER   = 0.045     # m per (water_ratio offset)^2
K_SPEED   = 2.0e-6    # m per (mm/s offset)^2
ZETA      = 0.00002   # design-complexity texture

# Geometry for 3-D filament rendering
FILAMENT_RADIUS = 0.004   # m, nominal extruded bead radius


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
    water_ratio: float,
    print_speed: float,
    design: str,
    material: str,
    segment_idx: int,
) -> float:
    """Deterministic lateral path deviation [m] — a bowl with an interior optimum.

    Minimised at (WATER_OPT[material], SPEED_OPT[design]); rises quadratically as
    either calibration parameter moves off that operating point.
    """
    wr_opt = WATER_OPT[material]
    ps_opt = SPEED_OPT[design]
    curv = segment_curvature(segment_idx)
    bowl = K_WATER * (water_ratio - wr_opt) ** 2 + K_SPEED * (print_speed - ps_opt) ** 2
    texture = ZETA * DESIGN_COMPLEXITY[design] * curv
    return DEV_FLOOR + bowl + texture


def energy_per_segment(
    print_speed: float,
    layer_height: float,
    material: str,
    layer_time: float,
) -> float:
    """Deterministic energy consumed per segment [J]."""
    viscosity = MATERIAL_VISCOSITY[material]
    return (print_speed * layer_height * viscosity) / layer_time * 10.0
