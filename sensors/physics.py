"""Pure, deterministic physics functions for the extrusion printing simulation.

Two materials (clay, concrete) × two designs (A, B) give four distinct operating points
with genuinely different optimal speeds and water ratios, creating a real Pareto front
between path accuracy and energy efficiency.

Optimal speed formula (at optimal water ratio, layer 0):
    spd_opt = sqrt(THETA * MAT_SAG[mat] / (DELTA * DESIGN_COMPLEXITY[design]))

Target optima:
    (A, clay)     ≈ 40 mm/s,  w_opt=0.42
    (B, clay)     ≈ 33 mm/s,  w_opt=0.42
    (A, concrete) ≈ 25 mm/s,  w_opt=0.36
    (B, concrete) ≈ 20 mm/s,  w_opt=0.36
"""

# ── Design properties ──────────────────────────────────────────────────────────
# A: simpler path geometry; B: more complex curvature → harder to print accurately.
DESIGN_COMPLEXITY   = {"A": 1.0,  "B": 1.45}  # path complexity — drives inertia error
DESIGN_ENERGY_SCALE = {"A": 1.0,  "B": 1.25}  # B has a longer total path → more energy

# ── Material properties ────────────────────────────────────────────────────────
# MAT_SAG: how much the material droops at low speed (higher → needs faster print)
# MAT_ENERGY: base energy scaling (concrete is stiffer → more motor load)
MAT_SAG    = {"clay": 1.6, "concrete": 0.6}
MAT_ENERGY = {"clay": 1.0, "concrete": 1.35}

# Optimal water ratio per material.
# flow = max(0.1, 1 − KAPPA · (water_ratio − W_OPTIMAL)²)
# Too little or too much water degrades flowability, raising deviation.
W_OPTIMAL = {"clay": 0.42, "concrete": 0.36}
KAPPA     = 20.0   # curvature of the flowability penalty well

# ── U-shaped deviation constants ──────────────────────────────────────────────
# deviation = (DELTA·speed·complexity + THETA·sag/(speed·flow)) · curvature
# THETA/DELTA = 1000 sets the optimal-speed targets listed above.
DELTA = 0.000011   # high-speed inertia coefficient [m / (mm/s · complexity)]
THETA = 0.011      # low-speed sag coefficient [m · mm/s / sag]

# ── Layer drift: deviation grows with layer index ──────────────────────────────
# As layers stack, small speed errors compound. When the print speed deviates from
# the layer-specific optimum the drift penalty grows proportionally.
#
#   spd_opt_layer = spd_opt_base + LAYER_SPD_SHIFT[mat] × layer_idx
#   drift = (LAYER_DRIFT_BASE + LAYER_DRIFT_COUPLING × |speed − spd_opt_layer|) × layer_idx
#
# Clay softens as heat builds → optimal speed creeps up each layer (+0.4 mm/s).
# Concrete cures and stiffens → optimal speed decreases each layer (−0.55 mm/s).
LAYER_SPD_SHIFT    = {"clay": +0.40, "concrete": -0.55}  # mm/s per layer
LAYER_DRIFT_BASE   = 0.000120   # m/layer — drift even at perfectly optimal speed
LAYER_DRIFT_COUPLING = 0.000015 # m/layer per mm/s speed error

# ── Energy constants ───────────────────────────────────────────────────────────
ETA = 0.8    # base energy per segment [J]
PHI = 0.18   # energy per unit print speed [J / (mm/s)]

# Water-ratio optimum for energy (different from W_OPTIMAL for deviation — creates Pareto conflict).
# Too little water → high pump resistance; too much → heat of vaporisation load.
W_ENERGY_OPT = {"clay": 0.38, "concrete": 0.32}
KAPPA_E      = 15.0   # curvature of energy penalty well

# ── Visualization ──────────────────────────────────────────────────────────────
FILAMENT_RADIUS = 0.004  # m — fixed filament radius for 3D plots


def segment_curvature(segment_idx: int) -> float:
    """Return a curvature factor for a given segment index (0-based)."""
    return 0.8 + 0.15 * segment_idx


def path_deviation(
    print_speed: float,
    design: str,
    segment_idx: int,
    water_ratio: float,
    material: str,
    layer_idx: int = 0,
) -> float:
    """Deterministic lateral path deviation [m].

    Combines a U-shaped speed response (inertia vs sag) with a per-layer drift
    term that amplifies when the speed is far from the layer-specific optimum.
    This makes layer-by-layer adaptation genuinely necessary: the optimal speed
    shifts each layer (clay up, concrete down) and errors compound.
    """
    complexity = DESIGN_COMPLEXITY[design]
    sag_f      = MAT_SAG[material]
    curv       = segment_curvature(segment_idx)
    w_opt      = W_OPTIMAL[material]
    flow       = max(0.1, 1.0 - KAPPA * (water_ratio - w_opt) ** 2)

    # U-shaped base deviation
    deviation_speed = DELTA * print_speed * complexity * curv
    deviation_sag   = THETA * sag_f / (print_speed * flow) * curv

    # Layer-specific optimal speed (curing/settling effect)
    spd_opt_base  = (THETA * sag_f / (DELTA * complexity)) ** 0.5
    spd_opt_layer = spd_opt_base + LAYER_SPD_SHIFT[material] * layer_idx

    # Drift compounds when speed deviates from the layer optimum
    drift = (LAYER_DRIFT_BASE + LAYER_DRIFT_COUPLING * abs(print_speed - spd_opt_layer)) * layer_idx

    return deviation_speed + deviation_sag + drift


def energy_per_segment(
    print_speed: float,
    material: str,
    design: str,
    water_ratio: float,
    segment_idx: int = 0,
    layer_idx: int = 0,
) -> float:
    """Deterministic energy consumed per segment [J].

    Energy scales with print speed, material stiffness, and design path length.
    A U-shaped water_ratio term (W_ENERGY_OPT differs from W_OPTIMAL) creates a
    genuine Pareto conflict between minimising deviation and minimising energy.
    """
    mat_e  = MAT_ENERGY[material]
    des_e  = DESIGN_ENERGY_SCALE[design]
    curv   = segment_curvature(segment_idx)
    avg_curv = 1.025  # mean of curvature(0..3)
    pos_scale = (curv / avg_curv) * (1.0 + 0.03 * layer_idx)
    water_factor = 1.0 + KAPPA_E * (water_ratio - W_ENERGY_OPT[material]) ** 2
    return (ETA + PHI * print_speed) * mat_e * des_e * pos_scale * water_factor
