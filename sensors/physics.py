"""Pure, deterministic physics functions for the extrusion printing simulation.

Two materials (clay, concrete) × two designs (A, B) give four distinct operating points
with genuinely different optimal speeds and water ratios, creating a real Pareto front
between path accuracy, energy efficiency, and production rate.

Optimal speed formula (at optimal water ratio, layer 0):
    spd_opt = sqrt(THETA * MAT_SAG[mat] / (DELTA * DESIGN_COMPLEXITY[design]))

Target optima:
    (A, clay)     ≈ 40 mm/s,  w_opt=0.42
    (B, clay)     ≈ 33 mm/s,  w_opt=0.42
    (A, concrete) ≈ 25 mm/s,  w_opt=0.36
    (B, concrete) ≈ 20 mm/s,  w_opt=0.36

Three-way Pareto conflict:
    path_accuracy:    low deviation — needs correct speed AND correct water ratio
    energy_efficiency: low energy  — needs correct water for pump load (different optimum from deviation)
    production_rate:  high rate    — needs high speed AND not too much water (nozzle slip)
"""

# ── Design properties ──────────────────────────────────────────────────────────
# A: simpler path geometry; B: more complex curvature → harder to print accurately.
DESIGN_COMPLEXITY   = {"A": 1.0,  "B": 1.45}  # path complexity — drives inertia error
DESIGN_ENERGY_SCALE = {"A": 1.0,  "B": 1.25}  # B has a longer total path → more energy

# Per-design segment curvature values (4 segments, 0-indexed).
# A: simple radial path — curvature increases monotonically towards the outer segments.
# B: complex geometry — sharpest curvature at segment 2 (inside corner), eases off at end.
SEGMENT_CURVATURE = {
    "A": [0.80, 0.90, 1.00, 1.10],
    "B": [0.85, 1.10, 1.25, 0.95],
}

# ── Material properties ────────────────────────────────────────────────────────
# MAT_SAG: how much the material droops at low speed (higher → needs faster print)
# MAT_ENERGY: base energy scaling (concrete is stiffer → more motor load)
MAT_SAG    = {"clay": 1.6, "concrete": 0.6}
MAT_ENERGY = {"clay": 1.0, "concrete": 1.35}

# Optimal water ratio per material.
# flow = max(0.1, 1 − KAPPA · (water_ratio − w_opt_effective)²)
# Too little or too much water degrades flowability, raising deviation.
W_OPTIMAL = {"clay": 0.42, "concrete": 0.36}
KAPPA     = 20.0   # curvature of the flowability penalty well

# Shear-thinning coupling: at high print speed the material shears more, so it
# flows well at slightly lower water content.  The effective optimum shifts as:
#   w_opt_eff = w_opt + ALPHA_WS[mat] * (speed − speed_opt) / speed_opt
# Higher ALPHA_WS → tighter diagonal valley in the (speed, water) landscape.
ALPHA_WS = {"clay": 0.08, "concrete": 0.06}

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

# Per-material energy layer slope: how energy cost changes as layers accumulate.
# Clay dries as heat builds → pump resistance drops → energy decreases each layer.
# Concrete cures exothermically → increasing resistance → energy rises each layer.
ENERGY_LAYER_SLOPE = {"clay": -0.012, "concrete": +0.022}

# ── Production-rate slip ───────────────────────────────────────────────────────
# At high water ratios the mix is too fluid: the nozzle cannot build up back-pressure
# and effective extrusion rate drops below the commanded print speed.
# slip_factor = max(SLIP_FLOOR, 1 − OMEGA · max(0, water_ratio − W_SLIP[mat])²)
W_SLIP     = {"clay": 0.45, "concrete": 0.41}  # onset of nozzle-slip [water_ratio]
OMEGA      = 80.0    # strength of slip penalty
SLIP_FLOOR = 0.70    # minimum achievable production rate factor

# ── Sharp features (realistic AM discontinuities) ────────────────────────────
# These create thresholds and interaction effects that smooth models struggle
# with, making exploration more valuable than uniform sampling.

# Layer adhesion threshold: above this speed, fresh layer doesn't bond properly.
# Clay (softer, longer open time) tolerates higher speed than concrete.
ADHESION_SPEED    = {"clay": 52.0, "concrete": 40.0}     # mm/s
ADHESION_PENALTY  = 0.0006                                 # m — deviation jump

# Workability floor: below this water ratio, material is too stiff to extrude.
# Creates a steep ramp, not a smooth quadratic.
W_WORKABILITY       = {"clay": 0.33, "concrete": 0.31}
WORKABILITY_PENALTY = 0.0010                               # m

# Pump cavitation: high speed + high water simultaneously → air entrainment.
# Rectangular interaction region in parameter space.
CAVITATION_SPEED  = 48.0                                   # mm/s
CAVITATION_WATER  = {"clay": 0.46, "concrete": 0.42}
CAVITATION_ENERGY = 2.5                                    # J — energy spike

# ── Visualization ──────────────────────────────────────────────────────────────
FILAMENT_RADIUS = 0.004  # m — fixed filament radius for 3D plots


def segment_curvature(segment_idx: int, design: str) -> float:
    """Return the curvature factor for a given segment and design.

    Design A has a monotonically increasing profile (simple radial path).
    Design B peaks at segment 2 (sharp inside corner), then eases off.
    """
    return SEGMENT_CURVATURE[design][segment_idx]


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

    The effective water optimum shifts with print speed (shear-thinning coupling):
    at higher speeds the material shears more, so it flows acceptably at slightly
    lower water content.  This creates a diagonal valley in (speed, water) space
    rather than two independent axes, making the optimum harder to find by
    axis-aligned sampling alone.
    """
    complexity = DESIGN_COMPLEXITY[design]
    sag_f      = MAT_SAG[material]
    curv       = segment_curvature(segment_idx, design)

    # Layer-specific optimal speed
    spd_opt_base  = (THETA * sag_f / (DELTA * complexity)) ** 0.5
    spd_opt_layer = spd_opt_base + LAYER_SPD_SHIFT[material] * layer_idx

    # Shear-thinning: w_opt shifts toward drier mix at higher speeds
    w_opt_eff = W_OPTIMAL[material] + ALPHA_WS[material] * (print_speed - spd_opt_base) / spd_opt_base
    flow = max(0.1, 1.0 - KAPPA * (water_ratio - w_opt_eff) ** 2)

    # U-shaped base deviation
    deviation_speed = DELTA * print_speed * complexity * curv
    deviation_sag   = THETA * sag_f / (print_speed * flow) * curv

    # Drift compounds when speed deviates from the layer optimum
    drift = (LAYER_DRIFT_BASE + LAYER_DRIFT_COUPLING * abs(print_speed - spd_opt_layer)) * layer_idx

    deviation = deviation_speed + deviation_sag + drift

    # Sharp feature: layer adhesion failure at high speed (amplifies with height)
    if print_speed > ADHESION_SPEED[material]:
        deviation += ADHESION_PENALTY * (1.0 + 0.2 * layer_idx)

    # Sharp feature: workability floor at low water ratio (steep ramp)
    if water_ratio < W_WORKABILITY[material]:
        shortfall = (W_WORKABILITY[material] - water_ratio) / 0.05
        deviation += WORKABILITY_PENALTY * shortfall

    return deviation


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

    The layer slope is material-specific: clay dries over time (less pump resistance,
    energy decreases per layer) while concrete cures exothermically (more resistance,
    energy increases per layer).
    """
    mat_e  = MAT_ENERGY[material]
    des_e  = DESIGN_ENERGY_SCALE[design]
    curv   = segment_curvature(segment_idx, design)
    avg_curv = sum(SEGMENT_CURVATURE["A"]) / 4  # 0.95 — normalise to a fixed reference
    layer_scale = 1.0 + ENERGY_LAYER_SLOPE[material] * layer_idx
    pos_scale = (curv / avg_curv) * layer_scale
    water_factor = 1.0 + KAPPA_E * (water_ratio - W_ENERGY_OPT[material]) ** 2
    energy = (ETA + PHI * print_speed) * mat_e * des_e * pos_scale * water_factor

    # Sharp feature: pump cavitation at high speed + high water simultaneously
    if print_speed > CAVITATION_SPEED and water_ratio > CAVITATION_WATER[material]:
        energy += CAVITATION_ENERGY

    return energy


def production_rate(
    print_speed: float,
    water_ratio: float,
    material: str,
) -> float:
    """Effective production rate [mm/s].

    At low-to-moderate water ratios, rate tracks print_speed linearly.
    Above W_SLIP the mix becomes too fluid: the nozzle loses back-pressure and
    actual extrusion rate falls below the commanded speed (nozzle-slip effect).
    Range: [SLIP_FLOOR × print_speed_min, 60] ≈ [14, 60] mm/s.
    """
    slip_factor = max(SLIP_FLOOR, 1.0 - OMEGA * max(0.0, water_ratio - W_SLIP[material]) ** 2)
    return print_speed * slip_factor
