"""Pure, deterministic physics functions for the extrusion printing simulation.

Single design × single material (clay) with a 2D parameter space:
  water_ratio ∈ [0.30, 0.50]
  print_speed ∈ [20, 60] mm/s

Optimal operating point:
    speed ≈ 40 mm/s,  water ≈ 0.42

Three-way Pareto conflict:
    path_accuracy:    low deviation — needs correct speed AND correct water ratio
    energy_efficiency: low energy  — needs different water optimum than deviation
    production_rate:  high rate    — needs high speed AND not too much water (nozzle slip)
"""

# ── Segment curvature ────────────────────────────────────────────────────────
# Non-linear pattern: alternating high/low creates segment-dependent behaviour
# that makes the response surface more challenging than a monotonic profile.
SEGMENT_CURVATURE = [0.85, 1.15, 0.95, 1.05]

# Fixed design/fabrication constants
COMPLEXITY     = 1.0    # path complexity (inertia scaling)
ENERGY_SCALE   = 1.0    # path length energy scaling
N_LAYERS       = 5
N_SEGMENTS     = 4
TARGET_HEIGHT  = 0.040  # m
PATH_LENGTH    = 0.40   # m

# ── Material properties (clay) ────────────────────────────────────────────────
SAG            = 1.6    # how much material droops at low speed
MAT_ENERGY     = 1.0    # base energy scaling
W_OPTIMAL      = 0.42   # water ratio for minimum deviation
KAPPA          = 20.0   # curvature of flowability penalty well

# Shear-thinning coupling: effective water optimum shifts with speed.
#   w_opt_eff = W_OPTIMAL + ALPHA_WS * (speed - speed_opt) / speed_opt
# Creates a diagonal valley in (speed, water) space.
ALPHA_WS       = 0.08

# ── U-shaped deviation ────────────────────────────────────────────────────────
# deviation = (DELTA*speed*complexity + THETA*sag/(speed*flow)) * curvature
DELTA          = 0.000011   # high-speed inertia coefficient
THETA          = 0.011      # low-speed sag coefficient

# ── Layer drift ───────────────────────────────────────────────────────────────
# Clay softens as heat builds → optimal speed creeps up each layer (+0.4 mm/s).
LAYER_SPD_SHIFT    = +0.40      # mm/s per layer
LAYER_DRIFT_BASE   = 0.000120   # m/layer — drift even at optimal speed
LAYER_DRIFT_COUPLING = 0.000015 # m/layer per mm/s speed error

# ── Energy ────────────────────────────────────────────────────────────────────
ETA            = 0.8    # base energy per segment [J]
PHI            = 0.18   # energy per unit print speed [J / (mm/s)]
W_ENERGY_OPT   = 0.38   # different from W_OPTIMAL → Pareto conflict
KAPPA_E        = 15.0   # curvature of energy penalty well
ENERGY_LAYER_SLOPE = -0.012  # clay dries → energy drops per layer

# ── Production-rate slip ──────────────────────────────────────────────────────
W_SLIP         = 0.45   # onset of nozzle-slip
OMEGA          = 80.0   # strength of slip penalty
SLIP_FLOOR     = 0.70   # minimum achievable rate factor

# ── Smooth features ──────────────────────────────────────────────────────────
# Adhesion threshold: smooth sigmoid instead of hard cliff.
# Deviation rises steeply but continuously above ADHESION_SPEED.
ADHESION_SPEED   = 52.0     # mm/s — centre of sigmoid
ADHESION_PENALTY = 0.0006   # m — maximum deviation addition
ADHESION_K       = 0.5      # sigmoid steepness (1/mm/s)

# Workability floor: smooth ramp below W_WORKABILITY.
W_WORKABILITY       = 0.33
WORKABILITY_PENALTY = 0.0010  # m

# ── Visualization ─────────────────────────────────────────────────────────────
FILAMENT_RADIUS = 0.004  # m — fixed filament radius for 3D plots


def segment_curvature(segment_idx: int) -> float:
    """Return the curvature factor for a given segment."""
    return SEGMENT_CURVATURE[segment_idx]


def path_deviation(
    print_speed: float,
    segment_idx: int,
    water_ratio: float,
    layer_idx: int = 0,
) -> float:
    """Deterministic lateral path deviation [m].

    U-shaped speed response (inertia vs sag) with per-layer drift and
    shear-thinning coupling that creates a diagonal valley in (speed, water).
    """
    curv = segment_curvature(segment_idx)

    # Layer-specific optimal speed
    spd_opt_base  = (THETA * SAG / (DELTA * COMPLEXITY)) ** 0.5
    spd_opt_layer = spd_opt_base + LAYER_SPD_SHIFT * layer_idx

    # Shear-thinning: w_opt shifts toward drier mix at higher speeds
    w_opt_eff = W_OPTIMAL + ALPHA_WS * (print_speed - spd_opt_base) / spd_opt_base
    flow = max(0.1, 1.0 - KAPPA * (water_ratio - w_opt_eff) ** 2)

    # U-shaped base deviation
    deviation_speed = DELTA * print_speed * COMPLEXITY * curv
    deviation_sag   = THETA * SAG / (print_speed * flow) * curv

    # Drift compounds when speed deviates from the layer optimum
    drift = (LAYER_DRIFT_BASE + LAYER_DRIFT_COUPLING * abs(print_speed - spd_opt_layer)) * layer_idx

    deviation = deviation_speed + deviation_sag + drift

    # Smooth sigmoid adhesion penalty at high speed (amplifies with height)
    sigmoid = 1.0 / (1.0 + _exp_safe(-ADHESION_K * (print_speed - ADHESION_SPEED)))
    deviation += ADHESION_PENALTY * sigmoid * (1.0 + 0.2 * layer_idx)

    # Smooth workability ramp at low water ratio
    if water_ratio < W_WORKABILITY:
        shortfall = (W_WORKABILITY - water_ratio) / 0.05
        deviation += WORKABILITY_PENALTY * shortfall

    return deviation


def energy_per_segment(
    print_speed: float,
    water_ratio: float,
    segment_idx: int = 0,
    layer_idx: int = 0,
) -> float:
    """Deterministic energy consumed per segment [J].

    U-shaped water_ratio response (W_ENERGY_OPT ≠ W_OPTIMAL) creates a
    genuine Pareto conflict with path accuracy. Energy decreases per layer
    as clay dries (less pump resistance).
    """
    curv       = segment_curvature(segment_idx)
    avg_curv   = sum(SEGMENT_CURVATURE) / N_SEGMENTS
    layer_scale = 1.0 + ENERGY_LAYER_SLOPE * layer_idx
    pos_scale  = (curv / avg_curv) * layer_scale
    water_factor = 1.0 + KAPPA_E * (water_ratio - W_ENERGY_OPT) ** 2
    return (ETA + PHI * print_speed) * MAT_ENERGY * ENERGY_SCALE * pos_scale * water_factor


def production_rate(
    print_speed: float,
    water_ratio: float,
) -> float:
    """Effective production rate [mm/s].

    Linear below W_SLIP, quadratic collapse above (nozzle-slip effect).
    Range: [SLIP_FLOOR * 20, 60] ≈ [14, 60] mm/s.
    """
    slip_factor = max(SLIP_FLOOR, 1.0 - OMEGA * max(0.0, water_ratio - W_SLIP) ** 2)
    return print_speed * slip_factor


def _exp_safe(x: float) -> float:
    """Numerically safe exponential (clamp to avoid overflow)."""
    if x > 500:
        return float("inf")
    if x < -500:
        return 0.0
    import math
    return math.exp(x)
