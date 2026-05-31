"""Schema-level constants and the ADVEI 2026 ``DatasetSchema`` builder.

Single source of truth for the mock's cross-module identifiers — parameter,
feature, and performance-attribute codes, study constants, and display labels.
Mirrors ``learning-by-printing/models/schema.py`` (the real-fabrication
counterpart) but drops everything hardware/wire-specific: there is no rtde
name mapping, no InfluxDB/NocoDB schema, no sensor identifiers. The mock
simulates at the *feature level*, so it speaks the clean ADVEI codes directly.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pred_fab.core import DatasetSchema


# === Domain iterator + axis codes ============================================

LAYER_ITERATOR_CODE = "layer_idx"
NODE_ITERATOR_CODE = "node_idx"
STRUCTURAL_DOMAIN = "structural"


class AxisCode:
    """Domain-axis (Dimension) codes — distinct from the iterator code that
    counts inside the axis. Sequence-aware models declare ``sequence_axis_code``
    using these."""

    LAYERS = "n_layers"
    NODES = "n_nodes"


# === Parameters ==============================================================

class ParamCode:
    """Process-parameter codes (the optimisation surface)."""

    PATH_OFFSET = "path_offset"
    LAYER_HEIGHT = "layer_height"
    CALIBRATION_FACTOR = "calibration_factor"
    PRINT_SPEED = "print_speed"


PARAM_BOUNDS: tuple[tuple[str, float, float], ...] = (
    (ParamCode.PATH_OFFSET,        1.0,  3.0),    # mm; per-print
    (ParamCode.LAYER_HEIGHT,       2.0,  3.0),    # mm; per-print → derives n_layers
    (ParamCode.CALIBRATION_FACTOR, 1.8,  2.2),    # dimensionless; per-print
    (ParamCode.PRINT_SPEED,        0.05, 0.1),    # m/s; per-layer (runtime)
)

PARAM_ROUND_DIGITS: dict[str, int] = {
    ParamCode.PATH_OFFSET:        1,
    ParamCode.LAYER_HEIGHT:        2,
    ParamCode.CALIBRATION_FACTOR: 2,
    ParamCode.PRINT_SPEED:        5,
}

# Per-print (static) params — one value per experiment.
STATIC_PARAMS = (ParamCode.PATH_OFFSET, ParamCode.LAYER_HEIGHT, ParamCode.CALIBRATION_FACTOR)
# Per-layer (trajectory) params — re-proposable along the layer axis.
TRAJECTORY_PARAMS = (ParamCode.PRINT_SPEED,)

# Fixed study constant (held at 0 for ADVEI; the corner-slowdown axis is not
# part of this paper). Kept so the simulator's corner physics has a defined value.
SLOWDOWN_FACTOR = 0.0


# === Features ================================================================

class FeatureCode:
    """Feature codes the simulator emits and the schema declares."""

    # Quality — per (layer, node)
    NODE_OVERLAP = "node_overlap"
    FILAMENT_WIDTH = "filament_width"
    # Quality / cost — per layer
    LOADCELL_RESIDUAL = "loadcell_residual"
    ROBOT_ENERGY = "robot_energy"
    PRINTING_DURATION = "printing_duration"
    # Per-layer aggregates of the per-node quality features
    NODE_OVERLAP_MEAN = "node_overlap_mean"
    FILAMENT_WIDTH_MEAN = "filament_width_mean"
    # Context (uncontrollable) — per layer
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    MATERIAL_AGE = "material_age"


# === Performance attributes ==================================================

class AttributeCode:
    """Performance attribute codes (derived from features by the evaluators)."""

    STRUCTURAL_INTEGRITY = "structural_integrity"   # per (layer, node)
    MATERIAL_DEPOSITION = "material_deposition"     # per (layer, node)
    EXTRUSION_STABILITY = "extrusion_stability"     # per layer
    ENERGY_FOOTPRINT = "energy_footprint"           # per experiment (sum over layers)
    FABRICATION_TIME = "fabrication_time"           # per experiment (sum over layers)


# === Display labels — paper-aligned naming for plots/reports =================

DISPLAY_LABELS: dict[str, str] = {
    ParamCode.PATH_OFFSET:              "Path offset",
    ParamCode.LAYER_HEIGHT:             "Layer height",
    ParamCode.CALIBRATION_FACTOR:       "Calibration factor",
    ParamCode.PRINT_SPEED:              "Print speed",
    FeatureCode.TEMPERATURE:            "Temperature",
    FeatureCode.HUMIDITY:               "Humidity",
    FeatureCode.MATERIAL_AGE:           "Batch age",
    FeatureCode.NODE_OVERLAP:           "Node overlap",
    FeatureCode.FILAMENT_WIDTH:         "Filament width",
    FeatureCode.NODE_OVERLAP_MEAN:      "Node overlap",
    FeatureCode.FILAMENT_WIDTH_MEAN:    "Filament width",
    FeatureCode.LOADCELL_RESIDUAL:      "Loadcell residual",
    FeatureCode.ROBOT_ENERGY:           "Robot energy",
    FeatureCode.PRINTING_DURATION:      "Printing duration",
    AttributeCode.STRUCTURAL_INTEGRITY: "Structural integrity",
    AttributeCode.MATERIAL_DEPOSITION:  "Material deposition",
    AttributeCode.EXTRUSION_STABILITY:  "Extrusion stability",
    AttributeCode.ENERGY_FOOTPRINT:     "Energy footprint",
    AttributeCode.FABRICATION_TIME:     "Fabrication time",
}

DISPLAY_UNITS: dict[str, str] = {
    ParamCode.PATH_OFFSET:           "mm",
    ParamCode.LAYER_HEIGHT:          "mm",
    ParamCode.PRINT_SPEED:           "m/s",
    FeatureCode.TEMPERATURE:         "°C",
    FeatureCode.HUMIDITY:            "%",
    FeatureCode.MATERIAL_AGE:        "h",
    FeatureCode.NODE_OVERLAP:        "mm",
    FeatureCode.FILAMENT_WIDTH:      "mm",
    FeatureCode.NODE_OVERLAP_MEAN:   "mm",
    FeatureCode.FILAMENT_WIDTH_MEAN: "mm",
    FeatureCode.LOADCELL_RESIDUAL:   "g²",
    FeatureCode.ROBOT_ENERGY:        "J",
    FeatureCode.PRINTING_DURATION:   "s",
}


def display_name(code: str) -> str:
    """Paper-aligned display name; falls back to the raw code if unknown."""
    return DISPLAY_LABELS.get(code, code)


def axis_label(code: str) -> str:
    """'Display Name [unit]' for plot axes; omits brackets when no unit is registered."""
    name = DISPLAY_LABELS.get(code, code)
    unit = DISPLAY_UNITS.get(code)
    return f"{name} [{unit}]" if unit else name


# === Study constants =========================================================

COMPONENT_HEIGHT_MM = 25.0           # curved-wall height; n_layers = ceil(height / layer_height)
TARGET_FILAMENT_WIDTH_MM = 7.0       # per-node target for material_deposition + structural_integrity

# Robot-energy evaluator bounds (Joules per experiment, summed across layers).
ENERGY_MIN_J = 70000.0               # score = 1 at/below this
ENERGY_MAX_J = 185000.0              # score = 0 above this
PATH_LENGTH_M = 1.149                # toolpath length per layer (contour + 7-node infill)

# Fabrication-time evaluator bounds (seconds per experiment, summed across layers).
DURATION_MIN_S = 110.0               # score = 1 below this
DURATION_MAX_S = 300.0               # score = 0 above this

# Domain-axis bounds. n_layers varies per experiment (derived from layer_height);
# the schema declares the full [min, max] range. n_nodes is constant at 7.
_LH_LO, _LH_HI = {c: (lo, hi) for c, lo, hi in PARAM_BOUNDS}[ParamCode.LAYER_HEIGHT]
MIN_N_LAYERS = int(math.ceil(COMPONENT_HEIGHT_MM / _LH_HI))   # max layer_height → fewest layers
MAX_N_LAYERS = int(math.ceil(COMPONENT_HEIGHT_MM / _LH_LO))   # min layer_height → most layers
N_NODES = 7

ADVEI_STUDY_CODE = "ADVEI_2026"


def derive_n_layers(layer_height_mm: float, component_height_mm: float = COMPONENT_HEIGHT_MM) -> int:
    """Number of print layers for a given layer height: ``ceil(component_height / layer_height)``."""
    if layer_height_mm <= 0:
        raise ValueError(f"layer_height_mm must be > 0; got {layer_height_mm}")
    return int(math.ceil(component_height_mm / layer_height_mm))


def build_advei_dataset_schema(root_folder: str) -> "DatasetSchema":
    """Construct the canonical ADVEI 2026 ``DatasetSchema`` — identical in shape
    to the learning-by-printing study (4 params, 10 features at mixed depths,
    5 performance attributes, one structural layer×node domain)."""
    from pred_fab.core import (
        DatasetSchema, Dimension, Domain, Domains,
        Feature, Features, Parameter, Parameters,
        PerformanceAttribute, PerformanceAttributes,
    )

    trajectory_codes = set(TRAJECTORY_PARAMS)
    parameters = Parameters([
        Parameter.real(
            code, min_val=low, max_val=high,
            runtime=(code in trajectory_codes),
            round_digits=PARAM_ROUND_DIGITS[code],
        )
        for code, low, high in PARAM_BOUNDS
    ])

    structural = Domain(STRUCTURAL_DOMAIN, [
        Dimension(AxisCode.LAYERS, LAYER_ITERATOR_CODE, min_val=MIN_N_LAYERS, max_val=MAX_N_LAYERS),
        Dimension(AxisCode.NODES, NODE_ITERATOR_CODE, min_val=N_NODES, max_val=N_NODES),
    ])
    domains = Domains([structural])

    features = Features([
        Feature(FeatureCode.NODE_OVERLAP,        domain=structural, depth=2),
        Feature(FeatureCode.FILAMENT_WIDTH,      domain=structural, depth=2),
        Feature(FeatureCode.LOADCELL_RESIDUAL,   domain=structural, depth=1),
        Feature(FeatureCode.ROBOT_ENERGY,        domain=structural, depth=1),
        Feature(FeatureCode.PRINTING_DURATION,   domain=structural, depth=1),
        Feature(FeatureCode.NODE_OVERLAP_MEAN,   domain=structural, depth=1),
        Feature(FeatureCode.FILAMENT_WIDTH_MEAN, domain=structural, depth=1),
        Feature(FeatureCode.TEMPERATURE,         domain=structural, depth=1, context=True),
        Feature(FeatureCode.HUMIDITY,            domain=structural, depth=1, context=True),
        Feature(FeatureCode.MATERIAL_AGE,        domain=structural, depth=1, context=True),
    ])

    performance = PerformanceAttributes([
        PerformanceAttribute.score(AttributeCode.STRUCTURAL_INTEGRITY),
        PerformanceAttribute.score(AttributeCode.MATERIAL_DEPOSITION),
        PerformanceAttribute.score(AttributeCode.EXTRUSION_STABILITY),
        PerformanceAttribute.score(AttributeCode.ENERGY_FOOTPRINT),
        PerformanceAttribute.score(AttributeCode.FABRICATION_TIME),
    ])

    return DatasetSchema(
        root_folder=root_folder,
        name=ADVEI_STUDY_CODE,
        parameters=parameters,
        features=features,
        performance=performance,
        domains=domains,
    )
