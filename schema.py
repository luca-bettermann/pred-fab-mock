"""Build the DatasetSchema for the ADVEI 2026 mock fabrication simulation.

ADVEI study — curved-wall clay extrusion. Five process parameters split
into per-print (static) and per-layer (trajectory) groups; one structural
domain (n_layers, n_nodes); five quality/cost performance attributes; six
target features at mixed depths plus two context features.

The mock simulates fabrication-time signals at the *feature* level rather
than at the raw-sensor level: ``sensors/physics.py`` produces synthetic
node_overlap, filament_width, extrusion_consistency, current_*, and
printing_duration values directly. The real-fabrication counterpart in
``learning-by-printing`` extracts the same features from real sensors.

Constants for the simulation (component height, supply voltage, etc.) are
hardcoded in ``sensors/physics.py``; only optimisable parameters live in
the schema.
"""

from pred_fab.core import (
    DatasetSchema,
    Parameters,
    Features,
    PerformanceAttributes,
    Dimension,
    Domain,
    Domains,
    Parameter,
    Feature,
    PerformanceAttribute,
)


ROOT_FOLDER = "."
SCHEMA_NAME = "advei_2026_mock"
SCHEMA_TITLE = "ADVEI 2026 — clay extrusion (mock)"


# --- Parameter bounds (also exported for the CCF grid builder) ---------------
# (code, low, high) tuples. Order is the factor labelling for grid-design
# generation; mirrors the canonical PARAM_BOUNDS in
# learning-by-printing/models/schema.py.
PARAM_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("path_offset",        0.0,   3.0),     # mm; per-print
    ("layer_height",       2.0,   3.0),     # mm; per-print → derives n_layers
    ("calibration_factor", 1.6,   2.2),     # dimensionless; per-print
    ("print_speed",        0.004, 0.008),   # m/s; per-layer (runtime)
    ("slowdown_factor",    0.0,   1.0),     # dimensionless; per-layer (runtime)
)

STATIC_PARAMS = ("path_offset", "layer_height", "calibration_factor")
TRAJECTORY_PARAMS = ("print_speed", "slowdown_factor")


def build_schema(root_folder: str = ROOT_FOLDER) -> DatasetSchema:
    """Construct the ADVEI 2026 mock schema."""
    params = Parameters([
        # Static (per-print)
        Parameter.real("path_offset",        min_val=0.0,   max_val=3.0),
        Parameter.real("layer_height",       min_val=2.0,   max_val=3.0),
        Parameter.real("calibration_factor", min_val=1.6,   max_val=2.2),
        # Trajectory (per-layer, runtime-schedulable)
        Parameter.real("print_speed",        min_val=0.004, max_val=0.008, runtime=True),
        Parameter.real("slowdown_factor",    min_val=0.0,   max_val=1.0,   runtime=True),
    ])

    # Tensor shape is fixed at max: 15 layers × 7 nodes. Experiments with
    # layer_height > 2.0 have fewer actual layers; padded positions are zero.
    structural = Domain("structural", [
        Dimension("n_layers", "layer_idx", min_val=15, max_val=15),
        Dimension("n_nodes",  "node_idx",  min_val=7,  max_val=7),
    ])
    domains = Domains([structural])

    features = Features([
        # Quality targets — depth 2 (per layer × node)
        Feature("node_overlap",         domain=structural, depth=2),
        Feature("filament_width",       domain=structural, depth=2),
        # Quality target — depth 1 (per layer)
        Feature("extrusion_consistency", domain=structural, depth=1),
        # Cost-driving targets — depth 1 (per layer)
        Feature("current_mean_feeder",  domain=structural, depth=1),
        Feature("current_mean_nozzle",  domain=structural, depth=1),
        Feature("printing_duration",    domain=structural, depth=1),
        # Context (uncontrollable, BME280) — depth 1
        Feature("temperature",          domain=structural, depth=1, context=True),
        Feature("humidity",             domain=structural, depth=1, context=True),
    ])

    performance = PerformanceAttributes([
        PerformanceAttribute.score("structural_integrity"),
        PerformanceAttribute.score("material_deposition"),
        PerformanceAttribute.score("extrusion_stability"),
        PerformanceAttribute.score("energy_footprint"),
        PerformanceAttribute.score("fabrication_time"),
    ])

    return DatasetSchema(
        root_folder=root_folder,
        name=SCHEMA_NAME,
        parameters=params,
        features=features,
        performance=performance,
        domains=domains,
    )
