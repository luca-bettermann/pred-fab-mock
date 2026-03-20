"""Build the DatasetSchema for the extrusion printing mock."""

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

ROOT_FOLDER = "./pfab_data"
SCHEMA_NAME = "extrusion_printing_v2"


def build_schema() -> DatasetSchema:
    """Construct and return the DatasetSchema for the extrusion printing simulation."""
    # --- Parameters (optimization parameters only — no dimension params) ---
    # Domain axes (n_layers, n_segments) are declared in the domain below.
    water_ratio = Parameter.real("water_ratio", min_val=0.30, max_val=0.50)
    design      = Parameter.categorical("design",   ["A", "B", "C"])
    material    = Parameter.categorical("material", ["standard", "reinforced", "flexible"])
    print_speed = Parameter.real("print_speed", min_val=20.0, max_val=60.0, runtime=True)

    params = Parameters()
    for p in [water_ratio, design, material, print_speed]:
        params.add(p)

    # --- Domains ---
    # spatial_segment: (layer, segment) — the measurement space for all sensors.
    # Axis sizes are fixed per design (5 layers × 4 segments = 20 evaluation steps).
    n_layers   = Dimension("n_layers",   "layer_idx",   min_val=5, max_val=5)
    n_segments = Dimension("n_segments", "segment_idx", min_val=4, max_val=4)
    spatial = Domain("spatial_segment", [n_layers, n_segments])
    domains = Domains()
    domains.add(spatial)

    # --- Features (tied to domain; depth=None means full domain depth) ---
    layer_width        = Feature.array("layer_width",        domain="spatial_segment")
    path_deviation     = Feature.array("path_deviation",     domain="spatial_segment")
    energy_per_segment = Feature.array("energy_per_segment", domain="spatial_segment")

    features = Features()
    for f in [layer_width, path_deviation, energy_per_segment]:
        features.add(f)

    # --- Performance ---
    path_accuracy     = PerformanceAttribute.score("path_accuracy")
    energy_efficiency = PerformanceAttribute.score("energy_efficiency")

    performance = PerformanceAttributes()
    for a in [path_accuracy, energy_efficiency]:
        performance.add(a)

    return DatasetSchema(
        root_folder=ROOT_FOLDER,
        name=SCHEMA_NAME,
        parameters=params,
        features=features,
        performance=performance,
        domains=domains,
    )
