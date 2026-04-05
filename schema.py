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
    params = Parameters([
        Parameter.real("water_ratio", min_val=0.30, max_val=0.50),
        Parameter.categorical("design",   ["A", "B", "C"]),
        Parameter.categorical("material", ["standard", "reinforced", "flexible"]),
        Parameter.real("print_speed", min_val=20.0, max_val=60.0, runtime=True),
    ])

    # --- Domains ---
    # spatial_segment: (layer, segment) — the measurement space for all sensors.
    # Axis sizes are fixed per design (5 layers × 4 segments = 20 evaluation steps).
    spatial = Domain("spatial_segment", [
        Dimension("n_layers",   "layer_idx",   min_val=5, max_val=5),
        Dimension("n_segments", "segment_idx", min_val=4, max_val=4),
    ])
    domains = Domains([spatial])

    # --- Features (tied to domain; depth=None means full domain depth) ---
    features = Features([
        Feature.array("layer_width",        domain=spatial),
        Feature.array("path_deviation",     domain=spatial),
        Feature.array("energy_per_segment", domain=spatial),
    ])
    # --- Performance ---
    performance = PerformanceAttributes([
        PerformanceAttribute.score("path_accuracy"),
        PerformanceAttribute.score("energy_efficiency"),
    ])

    return DatasetSchema(
        root_folder=ROOT_FOLDER,
        name=SCHEMA_NAME,
        parameters=params,
        features=features,
        performance=performance,
        domains=domains,
    )

# Note: layer_width has no evaluation model (no width-accuracy metric), so a
# "layer_width unused" warning is expected and intentional.
