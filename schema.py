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
SCHEMA_NAME = "extrusion_printing_v6"


def build_schema(root_folder: str = ROOT_FOLDER) -> DatasetSchema:
    """Construct and return the DatasetSchema for the extrusion printing simulation."""
    # --- Parameters (optimization parameters only — no dimension params) ---
    params = Parameters([
        Parameter.real("water_ratio", min_val=0.30, max_val=0.50),
        Parameter.categorical("design",   ["A", "B"]),
        Parameter.categorical("material", ["clay", "concrete"]),
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

    # --- Features ---
    # path_deviation and energy_per_segment vary per (layer, segment) position.
    # production_rate is constant across positions (depends only on process params),
    # so it is declared as a scalar feature with no spatial domain.
    path_dev = Feature.array("path_deviation", domain=spatial)
    layer_dim, segment_dim = spatial.axes

    features = Features([
        path_dev,
        Feature.array("energy_per_segment", domain=spatial),
        Feature.array("production_rate"),
        Feature.recursive("prev_layer_deviation",   source=path_dev, dimensions=(layer_dim,)),
        Feature.recursive("prev_segment_deviation", source=path_dev, dimensions=(segment_dim,)),
    ])
    # --- Performance ---
    performance = PerformanceAttributes([
        PerformanceAttribute.score("path_accuracy"),
        PerformanceAttribute.score("energy_efficiency"),
        PerformanceAttribute.score("production_rate"),
    ])

    return DatasetSchema(
        root_folder=root_folder,
        name=SCHEMA_NAME,
        parameters=params,
        features=features,
        performance=performance,
        domains=domains,
    )
