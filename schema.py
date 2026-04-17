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

ROOT_FOLDER = "."
SCHEMA_NAME = "extrusion_printing_v8"
SCHEMA_TITLE = "Extrusion-based Additive Manufacturing"


def build_schema(root_folder: str = ROOT_FOLDER) -> DatasetSchema:
    """Construct and return the DatasetSchema for the extrusion printing simulation.

    Two continuous parameters (water_ratio, print_speed) over a fixed
    spatial domain (5 layers x 4 segments). Recursive features at lag 1 and 2.
    """
    params = Parameters([
        Parameter.real("water_ratio", min_val=0.30, max_val=0.50),
        Parameter.real("print_speed", min_val=20.0, max_val=60.0, runtime=True),
    ])

    # Spatial domain: n_layers is a design intent variable [3..8], n_segments fixed at 4
    spatial = Domain("spatial_segment", [
        Dimension("n_layers",   "layer_idx",   min_val=4, max_val=8),
        Dimension("n_segments", "segment_idx", min_val=4, max_val=4),
    ])
    domains = Domains([spatial])

    # Features
    path_dev = Feature.array("path_deviation", domain=spatial)
    layer_dim, segment_dim = spatial.axes

    features = Features([
        path_dev,
        Feature.array("energy_per_segment", domain=spatial),
        Feature.array("production_rate"),
        # Recursive: lag 1 along each dimension
        *Feature.recursive("prev_layer_dev",   source=path_dev, dimensions=(layer_dim,),   max_depth=1),
        *Feature.recursive("prev_seg_dev", source=path_dev, dimensions=(segment_dim,), max_depth=1),
    ])

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
