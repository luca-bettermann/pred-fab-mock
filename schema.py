"""Build the DatasetSchema for the extrusion printing mock."""

from pred_fab.core import (
    DatasetSchema,
    Parameters,
    Features,
    PerformanceAttributes,
    Parameter,
    Feature,
    PerformanceAttribute,
)

ROOT_FOLDER = "./pfab_data"
SCHEMA_NAME = "extrusion_printing_v1"


def build_schema() -> DatasetSchema:
    """Construct and return the DatasetSchema for the extrusion printing simulation."""
    params = Parameters()
    params.add("layer_time",    Parameter.real("layer_time",    min_val=20.0,  max_val=70.0))
    params.add("layer_height",  Parameter.real("layer_height",  min_val=0.005, max_val=0.010))
    params.add("water_ratio",   Parameter.real("water_ratio",   min_val=0.30,  max_val=0.50))
    params.add("design",        Parameter.categorical("design",   ["A", "B", "C"]))
    params.add("material",      Parameter.categorical("material", ["standard", "reinforced", "flexible"]))
    params.add("print_speed",   Parameter.real("print_speed",   min_val=20.0,  max_val=60.0, runtime=True))
    params.add("n_layers",      Parameter.dimension("n_layers",   iterator_code="layer_idx",   level=1, min_val=5, max_val=5))
    params.add("n_segments",    Parameter.dimension("n_segments", iterator_code="segment_idx", level=2, min_val=4, max_val=4))

    features = Features()
    features.add("layer_width",       Feature.array("layer_width"))
    features.add("path_deviation",    Feature.array("path_deviation"))
    features.add("energy_per_segment", Feature.array("energy_per_segment"))

    performance = PerformanceAttributes()
    performance.add("path_accuracy",     PerformanceAttribute.score("path_accuracy"))
    performance.add("energy_efficiency", PerformanceAttribute.score("energy_efficiency"))

    return DatasetSchema(
        root_folder=ROOT_FOLDER,
        name=SCHEMA_NAME,
        parameters=params,
        features=features,
        performance=performance,
    )
