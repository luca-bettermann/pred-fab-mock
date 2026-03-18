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
    # --- Parameters ---
    # layer_time and layer_height are fabrication constants owned by FabricationSystem:
    #   - layer_time: fixed process duration per layer
    #   - layer_height: derived from design as target_height / n_layers, so the
    #     component height is a design property, not an optimization degree of freedom.
    # n_layers and n_segments are also derived from design (via FabricationSystem.get_dimensions).
    water_ratio = Parameter.real("water_ratio", min_val=0.30, max_val=0.50)
    design      = Parameter.categorical("design",   ["A", "B", "C"])
    material    = Parameter.categorical("material", ["standard", "reinforced", "flexible"])
    print_speed = Parameter.real("print_speed", min_val=20.0, max_val=60.0, runtime=True)
    n_layers    = Parameter.dimension("n_layers",   iterator_code="layer_idx",   level=1, min_val=5, max_val=5)
    n_segments  = Parameter.dimension("n_segments", iterator_code="segment_idx", level=2, min_val=4, max_val=4)

    params = Parameters()
    for p in [water_ratio, design, material, print_speed, n_layers, n_segments]:
        params.add(p.code, p)

    # --- Features ---
    layer_width        = Feature.array("layer_width")
    path_deviation     = Feature.array("path_deviation")
    energy_per_segment = Feature.array("energy_per_segment")

    features = Features()
    for f in [layer_width, path_deviation, energy_per_segment]:
        features.add(f.code, f)

    # --- Performance ---
    path_accuracy     = PerformanceAttribute.score("path_accuracy")
    energy_efficiency = PerformanceAttribute.score("energy_efficiency")

    performance = PerformanceAttributes()
    for a in [path_accuracy, energy_efficiency]:
        performance.add(a.code, a)

    return DatasetSchema(
        root_folder=ROOT_FOLDER,
        name=SCHEMA_NAME,
        parameters=params,
        features=features,
        performance=performance,
    )
