"""Build + configure the PfabAgent for the ADVEI 2026 mock.

Wires the synthetic FabricationSystem as the feature source (in place of
learning-by-printing's real sensor stack), registers the shared evaluation +
prediction models, and configures the κ-blend acquisition. Everything below the
feature layer is identical to the real study.
"""
from __future__ import annotations

from pred_fab import PfabAgent
from pred_fab.core import DatasetSchema

from fabrication import FabricationSystem
from models.schema import (
    AxisCode, ParamCode, N_NODES,
    TARGET_FILAMENT_WIDTH_MM, ENERGY_MIN_J, ENERGY_MAX_J,
    DURATION_MIN_S, DURATION_MAX_S, derive_n_layers, build_advei_dataset_schema,
)
from models.features import (
    NodeVisionFeature, NodeAggregateFeature, LoadcellFeature,
    RobotEnergyFeature, DurationFeature, EnvironmentFeature,
)
from models.evaluations import (
    StructuralIntegrityEval, MaterialDepositionEval, ExtrusionStabilityEval,
    EnergyFootprintEval, FabricationTimeEval,
)
from models.predictions import StructuralMLP, DeterministicDuration

_LH_DEFAULT = 2.5  # layer-height fallback when deriving n_layers


def build_schema(root_folder: str = ".") -> DatasetSchema:
    """The canonical ADVEI 2026 schema (see models/schema.py)."""
    return build_advei_dataset_schema(root_folder)


def build_fab(random_seed: int | None = None) -> FabricationSystem:
    """The synthetic fabrication harness that produces all feature values."""
    return FabricationSystem(random_seed=random_seed)


def build_agent(schema: DatasetSchema, fab: FabricationSystem, *, verbose: bool = True) -> PfabAgent:
    """Register models, initialise systems, configure the acquisition."""
    agent = PfabAgent(root_folder=".")

    # Feature models — one per modality, all reading the shared FabricationSystem.
    agent.register_feature_model(NodeVisionFeature, fab=fab)
    agent.register_feature_model(NodeAggregateFeature, fab=fab)
    agent.register_feature_model(LoadcellFeature, fab=fab)
    agent.register_feature_model(RobotEnergyFeature, fab=fab)
    agent.register_feature_model(DurationFeature, fab=fab)
    agent.register_feature_model(EnvironmentFeature, fab=fab)

    # Evaluation models — paper-matching scoring + bounds.
    agent.register_evaluation_model(StructuralIntegrityEval, target_overlap_mm=TARGET_FILAMENT_WIDTH_MM)
    agent.register_evaluation_model(MaterialDepositionEval, target_width_mm=TARGET_FILAMENT_WIDTH_MM, half_range_mm=1.0)
    agent.register_evaluation_model(ExtrusionStabilityEval)
    agent.register_evaluation_model(EnergyFootprintEval, target_energy=ENERGY_MIN_J, max_energy=ENERGY_MAX_J)
    agent.register_evaluation_model(FabricationTimeEval, duration_min_s=DURATION_MIN_S, duration_max_s=DURATION_MAX_S)

    # Prediction — the paper's MLP (4 uncertain depth-1 features) plus the
    # closed-form duration model.
    agent.register_prediction_model(StructuralMLP)
    agent.register_prediction_model(DeterministicDuration)

    agent.initialize_systems(schema, verbose_flag=verbose)

    # Variable sequence length: n_layers = ceil(component_height / layer_height).
    agent.calibration_system.dimension_derivations[AxisCode.LAYERS] = (
        lambda p: derive_n_layers(float(p.get(ParamCode.LAYER_HEIGHT, _LH_DEFAULT)))
    )
    agent.calibration_system.dimension_derivations[AxisCode.NODES] = lambda p: N_NODES

    return agent
