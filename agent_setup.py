"""Build and configure the PfabAgent for the ADVEI 2026 mock."""

from pred_fab import PfabAgent
from pred_fab.core import DatasetSchema

from sensors.fabrication import FabricationSystem
from models.feature_models import (
    NodeVisionFeature,
    LoadcellConsistencyFeature,
    RobotEnergyFeature,
    DurationFeature,
    EnvironmentFeature,
)
from models.evaluation_models import (
    StructuralIntegrityEval,
    MaterialDepositionEval,
    ExtrusionStabilityEval,
    EnergyFootprintEval,
    FabricationTimeEval,
)
from models.prediction_model import StructuralTransformer, DeterministicDuration


def build_agent(
    schema: DatasetSchema,
    fab: FabricationSystem,
    verbose: bool = True,
) -> PfabAgent:
    """Register all models, initialize systems, and return a configured PfabAgent."""
    agent = PfabAgent(root_folder=".")

    # Feature models (one per modality, mirroring learning-by-printing layout)
    agent.register_feature_model(NodeVisionFeature, fab=fab)
    agent.register_feature_model(LoadcellConsistencyFeature, fab=fab)
    agent.register_feature_model(RobotEnergyFeature, fab=fab)
    agent.register_feature_model(DurationFeature, fab=fab)
    agent.register_feature_model(EnvironmentFeature, fab=fab)

    # Evaluation models — one per performance attribute
    agent.register_evaluation_model(StructuralIntegrityEval)
    agent.register_evaluation_model(MaterialDepositionEval)
    agent.register_evaluation_model(ExtrusionStabilityEval)
    agent.register_evaluation_model(EnergyFootprintEval)
    agent.register_evaluation_model(FabricationTimeEval)

    # Prediction models — multi-depth transformer + deterministic duration
    agent.register_prediction_model(StructuralTransformer)
    agent.register_prediction_model(DeterministicDuration)

    agent.initialize_systems(schema, verbose_flag=verbose)

    return agent
