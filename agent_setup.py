"""Build and configure the PfabAgent for the extrusion printing showcase."""

from pred_fab import PfabAgent
from pred_fab.core import DatasetSchema

from sensors.camera import CameraSystem
from sensors.energy import EnergySensor
from models.feature_models import PrintingFeatureModel, EnergyFeatureModel, ProductionRateFeatureModel
from models.evaluation_models import PathAccuracyModel, EnergyConsumptionModel, ProductionRateModel
from models.prediction_model import DeviationPredictionModel, EnergyPredictionModel, ProductionRatePredictionModel


def build_agent(
    schema: DatasetSchema,
    camera: CameraSystem,
    energy_sensor: EnergySensor,
) -> PfabAgent:
    """Register all models, initialize systems, and return a configured PfabAgent."""
    agent = PfabAgent(root_folder="./pfab_data")

    # Register models
    agent.register_feature_model(PrintingFeatureModel, camera=camera)
    agent.register_feature_model(EnergyFeatureModel, energy_sensor=energy_sensor)
    agent.register_feature_model(ProductionRateFeatureModel)
    agent.register_evaluation_model(PathAccuracyModel)
    agent.register_evaluation_model(EnergyConsumptionModel)
    agent.register_evaluation_model(ProductionRateModel)
    agent.register_prediction_model(DeviationPredictionModel)
    agent.register_prediction_model(EnergyPredictionModel)
    agent.register_prediction_model(ProductionRatePredictionModel)

    # Initialize all systems against the schema
    agent.initialize_systems(schema, verbose_flag=True)

    return agent
