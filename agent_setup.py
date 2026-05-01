"""Build and configure the PfabAgent for the extrusion printing showcase."""

from pred_fab import PfabAgent
from pred_fab.core import DatasetSchema

from sensors.camera import CameraSystem
from sensors.energy import EnergySensor
from models.feature_models import DevFeature, EnergyFeature, RateFeature
from models.evaluation_models import PathAccuracy, EnergyEfficiency, ProductionRate
from models.prediction_model import DevTransformer, EnergyMLP, RateMLP


def build_agent(
    schema: DatasetSchema,
    camera: CameraSystem,
    energy_sensor: EnergySensor,
    verbose: bool = True,
) -> PfabAgent:
    """Register all models, initialize systems, and return a configured PfabAgent."""
    agent = PfabAgent(root_folder=".")

    agent.register_feature_model(DevFeature, camera=camera)
    agent.register_feature_model(EnergyFeature, energy_sensor=energy_sensor)
    agent.register_feature_model(RateFeature)
    agent.register_evaluation_model(PathAccuracy)
    agent.register_evaluation_model(EnergyEfficiency)
    agent.register_evaluation_model(ProductionRate)

    agent.register_prediction_model(DevTransformer)
    agent.register_prediction_model(EnergyMLP)
    agent.register_prediction_model(RateMLP)

    agent.initialize_systems(schema, verbose_flag=verbose)

    return agent
