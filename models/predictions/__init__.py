"""Prediction models — parameters → predicted features."""
from .structural_mlp import StructuralMLP
from .deterministic_duration import DeterministicDuration

__all__ = ["StructuralMLP", "DeterministicDuration"]
