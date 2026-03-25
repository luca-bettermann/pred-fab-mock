"""Shared helper utilities for the pred-fab-mock journey script."""

from typing import Any, Dict


def params_from_spec(spec: Any) -> Dict[str, Any]:
    """Extract a plain params dict from an ExperimentSpec."""
    return dict(spec.initial_params.to_dict())


def get_performance(exp_data: Any) -> Dict[str, float]:
    """Extract available performance values from an evaluated ExperimentData."""
    return {
        name: float(val)
        for name, val in exp_data.performance.get_values_dict().items()
        if val is not None
    }
