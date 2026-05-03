"""Small helpers reused across step scripts."""

from __future__ import annotations

from typing import Any


def params_from_spec(spec: Any) -> dict[str, Any]:
    """Extract a flat parameter dict from an ExperimentSpec / ParameterProposal.

    Tolerates both ExperimentSpec (delegates via __getitem__/keys) and a plain
    dict-like proposal. Returns a regular dict the workflow can pass into
    ``Dataset.create_experiment``.
    """
    if hasattr(spec, "initial_params"):
        return dict(spec.initial_params.values)
    if hasattr(spec, "values"):
        return dict(spec.values)
    return dict(spec)


def get_performance(exp_data: Any) -> dict[str, float]:
    """Read the final performance dict off an evaluated ExperimentData."""
    return {k: float(v) for k, v in exp_data.performance.get_values_dict().items()}
