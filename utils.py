"""Shared helper utilities for the pred-fab-mock journey script."""

from contextlib import contextmanager
from typing import Any, Dict, Iterator

from pred_fab.utils import PfabLogger


@contextmanager
def quiet_console(logger: PfabLogger) -> Iterator[None]:
    """Temporarily suppress console output from the logger."""
    logger.set_console_output(False)
    try:
        yield
    finally:
        logger.set_console_output(True)


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
