"""FabricationSystem — feature-level mock fabrication for ADVEI 2026.

Simulates one print run by computing every feature value directly from the
synthetic physics in :mod:`sensors.physics`. No raw sensor data is produced
(per-print ADVEI design choice — the real-fabrication counterpart in
``learning-by-printing`` does the raw-sensor-to-feature extraction).

Each call to :meth:`run_experiment` populates a per-experiment cache keyed
by ``(params_hash, layer_idx, node_idx)`` for depth-2 features and
``(params_hash, layer_idx)`` for depth-1 features. The mock feature models
in ``models/feature_models.py`` then read these cached values.

Trajectory params (``print_speed``, ``slowdown_factor``) can vary per
layer when an :class:`ExperimentData`'s parameter_updates contain per-layer
overrides; FabricationSystem reads the effective value at each layer via
``ExperimentData.get_effective_parameters_for_row`` (consumed via the
workflow helper that hands params down).
"""

from __future__ import annotations

import json
from typing import Any

from . import physics


class FabricationSystem:
    """Coordinates feature-level simulation for a single print run.

    Use :meth:`run_experiment` once per ``(params, n_layers, n_nodes)``
    triple; subsequent reads via :meth:`get_node_feature` or
    :meth:`get_layer_feature` hit the cache.
    """

    def __init__(self, random_seed: int | None = None) -> None:
        # Seed reserved for future per-feature noise injections (the current
        # physics is fully deterministic, so the seed is unused but kept on
        # the constructor for API stability with the legacy mock.)
        self._random_seed = random_seed
        self._node_cache: dict[tuple, dict[str, float]] = {}
        self._layer_cache: dict[tuple, dict[str, float]] = {}

    @staticmethod
    def _params_key(params: dict[str, Any]) -> str:
        return json.dumps({k: float(v) for k, v in params.items() if isinstance(v, (int, float))}, sort_keys=True)

    @staticmethod
    def get_dimensions(params: dict[str, Any]) -> tuple[int, int]:
        """Return ``(n_layers, n_nodes)`` — always the fixed tensor shape."""
        n_layers = int(params.get("n_layers", physics.MAX_N_LAYERS))
        n_nodes = int(params.get("n_nodes", 7))
        return n_layers, n_nodes

    # --- Public API ----------------------------------------------------------

    def run_experiment(self, params: dict[str, Any]) -> None:
        """Populate the cache for every (layer, node) cell of one experiment.

        All MAX_N_LAYERS layers are simulated — ``layer_height`` affects the
        per-layer physics but doesn't truncate the sequence.
        """
        n_layers, n_nodes = self.get_dimensions(params)
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx, n_nodes=n_nodes)

    def run_layer(
        self,
        params: dict[str, Any],
        layer_idx: int,
        n_nodes: int = 7,
    ) -> None:
        """Simulate and cache all per-layer + per-node features for one layer.

        ``params`` should be the *effective* parameter dict at this layer
        (i.e. with per-layer trajectory overrides already applied — the
        workflow helper handles that).
        """
        key_static = self._params_key(params)
        layer_key = (key_static, layer_idx)

        # Per-layer features
        if layer_key not in self._layer_cache:
            self._layer_cache[layer_key] = self._simulate_layer(params, layer_idx)

        # Per-(layer, node) features
        for node_idx in range(n_nodes):
            node_key = (key_static, layer_idx, node_idx)
            if node_key not in self._node_cache:
                self._node_cache[node_key] = self._simulate_node(
                    params, layer_idx, node_idx, n_nodes,
                )

    def get_layer_feature(
        self,
        params: dict[str, Any],
        feat_code: str,
        layer_idx: int,
    ) -> float:
        """Return one cached per-layer feature value."""
        key = (self._params_key(params), layer_idx)
        if key not in self._layer_cache:
            self.run_layer(params, layer_idx, n_nodes=int(params.get("n_nodes", 7)))
        return self._layer_cache[key][feat_code]

    def get_node_feature(
        self,
        params: dict[str, Any],
        feat_code: str,
        layer_idx: int,
        node_idx: int,
    ) -> float:
        """Return one cached per-(layer, node) feature value."""
        n_nodes = int(params.get("n_nodes", 7))
        key = (self._params_key(params), layer_idx, node_idx)
        if key not in self._node_cache:
            self.run_layer(params, layer_idx, n_nodes=n_nodes)
        return self._node_cache[key][feat_code]

    # --- Simulation internals ------------------------------------------------

    @staticmethod
    def _simulate_layer(params: dict[str, Any], layer_idx: int) -> dict[str, float]:
        """Compute per-layer features from effective ``params`` at ``layer_idx``."""
        return {
            "extrusion_consistency": physics.feature_extrusion_consistency(
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
                calibration_factor=float(params["calibration_factor"]),
                layer_idx=layer_idx,
            ),
            "current_mean_feeder": physics.feature_current_mean_feeder(
                calibration_factor=float(params["calibration_factor"]),
                layer_height_mm=float(params["layer_height"]),
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
                layer_idx=layer_idx,
            ),
            "current_mean_nozzle": physics.feature_current_mean_nozzle(
                calibration_factor=float(params["calibration_factor"]),
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
                layer_idx=layer_idx,
            ),
            "printing_duration": physics.feature_printing_duration(
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
            ),
            "temperature": physics.feature_temperature(layer_idx),
            "humidity": physics.feature_humidity(layer_idx),
        }

    @staticmethod
    def _simulate_node(
        params: dict[str, Any],
        layer_idx: int,
        node_idx: int,
        n_nodes: int,
    ) -> dict[str, float]:
        """Compute per-(layer, node) features."""
        return {
            "node_overlap": physics.feature_node_overlap(
                path_offset_mm=float(params["path_offset"]),
                layer_height_mm=float(params["layer_height"]),
                calibration_factor=float(params["calibration_factor"]),
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
                layer_idx=layer_idx,
                node_idx=node_idx,
                n_nodes=n_nodes,
            ),
            "filament_width": physics.feature_filament_width(
                path_offset_mm=float(params["path_offset"]),
                layer_height_mm=float(params["layer_height"]),
                calibration_factor=float(params["calibration_factor"]),
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
                layer_idx=layer_idx,
                node_idx=node_idx,
                n_nodes=n_nodes,
            ),
        }
