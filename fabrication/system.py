"""FabricationSystem — feature-level mock fabrication for ADVEI 2026.

Simulates one print run by computing every feature value directly from the
synthetic physics in :mod:`fabrication.physics` — no raw sensor data. Each
``run_experiment`` populates a cache keyed by ``(params_hash, layer_idx,
node_idx)`` for depth-2 features and ``(params_hash, layer_idx)`` for depth-1
features (including the per-layer node aggregates). The mock feature models in
``models/features`` then route cached values to their schema feature codes.

The real-fabrication counterpart in ``learning-by-printing`` does the
raw-sensor-to-feature extraction; this harness stands in for that whole layer.
"""
from __future__ import annotations

import json
from typing import Any

from models.schema import FeatureCode, N_NODES, derive_n_layers
from fabrication import physics


class FabricationSystem:
    """Coordinates feature-level simulation for a single print run."""

    def __init__(self, random_seed: int | None = None) -> None:
        # Reserved for future noise injection; the current physics is deterministic.
        self._random_seed = random_seed
        self._node_cache: dict[tuple, dict[str, float]] = {}
        self._layer_cache: dict[tuple, dict[str, float]] = {}

    @staticmethod
    def _params_key(params: dict[str, Any]) -> str:
        return json.dumps(
            {k: float(v) for k, v in params.items() if isinstance(v, (int, float))},
            sort_keys=True,
        )

    @staticmethod
    def get_dimensions(params: dict[str, Any]) -> tuple[int, int]:
        """``(n_layers, n_nodes)`` — n_layers is derived from layer_height per experiment."""
        if "n_layers" in params:
            n_layers = int(params["n_layers"])
        else:
            n_layers = derive_n_layers(float(params["layer_height"]))
        n_nodes = int(params.get("n_nodes", N_NODES))
        return n_layers, n_nodes

    # --- Public API ----------------------------------------------------------

    def run_experiment(self, params: dict[str, Any]) -> None:
        """Populate the cache for every (layer, node) cell of one experiment."""
        n_layers, n_nodes = self.get_dimensions(params)
        for layer_idx in range(n_layers):
            self.run_layer(params, layer_idx, n_nodes=n_nodes)

    def run_layer(self, params: dict[str, Any], layer_idx: int, n_nodes: int = N_NODES) -> None:
        """Simulate + cache all per-node and per-layer features for one layer.

        ``params`` is the *effective* parameter dict at this layer (per-layer
        trajectory overrides already applied by the caller).
        """
        key = self._params_key(params)
        node_vals: list[dict[str, float]] = []
        for node_idx in range(n_nodes):
            nk = (key, layer_idx, node_idx)
            if nk not in self._node_cache:
                self._node_cache[nk] = self._simulate_node(params, layer_idx, node_idx, n_nodes)
            node_vals.append(self._node_cache[nk])

        lk = (key, layer_idx)
        if lk not in self._layer_cache:
            self._layer_cache[lk] = self._simulate_layer(params, layer_idx, node_vals)

    def get_layer_feature(self, params: dict[str, Any], feat_code: str, layer_idx: int) -> float:
        """Return one cached per-layer feature value."""
        key = (self._params_key(params), layer_idx)
        if key not in self._layer_cache:
            self.run_layer(params, layer_idx, n_nodes=int(params.get("n_nodes", N_NODES)))
        return self._layer_cache[key][feat_code]

    def get_node_feature(self, params: dict[str, Any], feat_code: str,
                         layer_idx: int, node_idx: int) -> float:
        """Return one cached per-(layer, node) feature value."""
        key = (self._params_key(params), layer_idx, node_idx)
        if key not in self._node_cache:
            self.run_layer(params, layer_idx, n_nodes=int(params.get("n_nodes", N_NODES)))
        return self._node_cache[key][feat_code]

    # --- Simulation internals ------------------------------------------------

    @staticmethod
    def _simulate_node(params: dict[str, Any], layer_idx: int, node_idx: int,
                       n_nodes: int) -> dict[str, float]:
        return {
            FeatureCode.NODE_OVERLAP: physics.feature_node_overlap(
                path_offset_mm=float(params["path_offset"]),
                calibration_factor=float(params["calibration_factor"]),
                layer_idx=layer_idx, node_idx=node_idx, n_nodes=n_nodes,
            ),
            FeatureCode.FILAMENT_WIDTH: physics.feature_filament_width(
                calibration_factor=float(params["calibration_factor"]),
                print_speed_mps=float(params["print_speed"]),
                layer_height_mm=float(params["layer_height"]),
                layer_idx=layer_idx, node_idx=node_idx, n_nodes=n_nodes,
            ),
        }

    @staticmethod
    def _simulate_layer(params: dict[str, Any], layer_idx: int,
                        node_vals: list[dict[str, float]]) -> dict[str, float]:
        n = len(node_vals)
        overlap_mean = sum(v[FeatureCode.NODE_OVERLAP] for v in node_vals) / n
        width_mean = sum(v[FeatureCode.FILAMENT_WIDTH] for v in node_vals) / n
        return {
            FeatureCode.LOADCELL_RESIDUAL: physics.feature_loadcell_residual(
                print_speed_mps=float(params["print_speed"]),
                calibration_factor=float(params["calibration_factor"]),
                layer_idx=layer_idx,
            ),
            FeatureCode.ROBOT_ENERGY: physics.feature_robot_energy(
                print_speed_mps=float(params["print_speed"]),
                layer_height_mm=float(params["layer_height"]),
                layer_idx=layer_idx,
            ),
            FeatureCode.PRINTING_DURATION: physics.feature_printing_duration(
                print_speed_mps=float(params["print_speed"]),
            ),
            FeatureCode.TEMPERATURE: physics.feature_temperature(layer_idx),
            FeatureCode.HUMIDITY: physics.feature_humidity(layer_idx),
            FeatureCode.MATERIAL_AGE: physics.feature_material_age(layer_idx),
            FeatureCode.NODE_OVERLAP_MEAN: overlap_mean,
            FeatureCode.FILAMENT_WIDTH_MEAN: width_mean,
        }
