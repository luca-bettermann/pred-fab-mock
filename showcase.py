"""PrintingShowcase — drives the full PFAB journey on the simulated printer.

Encapsulates the agent, fabrication system, dataset, and per-phase logic so the
entry-point script reads as a plain narrative:

    showcase.baseline(...)   # space-filling experiments
    showcase.train()         # fit the prediction model
    showcase.explore(...)    # information-driven sampling
    showcase.infer(...)      # exploit the model for optimal parameters
    showcase.plots()         # render the figures

Everything below the sensor layer is synthetic, so the whole journey runs with
no hardware and no databases.
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from pred_fab.core import Dataset

from schema import build_schema
from agent_setup import build_agent
from sensors import CameraSystem, EnergySensor, FabricationSystem
from analysis import true_performance_grid
from utils import params_from_spec, get_performance
from visualization import (
    stage_average_field, plot_stage_print, plot_parameter_topology,
    plot_performance_trajectory, plot_feature_heatmaps, plot_prediction_accuracy,
    print_phase_header, print_section, print_experiment_row, print_phase_summary, print_done,
)

# One experiment outcome: (phase, code, params, performance)
Record = Tuple[str, str, Dict[str, Any], Dict[str, float]]


class PrintingShowcase:
    """A self-contained PFAB calibration campaign on the synthetic extrusion printer."""

    def __init__(
        self,
        design_intent: Dict[str, Any],
        bounds: Dict[str, Tuple[float, float]],
        performance_weights: Dict[str, float],
    ) -> None:
        self.intent = design_intent
        self.bounds = bounds
        self.weights = performance_weights

        print_phase_header(0, "Setup")
        self._reset_run_state()
        self.schema = build_schema()
        self.fab = FabricationSystem(CameraSystem(), EnergySensor())
        self.agent = build_agent(self.schema, self.fab.camera, self.fab.energy)
        self.agent.configure_calibration(
            bounds=bounds, performance_weights=performance_weights, fixed_params=design_intent,
        )
        # Spatial domain axes are fixed constants, not optimization params, so the
        # calibration candidates don't carry them — expose them as context.
        n_layers, n_segments = self.fab.get_dimensions(design_intent["design"])
        self.agent.update_context_snapshot({"n_layers": n_layers, "n_segments": n_segments})
        self.dataset = Dataset(schema=self.schema)

        self._records: List[Record] = []
        self._datamodule: Any = None
        self._seed_params: Dict[str, Any] = {}   # carry-forward for explore/infer
        self._last_baseline_exp: Any = None

    # ── Phases ────────────────────────────────────────────────────────────────

    def baseline(self, n: int = 10) -> None:
        """Space-filling (greedy-maximin) experiments — the initial dataset."""
        print_phase_header(1, "Baseline Sampling", f"{n} space-filling experiments")
        for i, spec in enumerate(self.agent.baseline_step(n=n)):
            params = self._full_params(spec)
            exp, _ = self._fabricate("baseline", f"baseline_{i + 1:02d}", params)
            self._seed_params, self._last_baseline_exp = params, exp
        print_phase_summary(self._phase_log("baseline"))

    def train(self) -> None:
        """Fit the prediction model on the data collected so far."""
        print_phase_header(2, "Initial Training")
        self._datamodule = self.agent.create_datamodule(self.dataset)
        self._datamodule.prepare(val_size=0.25)
        self.agent.train(self._datamodule, validate=True)

    def explore(self, rounds: int = 8, w_explore: float = 0.7) -> None:
        """Information-driven sampling — each round proposes, fabricates, and re-trains."""
        print_phase_header(3, "Exploration", f"{rounds} rounds  (w_explore={w_explore})")
        for i in range(rounds):
            spec = self.agent.exploration_step(self._datamodule, w_explore=w_explore, current_params=self._seed_params)
            params = self._full_params(spec, self._seed_params)
            self._fabricate("exploration", f"explore_{i + 1:02d}", params)
            self._retrain()
            self._seed_params = params
        print_phase_summary(self._phase_log("exploration"))

    def infer(self, rounds: int = 3) -> None:
        """Exploit the trained model to propose optimal parameters for the design intent."""
        print_phase_header(4, "Inference", f"{rounds} rounds  ·  exploit (κ=0)  ·  intent {self.intent}")
        params = self._seed_params
        for i in range(rounds):
            code = f"infer_{i + 1:02d}"
            exp = self.dataset.create_experiment(code, parameters=params)
            self.fab.run_experiment(params)
            spec = self.agent.inference_step(exp, self._datamodule, w_explore=0.0, current_params=params)
            self.dataset.save_experiment(code)
            self._retrain()
            self._record("inference", code, params, get_performance(exp))
            params = self._full_params(spec, params)
        print_phase_summary(self._phase_log("inference"))

    def plots(self) -> None:
        """Render every figure for the showcase to ./plots/."""
        print_section("Rendering figures → ./plots/")
        plot_prediction_accuracy(self.agent, self._datamodule)
        plot_feature_heatmaps(self._last_baseline_exp)
        plot_performance_trajectory(self._perf_history(), self._phases())

        water, speed, grid, optimum = true_performance_grid(
            self.intent["design"], self.intent["material"], self.fab, self.weights, self.bounds,
        )
        plot_parameter_topology(self._all_params(), self._phases(), water, speed, grid, optimum)

        # Average print per stage — define once, render per stage on a shared scale.
        fields = {ph: stage_average_field(self.fab.camera, self._stage_params(ph))
                  for ph in ("baseline", "exploration", "inference")}
        vmax = max(f.max_dev for f in fields.values())
        for phase, field in fields.items():
            plot_stage_print(field, phase, vmax=vmax, name=f"stage_{phase}")
        print_done()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _reset_run_state(self) -> None:
        for d in ("./pfab_data", "./local"):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs("./plots", exist_ok=True)

    def _with_dimensions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        n_layers, n_segments = self.fab.get_dimensions(params["design"])
        return {**params, "n_layers": n_layers, "n_segments": n_segments}

    def _full_params(self, spec: Any, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge fixed design intent + proposed params + derived dimensions."""
        return self._with_dimensions({**(base or {}), **self.intent, **params_from_spec(spec)})

    def _fabricate(self, phase: str, code: str, params: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        exp = self.dataset.create_experiment(code, parameters=params)
        self.fab.run_experiment(params)
        self.agent.evaluate(exp)
        self.dataset.save_experiment(code)
        perf = self._record(phase, code, params, get_performance(exp))
        return exp, perf

    def _retrain(self) -> None:
        self._datamodule.update()
        self.agent.train(self._datamodule, validate=False)

    def _record(self, phase: str, code: str, params: Dict[str, Any], perf: Dict[str, float]) -> Dict[str, float]:
        self._records.append((phase, code, params, perf))
        print_experiment_row(code, params, perf)
        return perf

    # Views over the collected records
    def _phase_log(self, phase: str) -> List[Tuple[str, Dict[str, Any], Dict[str, float]]]:
        return [(c, p, perf) for ph, c, p, perf in self._records if ph == phase]

    def _phases(self) -> List[str]:
        return [ph for ph, _, _, _ in self._records]

    def _all_params(self) -> List[Dict[str, Any]]:
        return [p for _, _, p, _ in self._records]

    def _perf_history(self) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        return [(p, perf) for _, _, p, perf in self._records]

    def _stage_params(self, phase: str) -> List[Dict[str, Any]]:
        return [p for ph, _, p, _ in self._records if ph == phase]
