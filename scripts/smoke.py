"""End-to-end smoke: discovery → evaluate → train → exploration → inference.

Validates that the ADVEI mock runs the full PFAB loop against pred-fab@advei-2026,
and prints the per-attribute performance spread so the physics tuning can be
checked for Pareto-richness (each attribute reachable, none trivially maxed).

Run from the repo root:  .venv/bin/python scripts/smoke.py
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pred_fab.core import Dataset

from cli.agent_setup import build_schema, build_fab, build_agent
from models.schema import N_NODES, derive_n_layers, AttributeCode

N_DISCOVERY = 8
ROOT = "./.smoke_data"


def with_dims(params: dict) -> dict:
    """Add the per-experiment tensor dimensions (variable n_layers from layer_height)."""
    return {**params, "n_layers": derive_n_layers(float(params["layer_height"])), "n_nodes": N_NODES}


def params_from_spec(spec) -> dict:
    return dict(spec.initial_params.values)


def main() -> None:
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    schema = build_schema(root_folder=ROOT)
    fab = build_fab()
    agent = build_agent(schema, fab, verbose=False)
    dataset = Dataset(schema=schema)

    print("\n=== 1. Discovery (κ=1) ===")
    specs = agent.discovery_step(n=N_DISCOVERY)
    perfs: dict[str, list[float]] = {a: [] for a in [
        AttributeCode.STRUCTURAL_INTEGRITY, AttributeCode.MATERIAL_DEPOSITION,
        AttributeCode.EXTRUSION_STABILITY, AttributeCode.ENERGY_FOOTPRINT,
        AttributeCode.FABRICATION_TIME,
    ]}
    for i, spec in enumerate(specs):
        params = with_dims(params_from_spec(spec))
        code = f"discovery/{i:03d}"
        exp = dataset.create_experiment(code, parameters=params, dataset_code="discovery")
        fab.run_experiment(params)
        agent.evaluate(exp)
        pdict = {k: float(v) for k, v in exp.performance.get_values_dict().items()}
        for a in perfs:
            if a in pdict:
                perfs[a].append(pdict[a])
        print(f"  {code}  L={params['n_layers']:2d}  "
              + "  ".join(f"{a.split('_')[0][:4]}={pdict.get(a, float('nan')):.2f}" for a in perfs))

    print("\n=== per-attribute spread over discovery (Pareto-richness check) ===")
    for a, vals in perfs.items():
        if vals:
            print(f"  {a:22s} min={min(vals):.2f}  max={max(vals):.2f}  mean={sum(vals)/len(vals):.2f}")

    print("\n=== 2. Train StructuralMLP ===")
    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.25)
    results = agent.train(dm, validate=True)
    print("  train results:", results)

    print("\n=== 3. Exploration (κ=0.6) ===")
    espec = agent.exploration_step(dm, kappa=0.6)
    print("  proposal:", {k: round(float(v), 4) for k, v in espec.initial_params.values.items()})

    print("\n=== 4. Inference (κ=0) ===")
    cur = dataset.get_experiment("discovery/000")
    ispec = agent.inference_step(cur, dm)
    print("  proposal:", {k: round(float(v), 4) for k, v in ispec.initial_params.values.items()})

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
