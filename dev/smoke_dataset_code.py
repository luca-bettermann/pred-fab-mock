"""Smoke test — verifies the ``dataset_code`` flow end-to-end against the mock.

Exercises:
1. Mock workflow tags experiments by phase (``baseline`` / ``test``).
2. ``DataModule.set_split_dataset`` filters by ``dataset_code``.
3. ``dataset_code`` round-trips through ``save_experiment`` → fresh ``Dataset`` → ``load_experiment``.

Run via::

    uv run dev/smoke_dataset_code.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pred_fab.core import DataModule
from pred_fab.core.dataset import Dataset
from pred_fab.utils import SplitType
from schema import build_schema
from shared import make_env, with_dims, run_experiment


def main() -> None:
    agent, fab, dataset = make_env("smoke_dataset_code", verbose=False)

    # Two experiments tagged with distinct dataset_codes.
    params = with_dims({"water_ratio": 0.40, "print_speed": 40.0})
    run_experiment(dataset, agent, fab, params, "baseline_01", dataset_code="baseline")
    run_experiment(dataset, agent, fab, params, "test_01",     dataset_code="test")

    # Inline-tag check.
    assert dataset.get_experiment("baseline_01").dataset_code == "baseline"
    assert dataset.get_experiment("test_01").dataset_code == "test"

    # set_split_dataset filters correctly.
    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["water_ratio", "print_speed", "n_layers", "n_segments"],
        input_features=[],
        output_columns=["path_deviation"],
    )
    dm.set_split_dataset("baseline")
    dm.set_split_dataset("test", split=SplitType.TEST)
    assert dm.get_split_codes(SplitType.TRAIN) == ["baseline_01"]
    assert dm.get_split_codes(SplitType.TEST) == ["test_01"]

    # Round-trip through disk: fresh Dataset reads dataset_code back.
    schema_root = dataset.local_data.root_folder
    schema_b = build_schema(root_folder=schema_root)
    dataset_b = Dataset(schema=schema_b)
    dataset_b.load_experiment("baseline_01", verbose=False)
    dataset_b.load_experiment("test_01", verbose=False)
    assert dataset_b.get_experiment("baseline_01").dataset_code == "baseline"
    assert dataset_b.get_experiment("test_01").dataset_code == "test"

    print("✓ dataset_code flow OK — tagging, set_split_dataset, round-trip all green.")


if __name__ == "__main__":
    main()
