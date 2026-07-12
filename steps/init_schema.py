"""Show the problem schema."""
import argparse

import sys as _sys; _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from steps._common import JourneyState, build_schema, save_session, print_phase_banner


def run(args: argparse.Namespace) -> None:
    from schema import SCHEMA_TITLE
    config = {}
    state = JourneyState()
    print_phase_banner("0.1", "Schema", SCHEMA_TITLE)
    schema = build_schema()
    schema.state_report()

    # Store schema bounds in config for configure --show
    schema_bounds: dict[str, list[float]] = {}
    for code, obj in schema.parameters.data_objects.items():
        lo = obj.constraints.get("min", None)
        hi = obj.constraints.get("max", None)
        if lo is not None and hi is not None:
            schema_bounds[code] = [lo, hi]
    for domain in schema.domains._domains.values():
        for dim in domain.axes:
            if dim.max_val is not None:
                schema_bounds[dim.code] = [float(dim.min_val), float(dim.max_val)]
    config["schema_bounds"] = schema_bounds

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the problem schema")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
