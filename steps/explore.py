"""Run model-guided exploration rounds (incremental)."""

import argparse
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pred_fab.utils.metrics import combined_score

from steps._common import (
    apply_schedule_args, ensure_plot_dir, load_session, next_code,
    rebuild, run_and_record, save_session, show_plot_with_header,
    get_performance, effective_weights,
)
from workflow import with_dimensions


def run(args: argparse.Namespace) -> None:
    config, state = load_session()
    agent, dataset, fab = rebuild(config)
    apply_schedule_args(agent, args, config)
    plot_dir = ensure_plot_dir()
    perf_weights = effective_weights(config)

    if not state.all_codes:
        raise RuntimeError("No baseline experiments yet — run `cli.py baseline --n N` first.")

    _B = "\033[1m"; _C = "\033[36m"; _R = "\033[0m"; _D = "\033[2m"
    bar = "━" * 58
    print(f"\n{_B}{_C}{bar}{_R}")
    print(f"{_B}{_C}  PHASE 3{_R}{_B} ▸ Exploration{_R}")
    print(f"  {_D}{args.n} rounds (κ={args.kappa}) — model guides search{_R}")
    print(f"{_B}{_C}{bar}{_R}\n")

    dm = agent.create_datamodule(dataset)
    dm.prepare(val_size=0.25)

    log: list[tuple[str, dict, dict]] = []
    for _ in range(args.n):
        agent.train(dm, validate=False)
        current = with_dimensions(state.prev_params) if state.prev_params else None
        spec = agent.exploration_step(dm, kappa=args.kappa, current_params=current)
        code = next_code(state, "explore")

        exp_data, params, _sched = run_and_record(
            dataset, agent, fab, spec, code,
            extra_params=state.prev_params, dataset_code="exploration",
        )
        dm.update()

        perf = get_performance(exp_data)
        state.record("exploration", code, params, perf)
        log.append((code, params, perf))
        score = combined_score(perf, perf_weights or {})
        print(f"  {code:<14s}  κ={args.kappa:.2f}  combined={score:.3f}")

    # Phase summary
    scores = [combined_score(p, perf_weights or {}) for _, _, p in log]
    print(f"\n  {_D}{'─' * 40}{_R}")
    print(f"  {len(log)} rounds  best={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")

    if getattr(args, "plot", False) and log:
        from pred_fab.plotting import plot_convergence

        import os
        conv_history = [combined_score(p, perf_weights or {}) for _, _, p in log]
        path_conv = os.path.join(plot_dir, "03_exploration_convergence.png")
        plot_convergence(path_conv, {"exploration": conv_history})
        show_plot_with_header(path_conv, "Exploration: Convergence", inline=args.plot)

    save_session(config, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-guided exploration rounds")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--schedule", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
