"""Per-step plotting helpers for the ADVEI 2026 mock CLI."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from pred_fab.core import DataModule
from pred_fab import PfabAgent
from pred_fab.utils import SplitType  # type: ignore[attr-defined]
from pred_fab.utils.metrics import combined_score as _combined_score
from sklearn.metrics import r2_score

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "structural_integrity": 1.0, "material_deposition": 1.0,
    "extrusion_stability": 1.0, "energy_footprint": 1.0, "fabrication_time": 1.0,
}


def combined_score(perf: Dict[str, Any], weights: Optional[Dict[str, float]]) -> float:
    return _combined_score(perf, weights or _DEFAULT_WEIGHTS)

from sensors.physics import (
    feature_node_overlap,
    feature_filament_width,
    feature_extrusion_consistency,
    feature_current_mean_feeder,
    feature_printing_duration,
    COMPONENT_HEIGHT_MM,
    TARGET_NODE_OVERLAP_MM,
    TARGET_FILAMENT_WIDTH_MM,
    PATH_LENGTH_PER_LAYER_M,
    n_layers_for_height,
)
from schema import PARAM_BOUNDS

os.makedirs("./plots", exist_ok=True)

_PHASE_COLORS = {
    "baseline": "#4C72B0",
    "exploration": "#DD8452",
    "inference": "#55A868",
    "grid": "#937860",
    "test": "#8172B2",
}

# Default mid-values for params not being swept in 2D plots.
_MID_PARAMS = {
    "path_offset": 1.5,
    "layer_height": 2.5,
    "calibration_factor": 1.9,
    "print_speed": 0.006,
    "slowdown_factor": 0.5,
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _save(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


# ── Feature heatmap (node_overlap across layer × node) ───────────────────────

def plot_feature_heatmap(
    params: Dict[str, Any],
    exp_code: str = "",
    save_dir: str = "./plots",
) -> None:
    """Heatmap of node_overlap values across (layer, node) for one experiment.

    The ADVEI equivalent of the main mock's 3D path comparison — shows how
    overlap varies spatially, with corners (high curvature) highlighted.
    """
    lh = float(params.get("layer_height", 2.5))
    n_layers = n_layers_for_height(lh)
    n_nodes = int(params.get("n_nodes", 7))

    grid = np.zeros((n_layers, n_nodes))
    for li in range(n_layers):
        for ni in range(n_nodes):
            grid[li, ni] = feature_node_overlap(
                path_offset_mm=float(params["path_offset"]),
                layer_height_mm=lh,
                calibration_factor=float(params["calibration_factor"]),
                print_speed_mps=float(params["print_speed"]),
                slowdown_factor=float(params["slowdown_factor"]),
                layer_idx=li,
                node_idx=ni,
                n_nodes=n_nodes,
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    norm = Normalize(vmin=0.0, vmax=max(float(grid.max()), TARGET_NODE_OVERLAP_MM * 1.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn",
                   norm=norm, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Node Overlap [mm]", fontsize=9)

    # Target line on colorbar
    cb.ax.axhline(TARGET_NODE_OVERLAP_MM, color="white", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Node Index", fontsize=10)
    ax.set_ylabel("Layer Index", fontsize=10)
    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 5)))

    title = f"Node Overlap — {exp_code}" if exp_code else "Node Overlap"
    meta = (f"spd={params.get('print_speed', 0):.4f}  "
            f"sf={params.get('slowdown_factor', 0):.2f}  "
            f"lh={lh:.1f}  cf={params.get('calibration_factor', 0):.2f}")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    ax.set_title(meta, fontsize=9, pad=8)

    _save(os.path.join(save_dir, f"feature_heatmap_{exp_code or 'latest'}.png"))


# ── Multi-feature layer profile ──────────────────────────────────────────────

def plot_layer_profiles(
    params: Dict[str, Any],
    exp_code: str = "",
    save_dir: str = "./plots",
) -> None:
    """Per-layer feature profiles (extrusion_consistency, currents, duration).

    Shows how depth-1 features evolve across layers for one experiment —
    the ADVEI equivalent of tracking deviation per layer.
    """
    lh = float(params.get("layer_height", 2.5))
    n_layers = n_layers_for_height(lh)
    layers = list(range(n_layers))

    consistency = [feature_extrusion_consistency(
        print_speed_mps=float(params["print_speed"]),
        slowdown_factor=float(params["slowdown_factor"]),
        calibration_factor=float(params["calibration_factor"]),
        layer_idx=li,
    ) for li in layers]

    feeder_current = [feature_current_mean_feeder(
        calibration_factor=float(params["calibration_factor"]),
        layer_height_mm=lh,
        print_speed_mps=float(params["print_speed"]),
        slowdown_factor=float(params["slowdown_factor"]),
        layer_idx=li,
    ) for li in layers]

    duration = [feature_printing_duration(
        print_speed_mps=float(params["print_speed"]),
        slowdown_factor=float(params["slowdown_factor"]),
    ) for _ in layers]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    fig.suptitle(f"Layer Profiles — {exp_code}" if exp_code else "Layer Profiles",
                 fontsize=12, fontweight="bold")

    axes[0].plot(layers, consistency, marker="o", color="#4C72B0", linewidth=1.8, markersize=4)
    axes[0].set_ylabel("Extrusion\nConsistency", fontsize=9)
    axes[0].set_ylim(0.2, 1.05)
    axes[0].axhline(1.0, color="#55A868", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(layers, feeder_current, marker="s", color="#DD8452", linewidth=1.8, markersize=4)
    axes[1].set_ylabel("Feeder\nCurrent [A]", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    axes[2].bar(layers, duration, color="#937860", alpha=0.75, width=0.7)
    axes[2].set_ylabel("Duration [s]", fontsize=9)
    axes[2].set_xlabel("Layer Index", fontsize=10)
    axes[2].grid(True, alpha=0.25, axis="y")

    _save(os.path.join(save_dir, f"layer_profiles_{exp_code or 'latest'}.png"))


# ── Prediction accuracy (generic) ───────────────────────────────────────────

def plot_prediction_accuracy(
    agent: PfabAgent,
    datamodule: DataModule,
    save_dir: str = "./plots",
) -> Dict[str, float]:
    """Scatter of predicted vs actual for all model outputs with R² annotation."""
    pred_system = agent.predict_system  # type: ignore[attr-defined]
    outputs = pred_system.get_system_outputs()

    val_batches = datamodule.get_batches(SplitType.VAL)
    if not val_batches:
        return {}

    all_val_batches = list(val_batches)
    y_val = np.vstack([b[1] for b in all_val_batches])

    y_pred_cols: List[np.ndarray] = []
    for model in pred_system.models:
        model_batches = pred_system._filter_batches_for_model(  # type: ignore[attr-defined]
            all_val_batches, model
        )
        X_m = np.vstack([b[0] for b in model_batches])
        y_pred_cols.append(model.forward_pass(X_m))
    y_pred = np.hstack(y_pred_cols) if y_pred_cols else y_val

    n_out = min(y_val.shape[1], len(outputs))
    fig, axes_arr = plt.subplots(1, n_out, figsize=(5 * n_out, 4.5))
    axes: list = [axes_arr] if n_out == 1 else list(axes_arr)  # type: ignore[arg-type]

    fig.suptitle("Prediction Accuracy — validation set", fontsize=12, fontweight="bold")

    r2_scores: Dict[str, float] = {}
    for i, (ax, name) in enumerate(zip(axes, outputs)):
        if i >= y_pred.shape[1] or i >= y_val.shape[1]:
            break
        r2 = r2_score(y_val[:, i], y_pred[:, i])
        r2_scores[name] = float(r2)
        lim = [
            min(float(y_val[:, i].min()), float(y_pred[:, i].min())),
            max(float(y_val[:, i].max()), float(y_pred[:, i].max())),
        ]
        ax.scatter(y_val[:, i], y_pred[:, i], alpha=0.7,
                   color=_PHASE_COLORS["exploration"], edgecolors="white", linewidths=0.4)
        ax.plot(lim, lim, "k--", linewidth=1.2, label="Perfect prediction")
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{name}\nR² = {r2:.3f}", fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.legend(fontsize=7)

    _save(os.path.join(save_dir, "prediction_accuracy.png"))
    return r2_scores


# ── Parameter space scatter ──────────────────────────────────────────────────

def plot_parameter_space(
    all_params: List[Dict[str, Any]],
    phases: List[str],
    perf_history: Optional[List[Tuple[Dict[str, Any], Dict[str, float]]]] = None,
    perf_weights: Optional[Dict[str, float]] = None,
    save_dir: str = "./plots",
) -> None:
    """2D scatter of print_speed vs slowdown_factor (the two runtime params).

    Points sized by experiment order, coloured by combined performance score.
    Phase encoded in marker shape.
    """
    speeds = [float(p.get("print_speed", 0.0)) for p in all_params]
    slowdowns = [float(p.get("slowdown_factor", 0.0)) for p in all_params]
    n = len(all_params)

    scores: Optional[List[float]] = None
    if perf_history is not None and len(perf_history) == n:
        scores = [
            combined_score(ph[1], perf_weights)
            for ph in perf_history
        ]

    _MARKERS = {"baseline": "o", "exploration": "*", "inference": "D",
                "grid": "s", "test": "^"}

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Parameter Space — print_speed vs slowdown_factor",
                 fontsize=12, fontweight="bold")

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    vmin = min(scores) if scores else 0.0
    vmax = max(scores) if scores else 1.0
    _last_sc = None

    for phase, marker in _MARKERS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if not idx:
            continue
        sizes = [40 + 4 * i for i in idx]
        if scores is not None:
            sc = ax.scatter(
                [speeds[i] for i in idx],
                [slowdowns[i] for i in idx],
                c=[scores[i] for i in idx],
                cmap=cmap, vmin=vmin, vmax=vmax,
                marker=marker, s=sizes,
                edgecolors="white", linewidths=0.8,
                alpha=0.9, zorder=3, label=phase.capitalize(),
            )
            _last_sc = sc
        else:
            ax.scatter(
                [speeds[i] for i in idx],
                [slowdowns[i] for i in idx],
                c=_PHASE_COLORS.get(phase, "#666"),
                marker=marker, s=sizes,
                edgecolors="white", linewidths=0.8,
                alpha=0.85, zorder=3, label=phase.capitalize(),
            )

    if scores is not None and _last_sc is not None:
        cb = fig.colorbar(_last_sc, ax=ax, pad=0.02)
        cb.set_label("Combined Score [0–1]", fontsize=9)

    ax.legend(fontsize=9, title="Phase", title_fontsize=8, loc="upper right")
    ax.set_xlabel("Print Speed [m/s]", fontsize=10)
    ax.set_ylabel("Slowdown Factor", fontsize=10)

    # Bounds from schema
    spd_bounds = next((lo, hi) for c, lo, hi in PARAM_BOUNDS if c == "print_speed")
    sf_bounds = next((lo, hi) for c, lo, hi in PARAM_BOUNDS if c == "slowdown_factor")
    ax.set_xlim(spd_bounds[0] * 0.95, spd_bounds[1] * 1.05)
    ax.set_ylim(sf_bounds[0] - 0.05, sf_bounds[1] + 0.05)
    ax.grid(True, alpha=0.25)
    _save(os.path.join(save_dir, "parameter_space.png"))


# ── Performance trajectory ───────────────────────────────────────────────────

def plot_performance_trajectory(
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]],
    phases: List[str],
    exp_codes: Optional[List[str]] = None,
    perf_weights: Optional[Dict[str, float]] = None,
    save_dir: str = "./plots",
) -> None:
    """Line plot of all 5 performance attributes over experiment history."""
    n = len(perf_history)
    xs = list(range(1, n + 1))

    perf_keys = ["structural_integrity", "material_deposition", "extrusion_stability",
                 "energy_footprint", "fabrication_time"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#937860", "#8172B2"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Performance Trajectory", fontsize=12, fontweight="bold")

    # Shade phase regions
    phase_alpha = {"baseline": 0.06, "grid": 0.04, "exploration": 0.08,
                   "test": 0.04, "inference": 0.08}
    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            ax.axvspan(idx[0] + 0.5, idx[-1] + 1.5,
                       color=color, alpha=phase_alpha.get(phase, 0.06))

    for key, color, marker in zip(perf_keys, colors, markers):
        vals = [pp[1].get(key, float("nan")) for pp in perf_history]
        short = {"structural_integrity": "SI", "material_deposition": "MD",
                 "extrusion_stability": "ES", "energy_footprint": "EF",
                 "fabrication_time": "FT"}[key]
        ax.plot(xs, vals, marker=marker, color=color,
                linewidth=1.4, markersize=3.5, label=short, alpha=0.85)

    # Combined score line
    combined_vals = [
        combined_score(pp[1], perf_weights) for pp in perf_history
    ]
    ax.plot(xs, combined_vals, color="black", linewidth=2.0,
            linestyle="--", label="Combined", alpha=0.7)

    # Phase labels at top
    for phase, color in _PHASE_COLORS.items():
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if idx:
            mid = (idx[0] + idx[-1]) / 2 + 1
            ax.text(mid, 1.03, phase.capitalize(), ha="center", va="bottom",
                    color=color, fontsize=7, transform=ax.get_xaxis_transform())

    if exp_codes and len(exp_codes) == len(xs):
        ax.set_xticks(xs[::max(1, n // 15)])
        ax.set_xticklabels([exp_codes[i] for i in range(0, n, max(1, n // 15))],
                           rotation=35, ha="right", fontsize=6)
    else:
        ax.set_xlabel("Experiment #", fontsize=10)

    ax.set_ylabel("Score [0–1]", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0.5, n + 0.5)
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(True, alpha=0.25)
    _save(os.path.join(save_dir, "performance_trajectory.png"))


# ── Physics topology ─────────────────────────────────────────────────────────

def plot_physics_topology(
    perf_weights: Optional[Dict[str, float]] = None,
    save_dir: str = "./plots",
) -> None:
    """2×3 grid: physics performance across (print_speed, slowdown_factor).

    Shows all 5 performance attributes + combined score as contour plots.
    Other params fixed at mid-values.
    """
    from models.evaluation_models import (
        StructuralIntegrityEval,
        MaterialDepositionEval,
        ExtrusionStabilityEval,
        EnergyFootprintEval,
        FabricationTimeEval,
    )

    N = 30
    spd_lo, spd_hi = 0.004, 0.008
    sf_lo, sf_hi = 0.0, 1.0
    spd_vals = np.linspace(spd_lo, spd_hi, N)
    sf_vals = np.linspace(sf_lo, sf_hi, N)
    SPD, SF = np.meshgrid(spd_vals, sf_vals)

    po = _MID_PARAMS["path_offset"]
    lh = _MID_PARAMS["layer_height"]
    cf = _MID_PARAMS["calibration_factor"]
    n_layers = n_layers_for_height(lh)
    n_nodes = 7

    # Compute per-cell scores
    si_grid = np.zeros((N, N))
    md_grid = np.zeros((N, N))
    es_grid = np.zeros((N, N))
    ef_grid = np.zeros((N, N))
    ft_grid = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            spd = float(SPD[i, j])
            sf = float(SF[i, j])

            # Structural integrity: node_overlap vs target
            overlaps = [
                feature_node_overlap(
                    path_offset_mm=po, layer_height_mm=lh, calibration_factor=cf,
                    print_speed_mps=spd, slowdown_factor=sf,
                    layer_idx=li, node_idx=ni, n_nodes=n_nodes,
                )
                for li in range(n_layers) for ni in range(n_nodes)
            ]
            avg_overlap = float(np.mean(overlaps))
            si_grid[i, j] = max(0.0, 1.0 - abs(avg_overlap - TARGET_NODE_OVERLAP_MM) / 1.0)

            # Material deposition: filament_width vs target
            widths = [
                feature_filament_width(
                    path_offset_mm=po, layer_height_mm=lh, calibration_factor=cf,
                    print_speed_mps=spd, slowdown_factor=sf,
                    layer_idx=li, node_idx=ni, n_nodes=n_nodes,
                )
                for li in range(n_layers) for ni in range(n_nodes)
            ]
            avg_width = float(np.mean(widths))
            md_grid[i, j] = max(0.0, 1.0 - abs(avg_width - TARGET_FILAMENT_WIDTH_MM) / 3.0)

            # Extrusion stability
            consistencies = [
                feature_extrusion_consistency(
                    print_speed_mps=spd, slowdown_factor=sf,
                    calibration_factor=cf, layer_idx=li,
                )
                for li in range(n_layers)
            ]
            avg_consist = float(np.mean(consistencies))
            es_grid[i, j] = max(0.0, 1.0 - abs(avg_consist - 1.0) / 0.5)

            # Energy footprint (feeder current vs low-current target)
            currents = [
                feature_current_mean_feeder(
                    calibration_factor=cf, layer_height_mm=lh,
                    print_speed_mps=spd, slowdown_factor=sf, layer_idx=li,
                )
                for li in range(n_layers)
            ]
            avg_current = float(np.mean(currents))
            ef_grid[i, j] = max(0.0, 1.0 - abs(avg_current - 0.6) / 1.5)

            # Fabrication time (duration vs fast target)
            dur = feature_printing_duration(print_speed_mps=spd, slowdown_factor=sf)
            ft_grid[i, j] = max(0.0, 1.0 - abs(dur - 80.0) / 120.0)

    # Combined
    weights = perf_weights or {
        "structural_integrity": 1, "material_deposition": 1,
        "extrusion_stability": 1, "energy_footprint": 1, "fabrication_time": 1,
    }
    w_si = weights.get("structural_integrity", 1)
    w_md = weights.get("material_deposition", 1)
    w_es = weights.get("extrusion_stability", 1)
    w_ef = weights.get("energy_footprint", 1)
    w_ft = weights.get("fabrication_time", 1)
    total_w = w_si + w_md + w_es + w_ef + w_ft
    combined_grid = (w_si * si_grid + w_md * md_grid + w_es * es_grid +
                     w_ef * ef_grid + w_ft * ft_grid) / total_w

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Physics Topology — print_speed × slowdown_factor\n"
        f"(fixed: po={po}, lh={lh}, cf={cf})",
        fontsize=12, fontweight="bold",
    )

    panels = [
        (axes[0, 0], si_grid, "Structural Integrity", "plasma"),
        (axes[0, 1], md_grid, "Material Deposition", "plasma"),
        (axes[0, 2], es_grid, "Extrusion Stability", "plasma"),
        (axes[1, 0], ef_grid, "Energy Footprint", "plasma"),
        (axes[1, 1], ft_grid, "Fabrication Time", "plasma"),
        (axes[1, 2], combined_grid, "Combined (weighted)", "RdYlGn"),
    ]

    for ax, grid, title, cmap_name in panels:
        cf_plot = ax.contourf(SPD * 1000, SF, grid, levels=25, cmap=cmap_name)
        ax.contour(SPD * 1000, SF, grid, levels=6, colors="white", alpha=0.3, linewidths=0.5)
        fig.colorbar(cf_plot, ax=ax, pad=0.02, fraction=0.046)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Print Speed [mm/s]", fontsize=8)
        ax.set_ylabel("Slowdown Factor", fontsize=8)
        ax.grid(True, alpha=0.12)

    _save(os.path.join(save_dir, "physics_topology.png"))


# ── Baseline scatter ─────────────────────────────────────────────────────────

def plot_baseline_scatter(
    experiments: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
    save_dir: str = "./plots",
) -> None:
    """Multi-panel scatter of baseline experiments across param pairs.

    Shows the space-filling quality of the Sobol sampling across the 5
    ADVEI parameters. Three panels: (speed, slowdown), (speed, layer_height),
    (calibration_factor, path_offset).
    """
    pairs = [
        ("print_speed", "slowdown_factor"),
        ("print_speed", "layer_height"),
        ("calibration_factor", "path_offset"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Baseline Experiments — Parameter Coverage",
                 fontsize=12, fontweight="bold")

    for ax, (xkey, ykey) in zip(axes, pairs):
        xs = [float(p.get(xkey, 0.0)) for _, p, _ in experiments]
        ys = [float(p.get(ykey, 0.0)) for _, p, _ in experiments]
        codes = [c for c, _, _ in experiments]

        ax.scatter(xs, ys, s=80, c=_PHASE_COLORS["baseline"], marker="o",
                   alpha=0.80, edgecolors="white", linewidths=1.2, zorder=3)

        for x, y, code in zip(xs, ys, codes):
            ax.annotate(code.split("_")[-1], (x, y), xytext=(3, 3),
                        textcoords="offset points", fontsize=6, alpha=0.65)

        # Axis bounds from PARAM_BOUNDS
        x_bounds = next(((lo, hi) for c, lo, hi in PARAM_BOUNDS if c == xkey), None)
        y_bounds = next(((lo, hi) for c, lo, hi in PARAM_BOUNDS if c == ykey), None)
        if x_bounds:
            margin = (x_bounds[1] - x_bounds[0]) * 0.05
            ax.set_xlim(x_bounds[0] - margin, x_bounds[1] + margin)
        if y_bounds:
            margin = (y_bounds[1] - y_bounds[0]) * 0.05
            ax.set_ylim(y_bounds[0] - margin, y_bounds[1] + margin)

        ax.set_xlabel(xkey, fontsize=9)
        ax.set_ylabel(ykey, fontsize=9)
        ax.grid(True, alpha=0.25)

    _save(os.path.join(save_dir, "baseline_scatter.png"))


# ── Acquisition topology (exploration & inference) ───────────────────────────

def plot_acquisition_topology(
    agent: PfabAgent,
    kappa: float,
    proposed: Dict[str, Any],
    history: List[Dict[str, Any]],
    label: str,
    save_dir: str = "./plots",
) -> None:
    """3-panel acquisition landscape: performance | uncertainty | combined.

    Evaluates the model on a 2D grid over (print_speed, slowdown_factor)
    with other params fixed at the proposed values (or mid-values).
    """
    dm = getattr(agent, "_active_datamodule", None)
    if dm is None:
        cal = getattr(agent, "calibration_system", None)
        if cal is not None:
            dm = getattr(cal, "_active_datamodule", None)
    if dm is None:
        return

    # Fixed params for the sweep
    po = float(proposed.get("path_offset", _MID_PARAMS["path_offset"]))
    lh = float(proposed.get("layer_height", _MID_PARAMS["layer_height"]))
    cf_val = float(proposed.get("calibration_factor", _MID_PARAMS["calibration_factor"]))
    n_layers = n_layers_for_height(lh)

    N_SPD, N_SF = 18, 18
    spd_grid = np.linspace(0.004, 0.008, N_SPD)
    sf_grid = np.linspace(0.0, 1.0, N_SF)
    SPD_mesh, SF_mesh = np.meshgrid(spd_grid, sf_grid)

    perf_grid = np.zeros((N_SF, N_SPD))
    unc_grid = np.zeros((N_SF, N_SPD))

    perf_weights = getattr(agent.calibration_system, "performance_weights", None)
    perf_names = getattr(agent.calibration_system, "perf_names_order", [])
    total_weight = sum((perf_weights or {}).values()) or 1.0

    for si in range(N_SF):
        for wi in range(N_SPD):
            p = {
                "path_offset": po, "layer_height": lh,
                "calibration_factor": cf_val,
                "print_speed": float(SPD_mesh[si, wi]),
                "slowdown_factor": float(SF_mesh[si, wi]),
                "n_layers": n_layers, "n_nodes": 7,
            }
            try:
                perf_dict = agent.calibration_system.perf_fn(p)  # type: ignore[attr-defined]
                weighted_sum = sum(
                    (perf_weights or {}).get(name, 0.0) * float(perf_dict.get(name, 0.0) or 0.0)
                    for name in perf_names
                )
                perf_grid[si, wi] = weighted_sum / total_weight
            except Exception:
                perf_grid[si, wi] = 0.0
            try:
                X_norm = dm.params_to_array(p)
                unc_grid[si, wi] = agent.pred_system.uncertainty(X_norm)  # type: ignore[attr-defined]
            except Exception:
                unc_grid[si, wi] = 1.0

    combined_grid = (1.0 - kappa) * perf_grid + kappa * unc_grid

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Acquisition Topology — {label}  ·  κ={kappa:.2f}",
        fontsize=11, fontweight="bold",
    )

    panels = [
        (axes[0], perf_grid, "Performance (predicted)", "YlGn", False),
        (axes[1], unc_grid, "Uncertainty (evidence gap)", "PuBu", False),
        (axes[2], combined_grid, "Combined acquisition", "RdYlGn", True),
    ]

    hist_spd = [float(p.get("print_speed", 0.0)) for p in history]
    hist_sf = [float(p.get("slowdown_factor", 0.0)) for p in history]
    prop_spd = float(proposed.get("print_speed", 0.006))
    prop_sf = float(proposed.get("slowdown_factor", 0.5))

    for panel_idx, (ax, grid, title, cmap_name, mark_proposed) in enumerate(panels):
        cf_plot = ax.contourf(SPD_mesh * 1000, SF_mesh, grid, levels=24,
                              cmap=cmap_name, alpha=0.92)
        ax.contour(SPD_mesh * 1000, SF_mesh, grid, levels=6,
                   colors="white", alpha=0.20, linewidths=0.5)
        fig.colorbar(cf_plot, ax=ax, pad=0.02, fraction=0.045)

        is_last = panel_idx == 2
        if hist_spd:
            ax.scatter(
                [s * 1000 for s in hist_spd], hist_sf,
                s=22, c="black", edgecolors="white", linewidths=0.6,
                zorder=5, label="Evaluated" if is_last else None,
            )

        if mark_proposed:
            ax.scatter([prop_spd * 1000], [prop_sf], s=180, marker="x",
                       color="#FFD700", linewidths=3.0, zorder=10,
                       label=f"Proposed (spd={prop_spd*1000:.1f}, sf={prop_sf:.2f})")
            ax.legend(fontsize=7, loc="lower right")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Print Speed [mm/s]", fontsize=9)
        ax.set_ylabel("Slowdown Factor", fontsize=9)
        ax.grid(True, alpha=0.12)

    _save(os.path.join(save_dir, f"{label}_topology.png"))
