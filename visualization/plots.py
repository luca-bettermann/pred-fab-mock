"""Plotting helpers for the extrusion-printing showcase.

Visual identity (palette, spines, DPI, colormaps) lives in `_style`. Each function
renders one figure; the showcase orchestrates them.
"""

from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pred_fab.core import ExperimentData, DataModule
from pred_fab import PfabAgent

from sensors.physics import FILAMENT_RADIUS
from . import _style
from ._style import (
    PHASE_COLORS, PHASE_LABELS, DEVIATION_CMAP, PERFORMANCE_CMAP, FONT,
    ZINC_300, ZINC_400, ZINC_500, ZINC_700, STEEL_300, STEEL_500, EMERALD_500,
    clean_spines, clean_3d_panes, style_colorbar, light_grid, save_fig,
)

_style.apply_style()

# Geometry shared by the 3-D path views
_N_LAYERS, _N_SEGMENTS, _N_PTS = 5, 4, 5
_SEG_LEN, _SEG_GAP = (_N_PTS - 1) * 0.01, 0.008
_LAYER_STEP = FILAMENT_RADIUS * 2.6
_Y_SCALE = 3.0   # lateral-offset exaggeration so deviation reads at a glance
_MM = 1000.0     # metres → millimetres for display


# ── 3-D print geometry ────────────────────────────────────────────────────────

def _make_filament_tube(
    xs: List[float], ys: List[float], z_center: float, radius: float, n_circ: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, Z) surface arrays for a cylindrical tube following (xs, ys) at z_center."""
    phi = np.linspace(0, 2.0 * np.pi, n_circ, endpoint=True)[:, np.newaxis]
    XS = np.array(xs, dtype=float)[np.newaxis, :]
    YS = np.array(ys, dtype=float)[np.newaxis, :]
    return np.repeat(XS, n_circ, axis=0), YS + radius * np.cos(phi), z_center + radius * np.sin(phi)


def _draw_print(ax: Any, offsets: Dict[Tuple[int, int], np.ndarray], norm: Normalize) -> None:
    """Render one print as stacked filament tubes: designed = wireframe ghost,
    measured = solid tube coloured + offset by its deviation."""
    for li in range(_N_LAYERS):
        z, x_off = li * _LAYER_STEP, 0.0
        for si in range(_N_SEGMENTS):
            dev = offsets[(li, si)]
            xs = [i * 0.01 + x_off for i in range(_N_PTS)]
            color = DEVIATION_CMAP(norm(float(np.mean(np.abs(dev)))))
            Xd, Yd, Zd = _make_filament_tube(xs, [0.0] * _N_PTS, z, FILAMENT_RADIUS, n_circ=16)
            ax.plot_wireframe(Xd, Yd, Zd, color=STEEL_300, alpha=0.3, linewidth=0.4, rstride=4, cstride=1)
            ax.plot(xs, [0.0] * _N_PTS, [z] * _N_PTS, color=STEEL_300, linestyle="--", linewidth=0.9, alpha=0.7, zorder=4)
            Xm, Ym, Zm = _make_filament_tube(xs, list(np.asarray(dev) * _Y_SCALE), z, FILAMENT_RADIUS, n_circ=20)
            ax.plot_surface(Xm, Ym, Zm, color=color, alpha=0.92, linewidth=0, antialiased=True, shade=True)
            x_off += _SEG_LEN + _SEG_GAP
    ax.set_box_aspect([9, 2.5, 2.2])
    clean_3d_panes(ax)
    ax.set_xlabel("Along-path [m]", labelpad=10)
    ax.set_ylabel(f"Lateral offset [×{_Y_SCALE:.0f}]", labelpad=10)
    ax.set_zticks([i * _LAYER_STEP for i in range(_N_LAYERS)])
    ax.set_zticklabels([f"L{i}" for i in range(_N_LAYERS)])
    ax.view_init(elev=28, azim=-62)


def _segment_deviations(camera: Any, params: Dict[str, Any]) -> Dict[Tuple[int, int], np.ndarray]:
    """Per-(layer, segment) point-wise lateral deviation [m] for one experiment."""
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for li in range(_N_LAYERS):
        for si in range(_N_SEGMENTS):
            data = camera.get_segment_data(params, li, si)
            out[(li, si)] = np.array([mp[1] - dp[1] for mp, dp in zip(data["measured_path"], data["designed_path"])])
    return out


class StageField(NamedTuple):
    """Averaged deviation field for one campaign stage."""
    offsets: Dict[Tuple[int, int], np.ndarray]
    mean_dev: float
    max_dev: float


def stage_average_field(camera: Any, params_list: List[Dict[str, Any]]) -> StageField:
    """Average the per-segment lateral deviation across a stage's experiments."""
    acc = {(li, si): np.zeros(_N_PTS) for li in range(_N_LAYERS) for si in range(_N_SEGMENTS)}
    for p in params_list:
        for key, dev in _segment_deviations(camera, p).items():
            acc[key] = acc[key] + dev
    n = max(len(params_list), 1)
    offsets = {k: v / n for k, v in acc.items()}
    mean_dev = float(np.mean([np.abs(v).mean() for v in offsets.values()]))
    max_dev = max(float(np.abs(v).max()) for v in offsets.values())
    return StageField(offsets, mean_dev, max_dev)


def plot_stage_print(field: StageField, phase: str, vmax: float, name: str) -> None:
    """Render one stage's average print (shared `vmax` keeps stages comparable)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    norm = Normalize(0.0, float(vmax) * 1.05)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_print(ax, field.offsets, norm)
    sm = ScalarMappable(cmap=DEVIATION_CMAP, norm=Normalize(0.0, float(vmax) * 1.05 * _MM))
    sm.set_array([])
    style_colorbar(fig.colorbar(sm, ax=ax, pad=0.09, shrink=0.5, aspect=16), "Path deviation [mm]")
    ax.set_title(PHASE_LABELS.get(phase, phase.title()), fontsize=FONT["title"] + 1,
                 color=PHASE_COLORS.get(phase, ZINC_700), pad=6)
    fig.text(0.5, 0.9, f"average print  —  mean deviation {field.mean_dev * _MM:.2f} mm",
             ha="center", va="top", fontsize=FONT["subtitle"], color=ZINC_400, style="italic")
    save_fig(name)


# ── Parameter space over the true-physics performance topology ────────────────

def plot_parameter_topology(
    all_params: List[Dict[str, Any]], phases: List[str],
    water: np.ndarray, speed: np.ndarray, perf_grid: np.ndarray,
    optimum: Tuple[float, float, float], name: str = "parameter_space",
) -> None:
    """Sampled points over the true-physics performance landscape, with the
    theoretical optimum marked — shows the agent converging on the truth."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Parameter space over true performance topology",
                 fontsize=FONT["title"], color=ZINC_700, fontweight="bold")

    cf = ax.contourf(water, speed, perf_grid, levels=24, cmap=PERFORMANCE_CMAP, alpha=0.95)
    ax.contour(water, speed, perf_grid, levels=8, colors="white", linewidths=0.3, alpha=0.4)

    def _xy(phase: str) -> Tuple[List[float], List[float]]:
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        return ([all_params[i]["water_ratio"] for i in idx], [all_params[i]["print_speed"] for i in idx])

    bx, by = _xy("baseline")
    ax.scatter(bx, by, facecolors="white", edgecolors=ZINC_700, linewidths=0.7, s=46,
               label=PHASE_LABELS["baseline"], zorder=3)
    ex, ey = _xy("exploration")
    ax.scatter(ex, ey, c=STEEL_500, edgecolors="white", linewidths=0.5, s=50,
               label=PHASE_LABELS["exploration"], zorder=3)
    ix, iy = _xy("inference")
    ax.scatter(ix, iy, c=EMERALD_500, edgecolors="white", linewidths=0.6, s=58,
               label=PHASE_LABELS["inference"], zorder=4)

    w_opt, s_opt, _ = optimum
    ax.scatter([w_opt], [s_opt], marker="x", c="white", s=110, linewidths=1.8, zorder=6, label="Theoretical optimum")

    ax.set_xlabel("Water ratio"); ax.set_ylabel("Print speed [mm/s]")
    ax.set_xlim(water.min(), water.max()); ax.set_ylim(speed.min(), speed.max())
    ax.legend(loc="upper left", framealpha=0.85)
    clean_spines(ax)
    style_colorbar(fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02), "System performance $S$")
    save_fig(name)


# ── Performance trajectory ────────────────────────────────────────────────────

def plot_performance_trajectory(
    perf_history: List[Tuple[Dict[str, Any], Dict[str, float]]], phases: List[str],
) -> None:
    """Path-accuracy + energy-efficiency across the campaign, with phase markers."""
    path_acc = [pp[1].get("path_accuracy", float("nan")) for pp in perf_history]
    energy_eff = [pp[1].get("energy_efficiency", float("nan")) for pp in perf_history]
    xs = list(range(1, len(path_acc) + 1))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.suptitle("Performance trajectory", fontsize=FONT["title"], color=ZINC_700, fontweight="bold")
    for phase in ("baseline", "exploration", "inference"):
        idx = [i for i, ph in enumerate(phases) if ph == phase]
        if not idx:
            continue
        if idx[0] > 0:
            ax.axvline(idx[0] + 0.5, color=ZINC_300, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text((idx[0] + idx[-1]) / 2 + 1, 1.04, PHASE_LABELS[phase], ha="center", va="bottom",
                color=PHASE_COLORS[phase], fontsize=FONT["legend"], fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.plot(xs, path_acc, marker="o", color=STEEL_500, linewidth=1.8, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.4, label="Path accuracy")
    ax.plot(xs, energy_eff, marker="o", color=EMERALD_500, linewidth=1.8, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.4, label="Energy efficiency")
    ax.set_xlabel("Experiment #"); ax.set_ylabel("Score [0–1]")
    ax.set_ylim(0, 1.08); ax.set_xlim(0.5, len(xs) + 0.5)
    ax.legend(loc="lower right")
    clean_spines(ax); light_grid(ax, axis="y")
    save_fig("performance_trajectory")


# ── Feature heatmaps & prediction accuracy ────────────────────────────────────

def _annotate_heatmap(ax: Any, grid: np.ndarray, fmt: str = "{:.4f}") -> None:
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, fmt.format(float(grid[i, j])), ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")


def plot_feature_heatmaps(exp_data: ExperimentData) -> None:
    """5×4 heatmaps of path_deviation (quality) and energy_per_segment."""
    cfg = [("path_deviation", "Path deviation [m]", DEVIATION_CMAP),
           ("energy_per_segment", "Energy per segment [J]", plt.cm.Blues)]  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Feature heatmaps — {exp_data.code}", fontsize=FONT["title"], color=ZINC_700, fontweight="bold")
    for ax, (fname, label, cmap) in zip(axes, cfg):
        grid = np.asarray(exp_data.features.get_value(fname))
        im = ax.imshow(grid, aspect="auto", cmap=cmap)
        ax.set_xlabel("Segment"); ax.set_ylabel("Layer")
        ax.set_title(label, fontsize=FONT["axis"], color=ZINC_700)
        ax.set_xticks(range(grid.shape[1])); ax.set_yticks(range(grid.shape[0]))
        style_colorbar(plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04))
        _annotate_heatmap(ax, grid)
    save_fig("feature_heatmaps")


def plot_prediction_accuracy(agent: PfabAgent, datamodule: DataModule) -> None:
    """Predicted vs actual scatter for each output feature, with R²."""
    from sklearn.metrics import r2_score
    from pred_fab.utils import SplitType  # type: ignore[attr-defined]

    outputs = agent.pred_system.get_system_outputs()
    val_batches = datamodule.get_batches(SplitType.VAL)
    if not val_batches:
        print("[plot_prediction_accuracy] No validation batches available.")
        return
    y_val = np.vstack([b[1] for b in val_batches])
    X_val = np.vstack([b[0] for b in val_batches])
    y_pred_list = [m.forward_pass(X_val) for m in agent.pred_system.models]
    y_pred = np.hstack(y_pred_list) if y_pred_list else y_val

    n_out = min(y_val.shape[1], len(outputs))
    fig, axes = plt.subplots(1, n_out, figsize=(5.2 * n_out, 4.6))
    if n_out == 1:
        axes = [axes]  # type: ignore[assignment]
    fig.suptitle("Prediction accuracy — validation set", fontsize=FONT["title"], color=ZINC_700, fontweight="bold")
    for i, (ax, fname) in enumerate(zip(axes, outputs)):
        if i >= y_pred.shape[1] or i >= y_val.shape[1]:
            break
        r2 = r2_score(y_val[:, i], y_pred[:, i])
        lim = [min(float(y_val[:, i].min()), float(y_pred[:, i].min())),
               max(float(y_val[:, i].max()), float(y_pred[:, i].max()))]
        ax.plot(lim, lim, linestyle="--", linewidth=0.8, color=ZINC_300, alpha=0.8, zorder=1)
        ax.scatter(y_val[:, i], y_pred[:, i], s=26, c=STEEL_500, edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{fname}    R² = {r2:.3f}", fontsize=FONT["axis"], color=ZINC_700)
        ax.set_xlim(lim); ax.set_ylim(lim)
        clean_spines(ax); light_grid(ax)
    save_fig("prediction_accuracy")
