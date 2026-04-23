"""Concept 07 — 3-D evidence packing, *stencil-only*.

No grid. Only the points the SamplingSpace actually evaluates:

    - yellow × at each placement z_j (the kernel centers)
    - a stencil of points on concentric (D−1)-spheres around each z_j,
      coloured by E(z_sample) using the Blues spectrum

This is what the method "sees" — nothing more, nothing less. Far cleaner
than a dense grid because every rendered dot is an evaluation the estimator
performs.

Gaussian kernel, σ set explicitly. No random sampling.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import differential_evolution

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW,
    save, uncertainty_cmap,
)
from sampling_space import SamplingSpace
from concept_02_raw_density import raw_density
from concept_03_actual_evidence import E_of_D
from concept_05_integrated_objective import integrated_E_via_stencil


# ---------- Placement via stencil-driven DE ----------

def optimize_placements(
    N: int = 5,
    D: int = 3,
    sigma: float = 0.12,
    seed: int = 0,
) -> np.ndarray:
    """Joint DE maximisation of ∫E dz over N placements in [0,1]^D."""
    def objective(x_flat: np.ndarray) -> float:
        centers = x_flat.reshape(N, D)
        I, _ = integrated_E_via_stencil(
            centers, np.ones(N), sigma=sigma, kernel="gaussian",
        )
        return -I

    bounds = [(0.02, 0.98)] * (N * D)
    res = differential_evolution(
        objective, bounds, seed=seed, maxiter=100, tol=1e-6,
        popsize=15, mutation=(0.5, 1.0), recombination=0.7, workers=1,
    )
    return res.x.reshape(N, D)


# ---------- Physical axis labels ----------

_PARAM_SPEC = {
    "water_ratio": ("water ratio",      (0.30, 0.50), "{:.2f}"),
    "print_speed": ("print speed [mm/s]", (20.0, 60.0), "{:.0f}"),
    "n_layers":    ("n_layers",          (4.0, 8.0),  "{:.0f}"),
}


def _label_axis(ax_set_ticks, ax_set_ticklabels, ax_set_label, param: str, n_ticks: int = 5, fontsize: int = 8):
    label, (lo, hi), fmt = _PARAM_SPEC[param]
    u = np.linspace(0, 1, n_ticks)
    ax_set_ticks(u)
    ax_set_ticklabels([fmt.format(lo + v * (hi - lo)) for v in u], fontsize=fontsize)
    ax_set_label(label, labelpad=6, color=ZINC[600])


# ---------- Core plot ----------

def _build_stencil_data(
    placements: np.ndarray, sigma: float, n_directions: int = 16,
    skip_center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (shell_samples (M, 3), E_at_samples (M,), in_domain_mask (M,)).

    `skip_center=True` drops the r=0 stencil point per kernel (it sits exactly
    on the placement, where the yellow × already lives).
    """
    D = placements.shape[1]
    N = placements.shape[0]
    space = SamplingSpace(D=D, sigma=sigma, kernel="gaussian",
                           n_directions=n_directions)
    # (N, M_stencil, D). Offsets[0] is the center of the stencil.
    samples_per = space.samples_for_batch(placements)
    if skip_center:
        samples_per = samples_per[:, 1:, :]  # drop the r=0 offset

    all_samples = samples_per.reshape(-1, D)

    # E at each sample, considering ALL placements' kernels
    D_vals = raw_density(all_samples, placements, np.ones(N), sigma, kernel="gaussian")
    E_vals = E_of_D(D_vals)

    in_domain = np.all((all_samples >= 0.0) & (all_samples <= 1.0), axis=1)
    return all_samples, E_vals, in_domain


def figure_stencil_packing(
    N: int = 5,
    sigma: float = 0.12,
    n_directions: int = 16,
    seed: int = 0,
    title_suffix: str = "",
):
    """Single 3-D view of yellow placements + blue stencil cloud."""
    apply_style()

    placements = optimize_placements(N=N, sigma=sigma, seed=seed)
    samples, E_at_samp, in_dom = _build_stencil_data(placements, sigma, n_directions)

    cmap = uncertainty_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)
    rgba = cmap(norm(E_at_samp))
    # Out-of-domain samples: greyed out so they're visibly "wasted"
    out_mask = ~in_dom
    rgba[out_mask] = [*[x / 255 for x in (180, 180, 180)], 0.20]
    # In-domain: alpha ramps with E
    rgba[in_dom, 3] = 0.25 + 0.65 * E_at_samp[in_dom]

    sizes = np.where(in_dom, 8.0 + 28.0 * E_at_samp, 5.0)

    fig = plt.figure(figsize=(9.0, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Stencil cloud
    ax.scatter(
        samples[:, 0], samples[:, 1], samples[:, 2],
        c=rgba, s=sizes, edgecolors="none", depthshade=False,
    )

    # Placements — bullseye of three markers so they pop against the blue cloud
    ax.scatter(
        placements[:, 0], placements[:, 1], placements[:, 2],
        c=YELLOW, marker="o", s=420, linewidths=0.0, alpha=1.0, zorder=18,
        depthshade=False,
    )
    ax.scatter(
        placements[:, 0], placements[:, 1], placements[:, 2],
        c="white", marker="o", s=220, linewidths=0.0, alpha=1.0, zorder=19,
        depthshade=False,
    )
    ax.scatter(
        placements[:, 0], placements[:, 1], placements[:, 2],
        c=YELLOW, marker="x", s=260, linewidths=3.5, zorder=20,
        depthshade=False,
    )

    # Axes in physical units
    _label_axis(ax.set_xticks, ax.set_xticklabels, ax.set_xlabel, "water_ratio")
    _label_axis(ax.set_yticks, ax.set_yticklabels, ax.set_ylabel, "print_speed")
    _label_axis(ax.set_zticks, ax.set_zticklabels, ax.set_zlabel, "n_layers")
    ax.tick_params(axis="both", colors=ZINC[500], labelsize=7)
    ax.xaxis.pane.set_edgecolor(ZINC[300])
    ax.yaxis.pane.set_edgecolor(ZINC[300])
    ax.zaxis.pane.set_edgecolor(ZINC[300])
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=22, azim=35)

    I_total, _ = integrated_E_via_stencil(placements, np.ones(N), sigma=sigma, kernel="gaussian")

    fig.suptitle(
        f"Stencil-only evidence packing   |   N={N}, σ={sigma},   "
        f"∫E dz = {I_total:.3f}   |   {len(samples)} stencil points total"
        + (f"   |   {title_suffix}" if title_suffix else ""),
        fontsize=10, color=ZINC[700],
    )
    path = save(fig, f"07a_stencil_packing_N{N}_sigma{sigma}")
    plt.close(fig)
    return path, placements


def figure_stencil_packing_rotations(
    N: int = 5,
    sigma: float = 0.12,
    n_directions: int = 16,
    seed: int = 0,
):
    """Same scene, three viewing angles."""
    apply_style()

    placements = optimize_placements(N=N, sigma=sigma, seed=seed)
    samples, E_at_samp, in_dom = _build_stencil_data(placements, sigma, n_directions)

    cmap = uncertainty_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)
    rgba = cmap(norm(E_at_samp))
    out_mask = ~in_dom
    rgba[out_mask] = [*[x / 255 for x in (180, 180, 180)], 0.20]
    rgba[in_dom, 3] = 0.25 + 0.65 * E_at_samp[in_dom]
    sizes = np.where(in_dom, 8.0 + 28.0 * E_at_samp, 5.0)

    views = [(22, 35), (22, 125), (62, 35)]
    fig = plt.figure(figsize=(15.5, 5.8), constrained_layout=True)
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        ax.scatter(
            samples[:, 0], samples[:, 1], samples[:, 2],
            c=rgba, s=sizes, edgecolors="none", depthshade=False,
        )
        ax.scatter(
            placements[:, 0], placements[:, 1], placements[:, 2],
            c=YELLOW, marker="o", s=320, linewidths=0.0, alpha=1.0, zorder=18,
            depthshade=False,
        )
        ax.scatter(
            placements[:, 0], placements[:, 1], placements[:, 2],
            c="white", marker="o", s=160, linewidths=0.0, alpha=1.0, zorder=19,
            depthshade=False,
        )
        ax.scatter(
            placements[:, 0], placements[:, 1], placements[:, 2],
            c=YELLOW, marker="x", s=200, linewidths=3.0, zorder=20,
            depthshade=False,
        )
        _label_axis(ax.set_xticks, ax.set_xticklabels, ax.set_xlabel, "water_ratio", fontsize=7)
        _label_axis(ax.set_yticks, ax.set_yticklabels, ax.set_ylabel, "print_speed", fontsize=7)
        _label_axis(ax.set_zticks, ax.set_zticklabels, ax.set_zlabel, "n_layers", fontsize=7)
        ax.xaxis.pane.set_edgecolor(ZINC[300])
        ax.yaxis.pane.set_edgecolor(ZINC[300])
        ax.zaxis.pane.set_edgecolor(ZINC[300])
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
        ax.grid(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.tick_params(axis="both", colors=ZINC[500], labelsize=6)
        ax.set_title(f"elev={elev}°, azim={azim}°", fontsize=9, color=ZINC[600], pad=0)

    fig.suptitle(
        f"Stencil-only evidence packing — three views   |   N={N}, σ={sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"07b_stencil_packing_rotations_N{N}_sigma{sigma}")
    plt.close(fig)
    return path


# ---------- Main ----------

if __name__ == "__main__":
    print("Optimising placements and rendering stencil-only 3-D evidence views...")
    for N in (3, 5):
        for sigma in (0.10, 0.15):
            p, pls = figure_stencil_packing(N=N, sigma=sigma)
            print(f"Saved: {p}")
            for k, pt in enumerate(pls):
                print(
                    f"    z_{k}: water={0.30 + pt[0] * 0.20:.3f}  "
                    f"speed={20 + pt[1] * 40:.2f}  n_layers={4 + pt[2] * 4:.2f}"
                )
    p_rot = figure_stencil_packing_rotations(N=5, sigma=0.12)
    print(f"Saved: {p_rot}")
