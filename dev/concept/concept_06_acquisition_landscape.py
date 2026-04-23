"""Concept 06 — what the optimizer sees.

Given a fixed set of existing points, the acquisition Δ∫E(z_new) tells us
how much a new candidate placed at z_new would raise the integrated evidence.
Sweep z_new across the domain → we get the acquisition surface that DE
maximises.

This is the key diagnostic: **if this surface has a clear maximum at
interior points away from existing data, the optimizer will find good
placements. If it's flat or degenerate, we have a problem.**

Pure stencil, no random anything.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    evidence_cmap, save, strip_spines,
)
from concept_02_raw_density import raw_density, grid_2d
from concept_03_actual_evidence import E_of_D
from concept_05_integrated_objective import integrated_E_sobol


def acquisition_landscape(
    existing_centers: np.ndarray,
    existing_weights: np.ndarray,
    sigma: float,
    res: int = 41,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Δ∫E(z_new) for z_new swept over [0,1]^2.

    Returns (x, y, Δ∫E_grid).
    """
    x = np.linspace(0.02, 0.98, res)
    y = np.linspace(0.02, 0.98, res)
    I_old, _ = integrated_E_sobol(existing_centers, existing_weights, sigma=sigma)

    dI = np.zeros((res, res))
    for j, yv in enumerate(y):
        for i, xv in enumerate(x):
            z_new = np.array([xv, yv])
            centers = np.vstack([existing_centers, z_new[None, :]])
            weights = np.concatenate([existing_weights, [1.0]])
            I_new, _ = integrated_E_sobol(centers, weights, sigma=sigma)
            dI[j, i] = I_new - I_old
    return x, y, dI


# ---------- Figure: acquisition landscape at different sigmas ----------

def figure_landscape_sigma_sweep():
    apply_style()
    existing = np.array([[0.30, 0.60], [0.70, 0.35]])
    e_weights = np.ones(len(existing))

    sigmas = [0.05, 0.10, 0.15, 0.25]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    for ax, sigma in zip(axes, sigmas):
        x, y, dI = acquisition_landscape(existing, e_weights, sigma, res=33)
        u, v, Z = grid_2d(res=121)
        D_now = raw_density(Z, existing, e_weights, sigma)
        E_now = E_of_D(D_now)

        im1 = ax.contourf(u, v, E_now, levels=20, cmap=evidence_cmap(),
                          vmin=0, vmax=1, alpha=0.35)
        cs = ax.contourf(x, y, dI, levels=20, cmap="YlOrRd", alpha=0.85)
        ax.contour(x, y, dI, levels=8, colors=ZINC[400], linewidths=0.4, alpha=0.6)

        j_max, i_max = np.unravel_index(np.argmax(dI), dI.shape)
        ax.scatter(x[i_max], y[j_max], marker="x", c=YELLOW, s=90, linewidths=1.5, zorder=6)
        ax.scatter(existing[:, 0], existing[:, 1],
                   c="white", edgecolors=ZINC[900], s=45, linewidths=0.8, zorder=5)

        ax.set_title(f"σ = {sigma}\nmax Δ∫E = {dI.max():.3f} at ({x[i_max]:.2f}, {y[j_max]:.2f})",
                     pad=6, fontsize=9)
        ax.set_xlabel("z_new,1")
        ax.set_ylabel("z_new,2")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        strip_spines(ax)

    fig.suptitle(
        "Δ∫E(z_new) — where would a new point contribute most?",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "06a_acquisition_landscape_sigma_sweep")
    plt.close(fig)
    return path


# ---------- Figure: landscape for 5 existing at a fixed σ ----------

def figure_landscape_with_existing(sigma: float = 0.12):
    apply_style()
    existing = np.array([
        [0.20, 0.25], [0.75, 0.20], [0.45, 0.50],
        [0.25, 0.80], [0.85, 0.75],
    ])
    e_weights = np.ones(len(existing))

    x, y, dI = acquisition_landscape(existing, e_weights, sigma, res=41)

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    im = ax.contourf(x, y, dI, levels=25, cmap="YlOrRd")
    ax.contour(x, y, dI, levels=10, colors=ZINC[400], linewidths=0.4, alpha=0.5)
    ax.scatter(existing[:, 0], existing[:, 1],
               c="white", edgecolors=ZINC[900], s=50, linewidths=0.8, zorder=5,
               label="existing")

    j_max, i_max = np.unravel_index(np.argmax(dI), dI.shape)
    ax.scatter(x[i_max], y[j_max], marker="x", c=YELLOW, s=100, linewidths=2.0,
               zorder=6, label="argmax Δ∫E")

    ax.set_xlabel("z_new,1")
    ax.set_ylabel("z_new,2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    strip_spines(ax)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Δ∫E")

    fig.suptitle(
        f"Acquisition surface for the 6th point — existing 5 kernels, σ = {sigma}, "
        f"max Δ∫E = {dI.max():.3f}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "06b_acquisition_5existing")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 70)
    print("Acquisition Δ∫E at a grid of candidate positions")
    print("(2 existing kernels at (0.30, 0.60) and (0.70, 0.35), σ=0.12)")
    print("=" * 70)

    existing = np.array([[0.30, 0.60], [0.70, 0.35]])
    e_weights = np.ones(2)
    I_old, _ = integrated_E_sobol(existing, e_weights, sigma=0.12)
    print(f"∫E(existing)   = {I_old:.4f}")
    print()

    test_points = [
        ("center",          np.array([0.50, 0.50])),
        ("interior gap",    np.array([0.20, 0.20])),
        ("corner",          np.array([0.02, 0.02])),
        ("on existing A",   np.array([0.30, 0.60])),
        ("near existing B", np.array([0.70, 0.30])),
    ]
    for label, z_new in test_points:
        centers = np.vstack([existing, z_new[None, :]])
        weights = np.concatenate([e_weights, [1.0]])
        I_new, _ = integrated_E_sobol(centers, weights, sigma=0.12)
        dI = I_new - I_old
        print(f"z_new = {label:<16s}  Δ∫E = {dI:+.4f}")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_landscape_sigma_sweep()
    p2 = figure_landscape_with_existing()
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
