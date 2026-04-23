"""Concept 05 — estimating ∫_[0,1]^D E(z) dz via Sobol quasi-Monte Carlo.

The integrated evidence is just a cube integral of a bounded smooth field:

    ∫_[0,1]^D  E(z) dz   with   E(z) = D(z) / (1 + D(z))

Sobol sampling is the chosen estimator — see [[PFAB - Integrated Evidence
Estimator]] in the knowledge base for the rationale (stateless, O(N·n) cost,
≤1 % error at σ ≥ 0.05, paired with a σ ≥ 0.03 floor for robustness).

This script:
    1. Visualises Sobol probes on top of the E(z) field — you can literally
       see where the estimator evaluates.
    2. Shows three configurations (stacked, spread, at-boundary) with their
       ∫E values — makes the "interior beats corner" behaviour quantitative.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from _style import (
    apply_style, ZINC, RED,
    evidence_cmap, save, strip_spines,
)
from concept_02_raw_density import raw_density, grid_2d
from concept_03_actual_evidence import E_of_D


# ---------- Core estimator ----------

def integrated_E_sobol(
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    n: int = 512,
    seed: int = 0,
) -> tuple[float, dict]:
    """∫_[0,1]^D E(z) dz via scrambled Sobol quasi-Monte Carlo.

    Returns (estimate, diagnostics). The integrand E(z) = D(z)/(1+D(z)) is
    bounded and smooth on the cube, so uniform low-discrepancy sampling
    converges at ≈ O(log(n)^D / n) for our operating σ range.
    """
    D = centers.shape[1]
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, weights, sigma)
    E_vals = D_vals / (1.0 + D_vals)
    total = float(E_vals.mean())
    diag = {
        "n_probes": n,
        "frac_saturated": float((E_vals > 0.5).mean()),
    }
    return total, diag


# ---------- Figure: Sobol probes overlay on E(z) field ----------

def figure_probes_overlay(sigma: float = 0.12, n: int = 512):
    apply_style()
    u, v, Z = grid_2d(res=201)
    centers = np.array([
        [0.30, 0.40], [0.65, 0.30], [0.80, 0.70], [0.35, 0.80],
    ])
    w = np.ones(len(centers))
    D_field = raw_density(Z, centers, w, sigma)
    E_field = E_of_D(D_field)

    sobol = qmc.Sobol(d=2, scramble=True, rng=0).random(n=n)
    D_at_probes = raw_density(sobol, centers, w, sigma)
    E_at_probes = E_of_D(D_at_probes)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.contourf(u, v, E_field, levels=25, cmap=evidence_cmap(), vmin=0, vmax=1)
    ax.contour(u, v, E_field, levels=[0.25, 0.5, 0.75],
               colors=ZINC[400], linewidths=0.5, alpha=0.6)

    # Size probes by their E value so saturated points pop
    sizes = 6 + 28 * E_at_probes
    ax.scatter(sobol[:, 0], sobol[:, 1],
               c=ZINC[700], s=sizes, alpha=0.55, edgecolors="none", zorder=4)

    ax.scatter(centers[:, 0], centers[:, 1],
               c=RED, marker="x", s=60, linewidths=1.5, zorder=6)

    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    strip_spines(ax)
    fig.colorbar(im, ax=ax, shrink=0.85, label="E(z)")

    total, diag = integrated_E_sobol(centers, w, sigma, n=n)
    ax.set_title(
        f"E(z) field + Sobol probes (size ∝ E)\n"
        f"σ={sigma}, N={len(centers)} kernels, "
        f"∫E dz ≈ {total:.3f}, n = {diag['n_probes']}",
        fontsize=10, color=ZINC[700],
    )
    fig.suptitle(
        "Sobol probes overlaid on the evidence field",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "05a_probes_overlay")
    plt.close(fig)
    return path


# ---------- Figure: three configurations ----------

def figure_configurations(sigma: float = 0.12):
    apply_style()
    u, v, Z = grid_2d(res=201)

    configs = [
        ("stacked at center",     np.array([[0.5, 0.5]] * 5)),
        ("spread interior",       np.array([
            [0.25, 0.30], [0.75, 0.25], [0.55, 0.55],
            [0.25, 0.75], [0.80, 0.80],
        ])),
        ("stacked at corner",     np.array([[0.02, 0.02]] * 5)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (label, centers) in zip(axes, configs):
        w = np.ones(len(centers))
        D = raw_density(Z, centers, w, sigma)
        E = E_of_D(D)
        total, _ = integrated_E_sobol(centers, w, sigma)

        im = ax.contourf(u, v, E, levels=25, cmap=evidence_cmap(), vmin=0, vmax=1)
        ax.contour(u, v, E, levels=[0.25, 0.5, 0.75],
                   colors=ZINC[400], linewidths=0.5, alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1], c=RED, marker="x",
                   s=60, linewidths=1.5, zorder=6)
        ax.set_title(f"{label}\n∫E dz ≈ {total:.3f}", pad=6, fontsize=10)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)

    fig.colorbar(im, ax=axes[-1], shrink=0.85, label="E")  # type: ignore[possibly-undefined]
    fig.suptitle(
        f"∫E dz decides placement — stacked is wasteful, boundary leaks   (σ={sigma})",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "05b_configurations")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 70)
    print("∫E dz for different configurations  (Gaussian, σ=0.12, 5 points)")
    print("=" * 70)
    configs = {
        "5 stacked at (0.5, 0.5)":    np.array([[0.5, 0.5]] * 5),
        "5 stacked at (0.02, 0.02)":  np.array([[0.02, 0.02]] * 5),
        "5 spread interior":          np.array([
            [0.25, 0.30], [0.75, 0.25], [0.55, 0.55],
            [0.25, 0.75], [0.80, 0.80],
        ]),
        "5 at corners of cube":       np.array([
            [0.02, 0.02], [0.02, 0.98], [0.98, 0.02],
            [0.98, 0.98], [0.50, 0.50],
        ]),
    }
    for name, centers in configs.items():
        w = np.ones(len(centers))
        total, diag = integrated_E_sobol(centers, w, sigma=0.12)
        print(f"{name:<32s}  ∫E dz = {total:.4f}  "
              f"(n = {diag['n_probes']}, "
              f"{100 * diag['frac_saturated']:.0f}% saturated)")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_probes_overlay()
    p2 = figure_configurations()
    for p in (p1, p2):
        print(f"Saved: {p}")
