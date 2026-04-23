"""Concept 05 — estimating ∫_[0,1]^D E(z) dz via a KernelField.

Putting the pieces together:

    ∫ E(z) dz  =  Σ_j  w_j · E_{z ~ ρ_j}[ 1 / (1 + D(z)) ]

The inner expectation is approximated deterministically by the KernelField
(a field of probes on concentric shells around each kernel). Each kernel
contributes its own "importance-weighted" view of the integrand; sum across
kernels to get the whole thing.

This script:
    1. Visualises probes on top of the E(z) field for a small configuration
       — you can literally see where the estimator evaluates.
    2. Shows three configurations (stacked, spread, at-boundary) with their
       ∫E values — makes the "interior beats corner" behaviour quantitative.

The stencil-vs-Sobol comparison lives in concept_10 (proper sample-budget
matching).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    evidence_cmap, save, strip_spines,
)
from kernel_field import KernelField
from concept_02_raw_density import raw_density, grid_2d
from concept_03_actual_evidence import E_of_D


# ---------- Core estimator ----------

def integrated_E_via_field(
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    n_directions: int = 16,
    radius_multipliers: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0),
    domain_low: float = 0.0,
    domain_high: float = 1.0,
) -> tuple[float, dict]:
    """∫_domain E(z) dz via a KernelField at each kernel center.

    Returns (estimate, diagnostics).
    """
    D = centers.shape[1]
    field = KernelField(
        D=D, sigma=sigma,
        radius_multipliers=radius_multipliers, n_directions=n_directions,
    )

    all_probes = field.probes_for_batch(centers)  # (N, M, D)
    N = all_probes.shape[0]
    M = all_probes.shape[1]
    probes_flat = all_probes.reshape(-1, D)

    in_domain = np.all(
        (probes_flat >= domain_low) & (probes_flat <= domain_high),
        axis=1,
    )

    D_vals = raw_density(probes_flat, centers, weights, sigma)
    integrand = 1.0 / (1.0 + D_vals)
    integrand *= in_domain.astype(float)
    integrand_mat = integrand.reshape(N, M)
    per_kernel_int = (integrand_mat * field.weights[None, :]).sum(axis=1)
    total = float((weights * per_kernel_int).sum())
    diag = {
        "n_probes_total": probes_flat.shape[0],
        "frac_in_domain": float(in_domain.mean()),
        "per_kernel": per_kernel_int,
    }
    return total, diag


# Alias for backward compatibility with other concept scripts
integrated_E_via_stencil = integrated_E_via_field


# ---------- Figure: probes overlay on E(z) field ----------

def figure_field_overlay(sigma: float = 0.12):
    apply_style()
    u, v, Z = grid_2d(res=201)
    centers = np.array([
        [0.30, 0.40], [0.65, 0.30], [0.80, 0.70], [0.35, 0.80],
    ])
    w = np.ones(len(centers))
    D_field = raw_density(Z, centers, w, sigma)
    E_field = E_of_D(D_field)

    field = KernelField(D=2, sigma=sigma, n_directions=16)
    all_probes = field.probes_for_batch(centers)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.contourf(u, v, E_field, levels=25, cmap=evidence_cmap(), vmin=0, vmax=1)
    ax.contour(u, v, E_field, levels=[0.25, 0.5, 0.75],
               colors=ZINC[400], linewidths=0.5, alpha=0.6)

    for k in range(all_probes.shape[0]):
        ax.scatter(
            all_probes[k, :, 0], all_probes[k, :, 1],
            c=ZINC[600], s=4, alpha=0.6, zorder=4,
        )

    ax.scatter(centers[:, 0], centers[:, 1],
               c=RED, marker="x", s=60, linewidths=1.5, zorder=6)

    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    strip_spines(ax)
    fig.colorbar(im, ax=ax, shrink=0.85, label="E(z)")

    total, diag = integrated_E_via_field(centers, w, sigma)
    ax.set_title(
        f"E(z) field + kernel-field probes\n"
        f"σ={sigma}, N={len(centers)} kernels, "
        f"∫E dz ≈ {total:.3f}, "
        f"probes in-domain: {100 * diag['frac_in_domain']:.0f}%",
        fontsize=10, color=ZINC[700],
    )
    fig.suptitle(
        "Kernel-field probes overlaid on the evidence field",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "05a_field_overlay")
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
        total, _ = integrated_E_via_field(centers, w, sigma)

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
        total, diag = integrated_E_via_field(centers, w, sigma=0.12)
        print(f"{name:<32s}  ∫E dz = {total:.4f}  "
              f"({100 * diag['frac_in_domain']:.0f}% probes in-domain)")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_field_overlay()
    p2 = figure_configurations()
    for p in (p1, p2):
        print(f"Saved: {p}")
