"""Concept 05 — estimating ∫_[0,1]^D E(z) dz via SamplingSpace.

Putting the pieces together:

    ∫ E(z) dz  =  Σ_j  w_j · E_{z ~ ρ_j}[ 1 / (1 + D(z)) ]

where the inner expectation is computed by the stencil. Each kernel contributes
its own "importance-weighted" view of the integrand; sum across kernels to get
the whole thing.

This script:
    1. Visualises stencil samples layered on top of the E(z) field for a
       small configuration — you can literally *see* where the estimator
       probes.
    2. Compares stencil estimate against a dense reference MC at different σ,
       demonstrating low variance even when σ is very small.
    3. Shows three configurations (stacked, spread, at-boundary) with their
       ∫E value — makes the "interior beats corner" behaviour quantitative.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    evidence_cmap, save, strip_spines,
)
from sampling_space import SamplingSpace
from concept_02_raw_density import raw_density, grid_2d
from concept_03_actual_evidence import E_of_D


# ---------- Core estimator ----------

def integrated_E_via_stencil(
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    kernel: str = "gaussian",
    n_directions: int = 16,
    radius_multipliers: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0),
    domain_low: float = 0.0,
    domain_high: float = 1.0,
) -> tuple[float, dict]:
    """∫_domain E(z) dz via stencil centered on each kernel.

    Returns (estimate, diagnostics).
    """
    D = centers.shape[1]
    space = SamplingSpace(
        D=D, sigma=sigma, kernel=kernel,  # type: ignore[arg-type]
        radius_multipliers=radius_multipliers, n_directions=n_directions,
    )

    # (N, M, D)
    all_samples = space.samples_for_batch(centers)
    N = all_samples.shape[0]
    M = all_samples.shape[1]
    samples_flat = all_samples.reshape(-1, D)

    # In-domain mask
    in_domain = np.all(
        (samples_flat >= domain_low) & (samples_flat <= domain_high),
        axis=1,
    )

    # D(z) at each sample — need to sum kernel contributions from all N centers
    D_vals = raw_density(samples_flat, centers, weights, sigma, kernel=kernel)
    integrand = 1.0 / (1.0 + D_vals)          # the 1/(1+D) term
    integrand *= in_domain.astype(float)      # leakage handled here
    # Per-kernel weighting: integrand shape is (N*M,), stencil weights are (M,)
    integrand_mat = integrand.reshape(N, M)
    per_kernel_int = (integrand_mat * space.weights[None, :]).sum(axis=1)  # (N,)
    # ∫E dz = Σ_j w_j · integral_from_kernel_j
    total = float((weights * per_kernel_int).sum())
    diag = {
        "n_samples_total": samples_flat.shape[0],
        "frac_in_domain": float(in_domain.mean()),
        "per_kernel": per_kernel_int,
    }
    return total, diag


# ---------- Figure: stencil overlay on E(z) field ----------

def figure_stencil_overlay(sigma: float = 0.12, kernel: str = "gaussian"):
    apply_style()
    u, v, Z = grid_2d(res=201)
    centers = np.array([
        [0.30, 0.40], [0.65, 0.30], [0.80, 0.70], [0.35, 0.80],
    ])
    w = np.ones(len(centers))
    D_field = raw_density(Z, centers, w, sigma, kernel=kernel)
    E_field = E_of_D(D_field)

    space = SamplingSpace(D=2, sigma=sigma, kernel=kernel,  # type: ignore[arg-type]
                           n_directions=16)
    all_samples = space.samples_for_batch(centers)  # (N, M, D)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.contourf(u, v, E_field, levels=25, cmap=evidence_cmap(), vmin=0, vmax=1)
    ax.contour(u, v, E_field, levels=[0.25, 0.5, 0.75],
               colors=ZINC[400], linewidths=0.5, alpha=0.6)

    # stencil samples on top
    for k in range(all_samples.shape[0]):
        ax.scatter(
            all_samples[k, :, 0], all_samples[k, :, 1],
            c=ZINC[600], s=4, alpha=0.6, zorder=4,
        )

    # kernel centers
    ax.scatter(centers[:, 0], centers[:, 1],
               c=RED, marker="x", s=60, linewidths=1.5, zorder=6)

    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    strip_spines(ax)
    fig.colorbar(im, ax=ax, shrink=0.85, label="E(z)")

    total, diag = integrated_E_via_stencil(centers, w, sigma, kernel=kernel)
    ax.set_title(
        f"E(z) field + stencil probes\n"
        f"σ={sigma}, {kernel}, N={len(centers)} kernels, "
        f"∫E dz ≈ {total:.3f}, "
        f"samples in-domain: {100 * diag['frac_in_domain']:.0f}%",
        fontsize=10, color=ZINC[700],
    )

    fig.suptitle(
        "Stencil samples overlaid on the evidence field",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"05a_stencil_overlay_{kernel}")
    plt.close(fig)
    return path


# ---------- Figure: stencil vs reference MC across σ ----------

def figure_accuracy_across_sigma(kernel: str = "gaussian"):
    apply_style()
    centers = np.array([
        [0.30, 0.40], [0.65, 0.30], [0.80, 0.70], [0.35, 0.80],
    ])
    w = np.ones(len(centers))

    sigmas = np.logspace(-2, -0.3, 18)  # 0.01 to ~0.5

    # Reference: dense grid quadrature in [0,1]^D
    def reference(sigma):
        u = np.linspace(0, 1, 201)
        U, V = np.meshgrid(u, u)
        Z = np.stack([U, V], axis=-1)
        D = raw_density(Z, centers, w, sigma, kernel=kernel)
        E = E_of_D(D)
        return float(np.trapezoid(np.trapezoid(E, u, axis=0), u, axis=0))

    def sobol_mc(sigma, M=128, seed=0):
        """For comparison — the old uniform Sobol estimator."""
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True, rng=seed).random(n=M)
        D = raw_density(sobol, centers, w, sigma, kernel=kernel)
        E = E_of_D(D)
        return float(E.mean())

    stencil = np.array([integrated_E_via_stencil(centers, w, s, kernel=kernel)[0] for s in sigmas])
    truth = np.array([reference(s) for s in sigmas])
    sobol = np.array([sobol_mc(s) for s in sigmas])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(sigmas, truth, color=ZINC[700], lw=1.8, label="Reference (grid 201²)")
    ax.plot(sigmas, stencil, color=STEEL[500], lw=1.8, ls="--", label="Stencil (81 pts × N)")
    ax.plot(sigmas, sobol, color=RED, lw=1.3, ls=":", label="Sobol uniform (M=128)")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("∫_[0,1]² E(z) dz")
    ax.set_title("Estimator vs reference", pad=6)
    ax.legend()
    strip_spines(ax)

    ax = axes[1]
    rel_sten = 100 * (stencil - truth) / (np.abs(truth) + 1e-12)
    rel_sob = 100 * (sobol - truth) / (np.abs(truth) + 1e-12)
    ax.plot(sigmas, rel_sten, color=STEEL[500], lw=1.8, label="Stencil")
    ax.plot(sigmas, rel_sob, color=RED, lw=1.3, label="Sobol uniform")
    ax.axhline(0, color=ZINC[400], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Why this matters: Sobol fails at small σ", pad=6)
    ax.legend()
    strip_spines(ax)

    fig.suptitle(
        f"Estimator comparison — ∫E dz for 4 fixed kernels ({kernel})",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"05b_stencil_vs_sobol_{kernel}")
    plt.close(fig)
    return path


# ---------- Figure: three configurations side by side ----------

def figure_configurations(sigma: float = 0.12, kernel: str = "gaussian"):
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
        D = raw_density(Z, centers, w, sigma, kernel=kernel)
        E = E_of_D(D)
        total, _ = integrated_E_via_stencil(centers, w, sigma, kernel=kernel)

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
        f"∫E dz decides placement — stacked is wasteful, boundary leaks  ({kernel}, σ={sigma})",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"05c_configurations_{kernel}")
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
        total, diag = integrated_E_via_stencil(centers, w, sigma=0.12, kernel="gaussian")
        print(f"{name:<32s}  ∫E dz = {total:.4f}  "
              f"({100 * diag['frac_in_domain']:.0f}% samples in-domain)")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_stencil_overlay()
    p2 = figure_accuracy_across_sigma()
    p3 = figure_configurations()
    for p in (p1, p2, p3):
        print(f"Saved: {p}")
