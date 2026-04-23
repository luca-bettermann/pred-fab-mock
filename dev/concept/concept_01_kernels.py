"""Concept 01 — the kernel ρ(z; z_j, σ).

Each datapoint carries a probability density centered at its location.
Total mass in ℝ^D equals 1. Two options under consideration:

    Gaussian (product):  ρ(z|z_j) = ∏_d  (1/(σ√2π)) · exp(-(z_d - z_j,d)²/(2σ²))
    Cauchy  (product):   ρ(z|z_j) = ∏_d  (σ/π) / ((z_d - z_j,d)² + σ²)

Both integrate to 1 over ℝ^D. They differ in **peak height** (how concentrated
evidence is at the datapoint) and **tail decay** (how far evidence reaches).

This script prints and plots both kernels across representative σ values, and
verifies mass conservation numerically. Output:
    - Figure: 1D shape comparison at three σ values
    - Figure: 2D contour comparison at the default σ
    - Terminal: peak height, half-height radius, ∫ρ over ℝ and over [0,1]
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW,
    save, strip_spines,
)


# ---------- Kernel math ----------

def gaussian_density(z: np.ndarray, z_j: np.ndarray, sigma: float) -> np.ndarray:
    """Product Gaussian density, integrates to 1 over ℝ^D.

    z:   (..., D) query points
    z_j: (D,) center
    """
    D = z.shape[-1]
    diff = z - z_j
    d2 = np.sum(diff ** 2, axis=-1)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D
    return norm * np.exp(-d2 / (2.0 * sigma ** 2))


def cauchy_density(z: np.ndarray, z_j: np.ndarray, sigma: float) -> np.ndarray:
    """Product 1-D Cauchy density, integrates to 1 over ℝ^D.

    ρ(z|z_j) = ∏_d (σ/π) · 1 / ((z_d - z_j,d)² + σ²)
    """
    diff = z - z_j
    # (..., D) per-dim density
    per_dim = (sigma / np.pi) / (diff ** 2 + sigma ** 2)
    return np.prod(per_dim, axis=-1)


# ---------- Analytical quantities ----------

def gaussian_peak(sigma: float, D: int) -> float:
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D


def cauchy_peak(sigma: float, D: int) -> float:
    return (1.0 / (sigma * np.pi)) ** D


def gaussian_half_height_radius(sigma: float) -> float:
    """Where ρ(r) = peak/2 for 1-D (or equivalently, all-axis-equal)."""
    # exp(-r²/2σ²) = 0.5 → r² = 2σ² ln 2
    return sigma * np.sqrt(2.0 * np.log(2.0))


def cauchy_half_height_radius(sigma: float) -> float:
    """1-D Cauchy half-height: (σ/π)/(r² + σ²) = peak/2 → r = σ."""
    return sigma


# ---------- Figure 1: 1-D shape comparison ----------

def figure_1d_shapes():
    apply_style()
    sigmas = [0.03, 0.10, 0.30]
    x = np.linspace(-1.0, 1.0, 2001)
    z_j = np.array([0.0])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for ax, sigma in zip(axes, sigmas):
        g = np.array([gaussian_density(np.array([xi]), z_j, sigma) for xi in x])
        c = np.array([cauchy_density(np.array([xi]), z_j, sigma) for xi in x])
        ax.plot(x, g, color=STEEL[500], label="Gaussian", lw=1.8)
        ax.plot(x, c, color=EMERALD[500], label="Cauchy", lw=1.8)
        ax.axvline(0, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"σ = {sigma}", pad=6)
        ax.set_xlabel("z − z_j")
        ax.set_ylabel("ρ(z)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)
        strip_spines(ax)

    axes[-1].legend(loc="upper right")
    fig.suptitle(
        "Kernel shape — Gaussian and Cauchy, 1-D, normalised to total mass 1",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "01a_kernel_shape_1d")
    plt.close(fig)
    return path


# ---------- Figure 2: 2-D contour comparison ----------

def figure_2d_contours(sigma: float = 0.1):
    apply_style()
    lim = 0.5
    res = 201
    x = np.linspace(-lim, lim, res)
    y = np.linspace(-lim, lim, res)
    X, Y = np.meshgrid(x, y)
    Z = np.stack([X, Y], axis=-1)  # (res, res, 2)
    z_j = np.array([0.0, 0.0])

    G = gaussian_density(Z, z_j, sigma)
    C = cauchy_density(Z, z_j, sigma)
    vmax = max(G.max(), C.max())

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)
    for ax, field, label, cmap_color in zip(
        axes, [G, C], ["Gaussian", "Cauchy"], [STEEL[500], EMERALD[500]]
    ):
        im = ax.contourf(x, y, field, levels=20,
                         cmap="Blues" if label == "Gaussian" else "Greens",
                         vmin=0, vmax=vmax)
        ax.contour(x, y, field, levels=6,
                   colors=ZINC[300], linewidths=0.6, alpha=0.6)
        ax.set_title(f"{label}, σ = {sigma}", pad=6)
        ax.set_xlabel("z₁ − z_j,1")
        ax.set_ylabel("z₂ − z_j,2")
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        strip_spines(ax)
        fig.colorbar(im, ax=ax, shrink=0.85, label="ρ")

    fig.suptitle(
        "2-D density cross-section — same σ, different tails",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "01b_kernel_shape_2d")
    plt.close(fig)
    return path


# ---------- Figure 3: mass diagnostic across σ ----------

def figure_mass_vs_sigma():
    """Verify each kernel's mass in ℝ^D ≈ 1 and show how much leaks outside [−1, 1]^D."""
    apply_style()
    sigmas = np.logspace(-2, 0, 30)  # 0.01 to 1.0
    D = 2
    # Numerical integration on a wide box: [-5σ_max, 5σ_max] for ℝ²
    # and over a unit box [−0.5, 0.5]² for "in-domain"
    wide = 4.0  # ±4 for finite integration
    res = 401

    def integrate(kernel_fn, sigma, half_extent):
        u = np.linspace(-half_extent, half_extent, res)
        X, Y = np.meshgrid(u, u)
        ZZ = np.stack([X, Y], axis=-1)
        vals = kernel_fn(ZZ, np.zeros(2), sigma)
        # trapezoid
        mass = np.trapezoid(np.trapezoid(vals, u, axis=0), u, axis=0)
        return float(mass)

    gauss_full = [integrate(gaussian_density, s, wide) for s in sigmas]
    cauchy_full = [integrate(cauchy_density, s, wide) for s in sigmas]
    gauss_unit = [integrate(gaussian_density, s, 0.5) for s in sigmas]
    cauchy_unit = [integrate(cauchy_density, s, 0.5) for s in sigmas]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(sigmas, gauss_full, color=STEEL[500], label="Gaussian, full ℝ²", lw=1.8)
    ax.plot(sigmas, cauchy_full, color=EMERALD[500], label="Cauchy, full ℝ²", lw=1.8)
    ax.axhline(1.0, color=ZINC[300], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("∫ρ dz")
    ax.set_title("Total mass in ℝ²  (numerical check → 1)", pad=6)
    ax.set_ylim(0.9, 1.05)
    ax.legend()
    strip_spines(ax)

    ax = axes[1]
    ax.plot(sigmas, gauss_unit, color=STEEL[500], label="Gaussian, [−½,½]²", lw=1.8)
    ax.plot(sigmas, cauchy_unit, color=EMERALD[500], label="Cauchy, [−½,½]²", lw=1.8)
    ax.axhline(1.0, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("∫_[−½,½]² ρ dz  =  α(z_j = 0)")
    ax.set_title("In-domain mass α(z_j)  (leakage = 1 − α)", pad=6)
    ax.legend()
    strip_spines(ax)

    fig.suptitle(
        "Mass conservation — Gaussian decays exponentially, Cauchy keeps heavier tails",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "01c_kernel_mass_vs_sigma")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 70)
    print("Kernel shape summary (centered, D dims)")
    print("=" * 70)
    print(f"{'σ':>6}  {'D':>2}  {'peak_G':>12}  {'peak_C':>12}  {'half_G':>8}  {'half_C':>8}")
    print("─" * 70)
    for sigma in [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for D in [2, 3]:
            print(
                f"{sigma:>6.2f}  {D:>2}  "
                f"{gaussian_peak(sigma, D):>12.2f}  "
                f"{cauchy_peak(sigma, D):>12.2f}  "
                f"{gaussian_half_height_radius(sigma):>8.3f}  "
                f"{cauchy_half_height_radius(sigma):>8.3f}"
            )
    print()


# ---------- Main ----------

if __name__ == "__main__":
    print_summary()
    p1 = figure_1d_shapes()
    p2 = figure_2d_contours(sigma=0.10)
    p3 = figure_mass_vs_sigma()
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
