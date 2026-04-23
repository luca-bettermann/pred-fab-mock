"""Concept 04 — the SamplingSpace stencil.

A deterministic, kernel-anchored set of query points. The same offsets go
around every datapoint, so we compute them once and reuse everywhere. Each
offset carries a weight derived from a radial-shell quadrature of the
Gaussian/Cauchy density.

This script visualises:
    1. the stencil geometry in 2-D (circle) and 3-D (sphere)
    2. where stencil points land relative to the kernel
    3. the weight distribution (size of dots = weight)
    4. integration accuracy: stencil-vs-true on a known integrand
    5. mass conservation at different kernel widths
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    density_cmap, save, strip_spines,
)
from sampling_space import SamplingSpace
from concept_01_kernels import gaussian_density


# ---------- Figure 1: stencil in 2-D with kernel contours behind it ----------

def figure_stencil_2d(sigma: float = 0.15, n_directions: int = 16):
    apply_style()
    space = SamplingSpace(D=2, sigma=sigma, kernel="gaussian", n_directions=n_directions)
    center = np.array([0.5, 0.5])
    samples = space.samples_at(center)

    # Background: the kernel we're integrating against
    u = np.linspace(0, 1, 301)
    U, V = np.meshgrid(u, u)
    Z = np.stack([U, V], axis=-1)
    K = gaussian_density(Z, center, sigma)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)

    for ax, color_by, title in [
        (axes[0], "uniform", "Stencil geometry (σ·{0.1, 0.2, 0.5, 1.0, 2.0} radii)"),
        (axes[1], "weight",  "Weighted by radial-shell quadrature (marker size = weight)"),
    ]:
        # kernel contours behind
        ax.contour(u, u, K, levels=8, colors=ZINC[300], linewidths=0.6, alpha=0.7)

        # shell radii as dashed circles
        for r_mult, r in zip(space.radius_multipliers, space.radii):
            circle = plt.Circle(center, r, fill=False, ls="--",
                                color=ZINC[300], lw=0.7, alpha=0.6)
            ax.add_patch(circle)
            ax.annotate(
                f"{r_mult}σ",
                (center[0] + r / np.sqrt(2), center[1] + r / np.sqrt(2)),
                fontsize=7, color=ZINC[400],
                xytext=(2, 2), textcoords="offset points",
            )

        # stencil points
        if color_by == "uniform":
            ax.scatter(samples[:, 0], samples[:, 1],
                       c=STEEL[500], s=25, edgecolors=ZINC[900], linewidths=0.5, zorder=5)
        else:
            # marker size ∝ weight (normalised for visibility)
            max_w = space.weights.max()
            sizes = 40 + 450 * (space.weights / max_w)
            ax.scatter(samples[:, 0], samples[:, 1],
                       c=STEEL[500], s=sizes, edgecolors=ZINC[900],
                       linewidths=0.5, alpha=0.85, zorder=5)

        ax.scatter([center[0]], [center[1]], marker="x",
                   c=RED, s=60, linewidths=1.5, zorder=6)
        ax.set_title(title, pad=6)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)

    fig.suptitle(
        f"SamplingSpace — deterministic stencil, σ = {sigma}, "
        f"{space.n_samples} points total",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "04a_stencil_2d")
    plt.close(fig)
    return path


# ---------- Figure 2: stencil in 3-D ----------

def figure_stencil_3d(sigma: float = 0.15, n_directions: int = 16):
    apply_style()
    space = SamplingSpace(D=3, sigma=sigma, kernel="gaussian", n_directions=n_directions)
    center = np.array([0.5, 0.5, 0.5])
    samples = space.samples_at(center)

    fig = plt.figure(figsize=(6.5, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Color-code by shell index
    # First sample is center; next n_directions are shell 0; etc.
    colors = [RED] + sum(
        [[STEEL[100 + 200 * k]] * n_directions for k in [1, 2, 3, 4, 4]], []
    )
    max_w = space.weights.max()
    sizes = 25 + 250 * (space.weights / max_w)

    for i, (pt, c, sz) in enumerate(zip(samples, colors, sizes)):
        ax.scatter(pt[0], pt[1], pt[2], c=c, s=sz, edgecolors=ZINC[900],
                   linewidths=0.4, alpha=0.85)

    # light σ·1 sphere for reference
    u_ang = np.linspace(0, 2 * np.pi, 30)
    v_ang = np.linspace(0, np.pi, 15)
    xs = center[0] + sigma * np.outer(np.cos(u_ang), np.sin(v_ang))
    ys = center[1] + sigma * np.outer(np.sin(u_ang), np.sin(v_ang))
    zs = center[2] + sigma * np.outer(np.ones_like(u_ang), np.cos(v_ang))
    ax.plot_wireframe(xs, ys, zs, color=ZINC[300], lw=0.3, alpha=0.3)

    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_zlabel("z₃")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(
        f"3-D stencil, σ = {sigma}, {space.n_samples} points\n"
        f"(wireframe: unit σ-sphere; red ×: kernel center)",
        fontsize=10, color=ZINC[700],
    )
    path = save(fig, "04b_stencil_3d")
    plt.close(fig)
    return path


# ---------- Figure 3: integration accuracy ----------

def figure_integration_accuracy():
    """Compare stencil's estimate of ∫f·ρ dz against a dense reference.

    Use f(z) = a + b·z₀ + c·z₀² (polynomial) so we have an analytical reference
    and can cleanly see stencil vs truth. Test across σ values.
    """
    apply_style()

    # The integrand f is just a gentle polynomial; I ≡ E_{ρ}[f(z)]
    def f(Z: np.ndarray) -> np.ndarray:
        # ℝ² polynomial for illustration
        x = Z[..., 0]
        y = Z[..., 1]
        return 0.5 + 0.8 * x - 0.3 * x ** 2 + 0.4 * y

    # For isotropic Gaussian centered at z_j with total mass 1, E[f] has a
    # known closed form for polynomials. Use Monte Carlo with M=1e6 as the
    # "truth" reference — simple and works for any f.
    def true_I(sigma, z_j, n_mc=2_000_000):
        rng = np.random.default_rng(42)
        eps = rng.standard_normal((n_mc, 2)) * sigma
        return float(f(z_j + eps).mean())

    def stencil_I(sigma, z_j, n_dir=16):
        space = SamplingSpace(D=2, sigma=sigma, kernel="gaussian", n_directions=n_dir)
        samples = space.samples_at(z_j)
        vals = f(samples)
        # ∫ f·ρ = Σ w_i · f(z_i). Weights already include density → no /ρ.
        return float(np.sum(space.weights * vals))

    z_j = np.array([0.5, 0.5])
    sigmas = np.logspace(-2, 0, 25)  # 0.01 to 1.0
    truth = np.array([true_I(s, z_j) for s in sigmas])
    est = np.array([stencil_I(s, z_j) for s in sigmas])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(sigmas, truth, color=ZINC[700], lw=1.8, label="Reference (MC 2·10⁶)")
    ax.plot(sigmas, est, color=STEEL[500], lw=1.8, ls="--", label="Stencil")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("∫ f(z) · ρ(z) dz")
    ax.set_title("Integration value vs σ", pad=6)
    ax.legend()
    strip_spines(ax)

    ax = axes[1]
    rel_err = (est - truth) / (np.abs(truth) + 1e-12)
    ax.plot(sigmas, 100 * rel_err, color=RED, lw=1.5)
    ax.axhline(0, color=ZINC[400], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Stencil error — scale-invariant by design", pad=6)
    strip_spines(ax)

    fig.suptitle(
        "SamplingSpace integration accuracy — polynomial integrand f · Gaussian ρ",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "04c_stencil_accuracy")
    plt.close(fig)
    return path


# ---------- Figure 4: mass conservation across D ----------

def figure_mass_vs_kernel():
    """The stencil's approximation of ∫ρ dz (should be 1) — shows the
    Gaussian/Cauchy asymmetry we observed earlier.
    """
    apply_style()
    sigmas = np.logspace(-2.5, 0, 25)
    n_dirs = 16

    mass_G_2d = [SamplingSpace(D=2, sigma=s, kernel="gaussian", n_directions=n_dirs).check_mass() for s in sigmas]
    mass_G_3d = [SamplingSpace(D=3, sigma=s, kernel="gaussian", n_directions=n_dirs).check_mass() for s in sigmas]
    mass_C_2d = [SamplingSpace(D=2, sigma=s, kernel="cauchy",   n_directions=n_dirs).check_mass() for s in sigmas]
    mass_C_3d = [SamplingSpace(D=3, sigma=s, kernel="cauchy",   n_directions=n_dirs).check_mass() for s in sigmas]

    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    ax.plot(sigmas, mass_G_2d, color=STEEL[500], lw=1.8, label="Gaussian, D=2")
    ax.plot(sigmas, mass_G_3d, color=STEEL[700], lw=1.8, ls="--", label="Gaussian, D=3")
    ax.plot(sigmas, mass_C_2d, color=EMERALD[500], lw=1.8, label="Cauchy, D=2")
    ax.plot(sigmas, mass_C_3d, color=EMERALD[700], lw=1.8, ls="--", label="Cauchy, D=3")
    ax.axhline(1.0, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("Σ stencil weights  ≈  ∫ρ dz")
    ax.set_title(
        "Stencil's estimate of kernel mass\n"
        "Gaussian: ~0.87-0.95 (2σ cutoff misses the tail)\n"
        "Cauchy: heavy-tailed → stencil misses most of the mass",
        pad=6,
    )
    ax.legend(loc="lower left")
    strip_spines(ax)

    fig.suptitle(
        "Mass conservation of the stencil — why Gaussian is more stencil-friendly",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "04d_stencil_mass")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 70)
    print("SamplingSpace summary")
    print("=" * 70)
    for D_dim in [2, 3]:
        for sigma in [0.01, 0.05, 0.10, 0.15, 0.30]:
            for kernel in ["gaussian", "cauchy"]:
                s = SamplingSpace(D=D_dim, sigma=sigma, kernel=kernel, n_directions=16)
                print(
                    f"D={D_dim}  σ={sigma:>5.2f}  {kernel:>8s}  "
                    f"n_samples={s.n_samples:>4d}  ∫ρ≈{s.check_mass():.4f}"
                )
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_stencil_2d()
    p2 = figure_stencil_3d()
    p3 = figure_integration_accuracy()
    p4 = figure_mass_vs_kernel()
    for p in (p1, p2, p3, p4):
        print(f"Saved: {p}")
