"""Concept 10 — estimator comparison at matched sample budget.

Three estimators for ∫_[0,1]^D E(z) dz  (E = D/(1+D), Gaussian kernels):

    KernelField  — deterministic spherical-shell probes around each kernel
    Importance   — random samples drawn from each kernel's density, fixed seed
    Sobol        — quasi-random uniform samples in the domain

All compared at **equal sample count** to isolate the *quality* of where the
samples live, not how many of them there are. σ is swept to show the
low-σ regime where Sobol fails. D is swept to show scaling.

Reference: dense grid quadrature (201² in 2-D, 51³ in 3-D).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    save, strip_spines,
)
from kernel_field import KernelField
from concept_02_raw_density import raw_density


# ---------- Estimators ----------

def estimate_field(centers, weights, sigma, n_total, D_dim=2):
    """Pick n_directions so total probes ≈ n_total.

    Points per kernel ≈ n_total / N.   With 5 radial shells plus the center,
    per-kernel = 1 + n_dirs·5 ≈ 5·n_dirs. Solve n_dirs ≈ (per_kernel-1)/5.
    """
    N = len(centers)
    per_kernel = max(2, n_total // N)
    n_dirs = max(2, (per_kernel - 1) // 5)
    field = KernelField(D=D_dim, sigma=sigma, n_directions=n_dirs)

    probes = field.probes_for_batch(centers)
    flat = probes.reshape(-1, D_dim)
    in_domain = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    D_vals = raw_density(flat, centers, weights, sigma)
    integrand = 1.0 / (1.0 + D_vals) * in_domain.astype(float)
    M = probes.shape[1]
    per_k = (integrand.reshape(N, M) * field.weights[None, :]).sum(axis=1)
    est = float((weights * per_k).sum())
    actual_points = N * M
    return est, actual_points


def estimate_importance(centers, weights, sigma, n_total, D_dim=2, seed=0):
    """Random samples from each kernel's Gaussian density (fixed seed).

    ∫ ρ_j · 1/(1+D) dz = E_{z~ρ_j}[ 1/(1+D(z)) ] · [assumes z stays in ℝ^D]
    """
    N = len(centers)
    per_kernel = max(2, n_total // N)
    rng = np.random.default_rng(seed)
    # Per-kernel samples drawn from that kernel's Gaussian
    eps = rng.standard_normal((N, per_kernel, D_dim)) * sigma
    samples = centers[:, None, :] + eps
    flat = samples.reshape(-1, D_dim)
    in_domain = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    D_vals = raw_density(flat, centers, weights, sigma)
    integrand = np.where(in_domain, 1.0 / (1.0 + D_vals), 0.0)
    per_k = integrand.reshape(N, per_kernel).mean(axis=1)
    est = float((weights * per_k).sum())
    return est, N * per_kernel


def estimate_sobol(centers, weights, sigma, n_total, D_dim=2, seed=0):
    """Uniform quasi-MC over [0,1]^D; total sample count = n_total."""
    # Next power of 2 ≥ n_total for Sobol's discrepancy properties
    n = 2 ** int(np.ceil(np.log2(max(n_total, 2))))
    sobol = qmc.Sobol(d=D_dim, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, weights, sigma)
    E_vals = D_vals / (1.0 + D_vals)
    est = float(E_vals.mean())
    return est, n


def reference_grid(centers, weights, sigma, D_dim=2, res=201):
    if D_dim == 2:
        u = np.linspace(0, 1, res)
        U, V = np.meshgrid(u, u)
        Z = np.stack([U, V], axis=-1)
        D_vals = raw_density(Z, centers, weights, sigma)
        E = D_vals / (1.0 + D_vals)
        return float(np.trapezoid(np.trapezoid(E, u, axis=0), u, axis=0))
    # D=3
    res3 = max(31, res // 4)
    u = np.linspace(0, 1, res3)
    U, V, W = np.meshgrid(u, u, u, indexing="ij")
    Z = np.stack([U, V, W], axis=-1)
    D_vals = raw_density(Z.reshape(-1, 3), centers, weights, sigma).reshape(res3, res3, res3)
    E = D_vals / (1.0 + D_vals)
    I = np.trapezoid(np.trapezoid(np.trapezoid(E, u, axis=0), u, axis=0), u, axis=0)
    return float(I)


# ---------- Figure: σ sweep at fixed budget ----------

def figure_sigma_sweep(D_dim: int = 2, n_total: int = 300):
    apply_style()
    # Fixed kernel config
    if D_dim == 2:
        centers = np.array([
            [0.30, 0.40], [0.65, 0.30], [0.80, 0.70], [0.35, 0.80],
        ])
    else:
        centers = np.array([
            [0.30, 0.40, 0.30], [0.65, 0.30, 0.70],
            [0.80, 0.70, 0.40], [0.35, 0.80, 0.60],
        ])
    weights = np.ones(len(centers))

    sigmas = np.logspace(-2, -0.3, 20)
    truth = np.array([reference_grid(centers, weights, s, D_dim=D_dim) for s in sigmas])

    # Track actual-point counts per estimator
    sten = np.array([estimate_field(centers, weights, s, n_total, D_dim=D_dim)[0] for s in sigmas])
    imps = np.array([estimate_importance(centers, weights, s, n_total, D_dim=D_dim)[0] for s in sigmas])
    sobs = np.array([estimate_sobol(centers, weights, s, n_total, D_dim=D_dim)[0] for s in sigmas])

    n_sten = estimate_field(centers, weights, 0.1, n_total, D_dim=D_dim)[1]
    n_imps = estimate_importance(centers, weights, 0.1, n_total, D_dim=D_dim)[1]
    n_sobs = estimate_sobol(centers, weights, 0.1, n_total, D_dim=D_dim)[1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    ax = axes[0]
    ax.plot(sigmas, truth, color=ZINC[900], lw=1.8, label="Reference (grid)")
    ax.plot(sigmas, sten, color=STEEL[500], lw=1.6, ls="--",
            label=f"KernelField ({n_sten} pts)")
    ax.plot(sigmas, imps, color=EMERALD[500], lw=1.4, ls=":",
            label=f"Importance sampling ({n_imps} pts)")
    ax.plot(sigmas, sobs, color=RED, lw=1.4, ls=":",
            label=f"Sobol uniform ({n_sobs} pts)")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("∫E dz")
    ax.set_title(f"Estimator value vs reference  (D={D_dim})", pad=6)
    ax.legend(fontsize=7)
    strip_spines(ax)

    ax = axes[1]
    for est, color, label in [
        (sten, STEEL[500], "KernelField"),
        (imps, EMERALD[500], "Importance"),
        (sobs, RED, "Sobol uniform"),
    ]:
        rel = 100 * (est - truth) / (np.abs(truth) + 1e-12)
        ax.plot(sigmas, rel, color=color, lw=1.6, label=label)
    ax.axhline(0, color=ZINC[400], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("relative error [%]")
    ax.set_ylim(-100, 150)
    ax.set_title("Relative error — matched budget", pad=6)
    ax.legend(fontsize=7)
    strip_spines(ax)

    fig.suptitle(
        f"Three estimators, same sample budget ≈ {n_total}, D = {D_dim}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"10a_sigma_sweep_D{D_dim}")
    plt.close(fig)
    return path


# ---------- Figure: D-scaling at fixed σ ----------

def figure_d_scaling(n_total: int = 300, sigma: float = 0.10):
    apply_style()
    Ds = [2, 3, 4, 5]
    rng = np.random.default_rng(0)
    results = {"KernelField": [], "Importance": [], "Sobol": [], "Reference": []}
    errors = {"KernelField": [], "Importance": [], "Sobol": []}

    for D_dim in Ds:
        centers = rng.uniform(0.15, 0.85, size=(4, D_dim))
        weights = np.ones(len(centers))
        # Reference via dense grid (coarser at higher D for speed)
        if D_dim <= 3:
            truth = reference_grid(centers, weights, sigma, D_dim=D_dim, res=81)
        else:
            # Fall back to dense MC for D>3 as reference
            mc = rng.standard_normal((500_000, D_dim)) * 0 + rng.random((500_000, D_dim))
            D_vals = raw_density(mc, centers, weights, sigma)
            E_vals = D_vals / (1.0 + D_vals)
            truth = float(E_vals.mean())

        s, _ = estimate_field(centers, weights, sigma, n_total, D_dim=D_dim)
        i, _ = estimate_importance(centers, weights, sigma, n_total, D_dim=D_dim)
        b, _ = estimate_sobol(centers, weights, sigma, n_total, D_dim=D_dim)
        results["KernelField"].append(s)
        results["Importance"].append(i)
        results["Sobol"].append(b)
        results["Reference"].append(truth)
        for k, v in [("KernelField", s), ("Importance", i), ("Sobol", b)]:
            errors[k].append(100 * (v - truth) / (abs(truth) + 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    ax = axes[0]
    ax.plot(Ds, results["Reference"], color=ZINC[900], lw=1.8, marker="o", label="Reference")
    for k, c in [("KernelField", STEEL[500]), ("Importance", EMERALD[500]), ("Sobol", RED)]:
        ax.plot(Ds, results[k], color=c, lw=1.6, marker="s", ls="--", label=k)
    ax.set_xticks(Ds)
    ax.set_xlabel("D (dimensions)")
    ax.set_ylabel("∫E dz")
    ax.set_title(f"Integral value vs D  (σ={sigma}, budget≈{n_total})", pad=6)
    ax.legend(fontsize=7)
    strip_spines(ax)

    ax = axes[1]
    for k, c in [("KernelField", STEEL[500]), ("Importance", EMERALD[500]), ("Sobol", RED)]:
        ax.plot(Ds, errors[k], color=c, lw=1.6, marker="s", label=k)
    ax.axhline(0, color=ZINC[400], lw=0.8, ls="--")
    ax.set_xticks(Ds)
    ax.set_xlabel("D (dimensions)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Error vs D", pad=6)
    ax.legend(fontsize=7)
    strip_spines(ax)

    fig.suptitle(
        "Scaling with D — does each estimator hold up in higher dimensions?",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "10b_d_scaling")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 70)
    print("Estimator comparison at matched budget  (∫E dz, 4 kernels, D=2, σ=0.05)")
    print("=" * 70)
    centers = np.array([
        [0.30, 0.40], [0.65, 0.30], [0.80, 0.70], [0.35, 0.80],
    ])
    weights = np.ones(len(centers))
    truth = reference_grid(centers, weights, 0.05, D_dim=2)
    print(f"Reference (grid 201²):       {truth:.5f}")
    print()
    for n_total in [60, 150, 400]:
        s, n_s = estimate_field(centers, weights, 0.05, n_total, D_dim=2)
        i, n_i = estimate_importance(centers, weights, 0.05, n_total, D_dim=2)
        b, n_b = estimate_sobol(centers, weights, 0.05, n_total, D_dim=2)
        print(f"budget≈{n_total:>4d}:  "
              f"KernelField={s:.4f} ({n_s} pts),  "
              f"Importance={i:.4f} ({n_i} pts),  "
              f"Sobol={b:.4f} ({n_b} pts)")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_sigma_sweep(D_dim=2, n_total=300)
    p2 = figure_sigma_sweep(D_dim=3, n_total=300)
    p3 = figure_d_scaling(n_total=300, sigma=0.10)
    for p in (p1, p2, p3):
        print(f"Saved: {p}")
