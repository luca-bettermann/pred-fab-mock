"""Concept 08 — evidence transforms: mass-norm vs peak-norm.

Under the mass-normalised kernel (∫ρ = 1 in ℝ^D), one kernel in D=3 at σ=0.1
has **peak density ≈ 63.5**. The transform

    E_mass(D) = D / (1 + D)

then reads E ≈ 0.98 at the kernel center, and saturation spreads far from the
data — *vanishing gradient* for the optimiser.

Peak-normalised kernel: drop the `(σ√2π)^{-D}` factor, so every kernel has
peak = 1 at its own center. Equivalently, use a σ-dependent threshold in the
mass-norm transform:

    E_peak(D) = D / (peak + D),       peak = (σ√2π)^{-D}

With peak-norm: one datapoint → E = 0.5 at its centre, two coincident → 0.67,
three → 0.75, and so on. Calibrated to "how many experiments have I stacked
here", independent of σ.

This script visualises both transforms + their consequences:
    1. transform curves E(D)
    2. E(z) field for the same 5-kernel configuration
    3. acquisition landscape Δ∫E under each transform
    4. ∫E dz for spread vs stacked under each transform
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    evidence_cmap, save, strip_spines,
)
from scipy.stats import qmc

from concept_02_raw_density import raw_density, grid_2d
from concept_05_integrated_objective import integrated_E_sobol


# ---------- Helpers ----------

def gaussian_peak(sigma: float, D: int) -> float:
    """Mass-normalised Gaussian density at its own centre (peak value)."""
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) ** D


def E_mass(D_vals: np.ndarray) -> np.ndarray:
    """Original transform — saturates at D ≈ 1."""
    return D_vals / (1.0 + D_vals)


def E_peak(D_vals: np.ndarray, peak: float) -> np.ndarray:
    """Peak-threshold transform — E=0.5 at one kernel's centre, independent of σ."""
    return D_vals / (peak + D_vals)


# ---------- Core estimator under each transform ----------

def integrated_E_peak_sobol(
    centers: np.ndarray, weights: np.ndarray, sigma: float,
    n: int = 512, seed: int = 0,
) -> float:
    """∫_[0,1]^D D(z)/(peak + D(z)) dz via scrambled Sobol QMC."""
    D_dim = centers.shape[1]
    peak = gaussian_peak(sigma, D_dim)
    sobol = qmc.Sobol(d=D_dim, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, weights, sigma)
    return float((D_vals / (peak + D_vals)).mean())


# ---------- Figure 1: scalar transforms ----------

def figure_transform_curves():
    apply_style()
    D_vals = np.linspace(0, 100, 500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)

    # Left panel: E vs D for both transforms, σ=0.1 D=3 for peak value
    peak_010_3d = gaussian_peak(0.10, 3)
    peak_010_2d = gaussian_peak(0.10, 2)
    ax = axes[0]
    ax.plot(D_vals, E_mass(D_vals), color=STEEL[500], lw=2.0, label="E = D/(1+D)  (mass-norm)")
    ax.plot(D_vals, E_peak(D_vals, peak_010_3d), color=EMERALD[500], lw=2.0,
            label=f"E = D/(peak+D),  peak={peak_010_3d:.1f}  (σ=0.1, D=3)")
    ax.plot(D_vals, E_peak(D_vals, peak_010_2d), color=EMERALD[700], lw=1.5, ls="--",
            label=f"E = D/(peak+D),  peak={peak_010_2d:.1f}  (σ=0.1, D=2)")
    ax.axhline(0.5, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("raw density D")
    ax.set_ylabel("E")
    ax.set_title("Saturation threshold shifts with the transform", pad=6)
    ax.legend(loc="lower right", fontsize=7)
    strip_spines(ax)

    # Right panel: zoom on low-D to show E_mass saturates WAY faster
    ax = axes[1]
    D_zoom = np.linspace(0, 5, 500)
    ax.plot(D_zoom, E_mass(D_zoom), color=STEEL[500], lw=2.0, label="mass-norm")
    for peak, D_lbl, c in [(peak_010_3d, "σ=0.10 D=3", EMERALD[500])]:
        ax.plot(D_zoom, E_peak(D_zoom, peak), color=c, lw=2.0, label=f"peak-norm, {D_lbl}")
    # Annotate: typical D value at middle of 5 kernels
    ax.axvline(1.0, color=RED, lw=0.8, ls="--", alpha=0.5)
    ax.annotate("D=1\n(mass-norm saturation pivot)", (1.0, 0.5),
                xytext=(10, -5), textcoords="offset points",
                fontsize=7, color=ZINC[500])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("raw density D (zoom)")
    ax.set_ylabel("E")
    ax.set_title("Low-D zoom — mass-norm saturates far too fast", pad=6)
    ax.legend(loc="lower right", fontsize=7)
    strip_spines(ax)

    fig.suptitle(
        "Scalar transform curves — E(D) under mass-norm vs peak-norm",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "08a_transform_curves")
    plt.close(fig)
    return path


# ---------- Figure 2: E-field under both transforms ----------

def figure_field_comparison(sigma: float = 0.10):
    apply_style()
    u, v, Z = grid_2d(res=201)
    centers = np.array([
        [0.30, 0.30], [0.70, 0.30], [0.50, 0.50],
        [0.30, 0.70], [0.70, 0.70],
    ])
    w = np.ones(len(centers))
    D_field = raw_density(Z, centers, w, sigma)
    peak = gaussian_peak(sigma, 2)

    E_m = E_mass(D_field)
    E_p = E_peak(D_field, peak)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, field, title in [
        (axes[0], E_m, f"mass-norm  E = D/(1+D)\nE mostly ≥ 0.9 — flat landscape"),
        (axes[1], E_p, f"peak-norm  E = D/(peak+D), peak={peak:.1f}\nE reads 0.5 per isolated point"),
    ]:
        im = ax.contourf(u, v, field, levels=25, cmap=evidence_cmap(), vmin=0, vmax=1)
        ax.contour(u, v, field, levels=[0.25, 0.5, 0.75],
                   colors=ZINC[400], linewidths=0.5, alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1],
                   c="white", edgecolors=ZINC[900], s=45, linewidths=0.8, zorder=5)
        ax.set_title(title, pad=6, fontsize=10)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)
        fig.colorbar(im, ax=ax, shrink=0.85, label="E")

    fig.suptitle(
        f"Same 5 kernels, two transforms   |   σ = {sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "08b_field_comparison")
    plt.close(fig)
    return path


# ---------- Figure 3: acquisition landscape under each ----------

def figure_acquisition_comparison(sigma: float = 0.10):
    apply_style()
    existing = np.array([[0.30, 0.60], [0.70, 0.35], [0.50, 0.85]])
    e_w = np.ones(len(existing))
    peak = gaussian_peak(sigma, 2)

    res = 41
    x = np.linspace(0.02, 0.98, res)
    y = np.linspace(0.02, 0.98, res)

    I_m_old, _ = integrated_E_sobol(existing, e_w, sigma=sigma)
    I_p_old = integrated_E_peak_sobol(existing, e_w, sigma=sigma)

    dI_m = np.zeros((res, res))
    dI_p = np.zeros((res, res))
    for j, yv in enumerate(y):
        for i, xv in enumerate(x):
            z_new = np.array([xv, yv])
            c = np.vstack([existing, z_new[None, :]])
            w = np.concatenate([e_w, [1.0]])
            I_m_new, _ = integrated_E_sobol(c, w, sigma=sigma)
            I_p_new = integrated_E_peak_sobol(c, w, sigma=sigma)
            dI_m[j, i] = I_m_new - I_m_old
            dI_p[j, i] = I_p_new - I_p_old

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, dI, title in [
        (axes[0], dI_m, f"mass-norm Δ∫E   range: {dI_m.min():.4f} → {dI_m.max():.4f}"),
        (axes[1], dI_p, f"peak-norm Δ∫E   range: {dI_p.min():.4f} → {dI_p.max():.4f}"),
    ]:
        # Normalise each independently so the *shape* of the landscape is comparable
        im = ax.contourf(x, y, dI, levels=25, cmap="YlOrRd")
        ax.contour(x, y, dI, levels=10, colors=ZINC[400], linewidths=0.4, alpha=0.5)
        ax.scatter(existing[:, 0], existing[:, 1],
                   c="white", edgecolors=ZINC[900], s=45, linewidths=0.8, zorder=5)
        j_max, i_max = np.unravel_index(np.argmax(dI), dI.shape)
        ax.scatter(x[i_max], y[j_max], marker="x", c=YELLOW, s=100, linewidths=2.0, zorder=6)
        ax.set_title(title, pad=6, fontsize=9)
        ax.set_xlabel("z_new,1")
        ax.set_ylabel("z_new,2")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Δ∫E")

    fig.suptitle(
        f"Acquisition landscape under each transform   |   3 existing kernels, σ={sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "08c_acquisition_comparison")
    plt.close(fig)
    return path


# ---------- Figure 4: ∫E values per configuration ----------

def figure_ranking_table(sigma: float = 0.10):
    """Same 4 configurations (stacked/spread/corners/one-point) scored under each transform.

    Shows that peak-norm gives larger relative differences between configurations →
    stronger signal for the optimiser.
    """
    apply_style()
    configs = [
        ("1 center",            np.array([[0.5, 0.5]])),
        ("5 stacked at center", np.array([[0.5, 0.5]] * 5)),
        ("5 stacked at corner", np.array([[0.02, 0.02]] * 5)),
        ("5 spread interior",   np.array([
            [0.25, 0.30], [0.75, 0.25], [0.55, 0.55],
            [0.25, 0.75], [0.80, 0.80],
        ])),
        ("5 at cube corners",   np.array([
            [0.02, 0.02], [0.02, 0.98], [0.98, 0.02],
            [0.98, 0.98], [0.50, 0.50],
        ])),
    ]
    labels = [c[0] for c in configs]
    I_mass = [integrated_E_sobol(c[1], np.ones(len(c[1])), sigma=sigma)[0] for c in configs]
    I_peak = [integrated_E_peak_sobol(c[1], np.ones(len(c[1])), sigma=sigma) for c in configs]

    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    xs = np.arange(len(labels))
    # Separate scales — plot each transform on its own y-axis (relative view)
    ax.bar(xs - 0.2, I_mass, width=0.38, color=STEEL[500], label="mass-norm ∫E dz")
    ax2 = ax.twinx()
    ax2.bar(xs + 0.2, I_peak, width=0.38, color=EMERALD[500], label="peak-norm ∫E dz")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("∫E dz (mass-norm)", color=STEEL[700])
    ax2.set_ylabel("∫E dz (peak-norm)", color=EMERALD[700])
    ax.tick_params(axis="y", labelcolor=STEEL[700])
    ax2.tick_params(axis="y", labelcolor=EMERALD[700])
    ax.set_title("Same configurations, two transforms", pad=6)
    strip_spines(ax)
    ax2.spines["top"].set_visible(False)
    fig.suptitle(
        f"Ranking stays the same; absolute scale and contrast differ   |   σ = {sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "08d_ranking_comparison")
    plt.close(fig)
    return path, labels, I_mass, I_peak


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 70)
    print("∫E dz under both transforms  (σ=0.10)")
    print("=" * 70)
    peak_2d = gaussian_peak(0.10, 2)
    print(f"peak density at σ=0.10, D=2:  {peak_2d:.3f}")
    print(f"peak density at σ=0.10, D=3:  {gaussian_peak(0.10, 3):.3f}")
    print()
    print(f"{'config':<28s}{'mass-norm':>14s}{'peak-norm':>14s}")
    print("─" * 58)
    configs = [
        ("1 center point",      np.array([[0.5, 0.5]])),
        ("5 stacked at center", np.array([[0.5, 0.5]] * 5)),
        ("5 spread interior",   np.array([
            [0.25, 0.30], [0.75, 0.25], [0.55, 0.55],
            [0.25, 0.75], [0.80, 0.80],
        ])),
        ("5 at cube corners",   np.array([
            [0.02, 0.02], [0.02, 0.98], [0.98, 0.02],
            [0.98, 0.98], [0.50, 0.50],
        ])),
    ]
    for name, c in configs:
        w = np.ones(len(c))
        I_m = integrated_E_sobol(c, w, sigma=0.10)[0]
        I_p = integrated_E_peak_sobol(c, w, sigma=0.10)
        print(f"{name:<28s}{I_m:>14.4f}{I_p:>14.4f}")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_transform_curves()
    p2 = figure_field_comparison()
    p3 = figure_acquisition_comparison()
    p4, *_ = figure_ranking_table()
    for p in (p1, p2, p3, p4):
        print(f"Saved: {p}")
