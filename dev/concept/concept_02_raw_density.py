"""Concept 02 — the raw density D(z).

Given N datapoints at z_1, ..., z_N, the raw evidence density is the sum of
their kernels:

    D(z)  =  Σ_j  w_j · ρ_j(z)

Each point carries mass w_j (default 1). D(z) is unbounded. We use it as the
pre-transform quantity going into E(z) = D/(1+D).

This script visualises D(z) over [0,1]² for a few illustrative configurations,
and prints how D behaves at the kernel centers, at midpoints, and at the edges.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW,
    density_cmap, save, strip_spines,
)
from concept_01_kernels import gaussian_density, cauchy_density


# ---------- D(z) ----------

def raw_density(
    Z: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float,
    kernel: str = "gaussian",
) -> np.ndarray:
    """D(z) = Σ w_j · ρ_j(z). Z: (..., D), returns (...)."""
    kernel_fn = gaussian_density if kernel == "gaussian" else cauchy_density
    out = np.zeros(Z.shape[:-1])
    for z_j, w in zip(centers, weights):
        out += w * kernel_fn(Z, z_j, sigma)
    return out


# ---------- Grid evaluation ----------

def grid_2d(res: int = 201):
    u = np.linspace(0.0, 1.0, res)
    v = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(u, v)
    Z = np.stack([U, V], axis=-1)
    return u, v, Z


# ---------- Figure: three configurations ----------

def figure_three_configs(sigma: float = 0.10, kernel: str = "gaussian"):
    apply_style()
    u, v, Z = grid_2d()

    # Three configurations
    configs = [
        ("one point", np.array([[0.5, 0.5]])),
        ("two close (0.5·σ apart)", np.array([[0.5, 0.5 - 0.25 * sigma], [0.5, 0.5 + 0.25 * sigma]])),
        ("two far (4·σ apart)", np.array([[0.35, 0.5], [0.65, 0.5]])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    vmax = 0.0
    D_fields = []
    for name, centers in configs:
        w = np.ones(len(centers))
        D = raw_density(Z, centers, w, sigma, kernel=kernel)
        vmax = max(vmax, D.max())
        D_fields.append((name, centers, D))

    for ax, (name, centers, D) in zip(axes, D_fields):
        im = ax.contourf(u, v, D, levels=30, cmap=density_cmap(), vmin=0, vmax=vmax)
        ax.contour(u, v, D, levels=10, colors=ZINC[400], linewidths=0.4, alpha=0.5)
        ax.scatter(centers[:, 0], centers[:, 1],
                   c="white", edgecolors=ZINC[700], s=40, linewidths=0.8, zorder=5)
        ax.set_title(name, pad=6)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)
        fig.colorbar(im, ax=ax, shrink=0.85, label="D(z)")

    fig.suptitle(
        f"Raw density D(z) = Σ_j w_j · ρ_j(z)   |   {kernel}, σ = {sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"02a_raw_density_{kernel}_configs")
    plt.close(fig)
    return path


# ---------- Figure: five spread kernels ----------

def figure_five_spread(sigma: float = 0.12, kernel: str = "gaussian"):
    apply_style()
    u, v, Z = grid_2d()
    centers = np.array([
        [0.20, 0.25], [0.75, 0.20], [0.50, 0.50],
        [0.25, 0.80], [0.80, 0.75],
    ])
    w = np.ones(len(centers))
    D = raw_density(Z, centers, w, sigma, kernel=kernel)

    fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
    im = ax.contourf(u, v, D, levels=30, cmap=density_cmap())
    ax.contour(u, v, D, levels=10, colors=ZINC[400], linewidths=0.4, alpha=0.5)
    ax.scatter(centers[:, 0], centers[:, 1],
               c="white", edgecolors=ZINC[700], s=45, linewidths=0.8, zorder=5)
    for i, (x, y) in enumerate(centers):
        ax.annotate(
            f"z_{i}", (x, y), xytext=(6, 6), textcoords="offset points",
            color=ZINC[400], fontsize=8,
        )
    ax.set_title(f"five spread kernels   |   {kernel}, σ = {sigma}", pad=6)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    strip_spines(ax)
    fig.colorbar(im, ax=ax, shrink=0.85, label="D(z)")

    fig.suptitle(
        "Raw density for five well-spread points",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"02b_raw_density_{kernel}_five_spread")
    plt.close(fig)
    return path


# ---------- Terminal print: magnitudes ----------

def print_magnitudes(sigma: float = 0.10, D_dim: int = 2):
    """Show D(z) at representative query points for various configurations."""
    print()
    print("=" * 70)
    print(f"D(z) magnitudes    (Gaussian kernel, σ = {sigma}, D = {D_dim})")
    print("=" * 70)

    def D_at(z, centers, weights):
        return float(raw_density(z[None, :], centers, weights, sigma, "gaussian")[0])

    # Peak density of one kernel at z_j
    peak = 1.0 / (sigma * np.sqrt(2 * np.pi)) ** D_dim
    print(f"single-kernel peak ρ(z_j)            = {peak:.3f}")

    # Two coincident kernels
    centers2 = np.tile(np.array([[0.5] * D_dim]), (2, 1))
    w2 = np.array([1.0, 1.0])
    print(f"D at two stacked points              = {D_at(np.full(D_dim, 0.5), centers2, w2):.3f}")

    # Five spread kernels (1 D = project), measure D at a random midpoint
    np.random.seed(0)
    centers5 = np.random.uniform(0.15, 0.85, (5, D_dim))
    w5 = np.ones(5)
    print(f"D at one of five spread z_j          = {D_at(centers5[0], centers5, w5):.3f}")
    print(f"D at the centroid of five spread     = {D_at(centers5.mean(axis=0), centers5, w5):.3f}")
    print(f"D at a corner (0.01, 0.01) when all far = "
          f"{D_at(np.full(D_dim, 0.01), centers5, w5):.3f}")
    print()


if __name__ == "__main__":
    print_magnitudes()
    p1 = figure_three_configs(sigma=0.10, kernel="gaussian")
    p2 = figure_five_spread(sigma=0.12, kernel="gaussian")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
