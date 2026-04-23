"""Concept 03 — the saturating transform  E(z) = D(z) / (1 + D(z)).

D(z) is unbounded; E(z) is in [0, 1). This transform encodes "you can never
be 100% sure" — E approaches 1 as D → ∞, concavely.

u(z) = 1 − E(z) is the *uncertainty* view (same information, flipped).

This script:
    - plots the scalar transform E = D/(1+D) with annotations
    - shows the same D-field rendered as D, E, and u side-by-side
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, RED,
    density_cmap, evidence_cmap, uncertainty_cmap,
    save, strip_spines,
)
from concept_02_raw_density import raw_density, grid_2d


def E_of_D(D: np.ndarray) -> np.ndarray:
    """Saturating transform."""
    return D / (1.0 + D)


def u_of_D(D: np.ndarray) -> np.ndarray:
    """Uncertainty = 1 − E."""
    return 1.0 / (1.0 + D)


# ---------- Figure: the scalar transform ----------

def figure_scalar_transform():
    apply_style()
    D_vals = np.linspace(0, 10, 500)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    ax.plot(D_vals, E_of_D(D_vals), color=EMERALD[500], lw=2.0, label="E = D/(1+D)")
    ax.plot(D_vals, u_of_D(D_vals), color=STEEL[500], lw=2.0, label="u = 1/(1+D)")

    # Annotations: specific D values
    for D_pt in [0, 1, 3, 9]:
        ax.axvline(D_pt, color=ZINC[200], lw=0.6, ls="--")
        ax.annotate(
            f"D={D_pt}\nE={E_of_D(D_pt):.2f}",
            (D_pt, E_of_D(D_pt)),
            xytext=(5, -15), textcoords="offset points",
            fontsize=7, color=ZINC[500],
        )

    ax.axhline(1.0, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("raw density  D(z)")
    ax.set_ylabel("transformed value")
    ax.set_title("saturating transform — E grows, u decays, both in [0, 1]", pad=6)
    ax.legend()
    strip_spines(ax)

    fig.suptitle(
        "Scalar transform D ↦ E and D ↦ u",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "03a_scalar_transform")
    plt.close(fig)
    return path


# ---------- Figure: D, E, u fields on the same configuration ----------

def figure_three_views(sigma: float = 0.10, kernel: str = "gaussian"):
    apply_style()
    u, v, Z = grid_2d()
    centers = np.array([
        [0.30, 0.30], [0.70, 0.30], [0.50, 0.50],
        [0.30, 0.70], [0.70, 0.70],
    ])
    w = np.ones(len(centers))
    D = raw_density(Z, centers, w, sigma, kernel=kernel)
    E = E_of_D(D)
    U = u_of_D(D)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)

    for ax, field, title, cmap, vmax, label in [
        (axes[0], D, "D(z)  — raw density (unbounded)", density_cmap(), None, "D"),
        (axes[1], E, "E(z) = D/(1+D)  — actual evidence, [0,1)", evidence_cmap(), 1.0, "E"),
        (axes[2], U, "u(z) = 1/(1+D)  — uncertainty, (0,1]", uncertainty_cmap(), 1.0, "u"),
    ]:
        if vmax is None:
            vmax = float(field.max())
        im = ax.contourf(u, v, field, levels=30, cmap=cmap, vmin=0, vmax=vmax)
        ax.contour(u, v, field, levels=8, colors=ZINC[400], linewidths=0.4, alpha=0.5)
        ax.scatter(centers[:, 0], centers[:, 1],
                   c="white", edgecolors=ZINC[700], s=40, linewidths=0.8, zorder=5)
        ax.set_title(title, pad=6)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)

    fig.suptitle(
        f"Same 5-kernel configuration seen as D, E, u   |   {kernel}, σ = {sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"03b_D_E_u_views_{kernel}")
    plt.close(fig)
    return path


if __name__ == "__main__":
    p1 = figure_scalar_transform()
    print(f"Saved: {p1}")
    for kernel in ("gaussian", "cauchy"):
        p2 = figure_three_views(sigma=0.10, kernel=kernel)
        print(f"Saved: {p2}")
