"""Concept 09 — KernelField design study.

Two design knobs set the quadrature quality:

    radius_multipliers     which σ-multiples we place shells at
    n_directions           how many angular probes per shell

More shells / more directions → better accuracy, more compute. This script
quantifies the trade-off:

    1. Geometry plot of four candidate field variants in 2-D
    2. Mass conservation Σw → 1 for each variant across σ
    3. Integration error on a known polynomial test integrand
    4. Cost/accuracy table: points-per-kernel vs max error

    5. D-scaling: fix a variant, sweep D from 2 to 6, track
       mass conservation and integrand error. Answers "do we need
       different designs in higher D?"
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED, SERIES,
    save, strip_spines,
)
from kernel_field import KernelField


# ---------- Four candidate variants ----------

VARIANTS = {
    "current": {
        "radii": (0.1, 0.2, 0.5, 1.0, 2.0),
        "dirs": 16,
    },
    "compact": {
        # User proposal: drop innermost shells, more directions
        "radii": (0.25, 0.5, 1.0, 2.0),
        "dirs": 24,
    },
    "extended 3σ": {
        "radii": (0.25, 0.5, 1.0, 2.0, 3.0),
        "dirs": 20,
    },
    "fine angular": {
        "radii": (0.1, 0.2, 0.5, 1.0, 2.0),
        "dirs": 32,
    },
}


# ---------- Figure: geometry ----------

def figure_geometry_2d(sigma: float = 0.15):
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    for ax, (name, cfg) in zip(axes, VARIANTS.items()):
        field = KernelField(
            D=2, sigma=sigma,
            radius_multipliers=cfg["radii"], n_directions=cfg["dirs"],
        )
        probes = field.probes_at(np.array([0.5, 0.5]))
        # Shell circles
        for r_mult, r in zip(field.radius_multipliers, field.radii):
            circle = plt.Circle((0.5, 0.5), r, fill=False, ls="--",
                                color=ZINC[300], lw=0.6, alpha=0.6)
            ax.add_patch(circle)
        # Probe points (size ∝ weight)
        max_w = field.weights.max()
        sizes = 30 + 250 * (field.weights / max_w)
        ax.scatter(probes[:, 0], probes[:, 1],
                   c=STEEL[500], s=sizes, edgecolors=ZINC[900],
                   linewidths=0.4, alpha=0.85)
        ax.scatter([0.5], [0.5], marker="x", c=RED, s=60, linewidths=1.5, zorder=6)
        ax.set_title(
            f"{name}\n{field.n_probes} probes · radii={cfg['radii']} · {cfg['dirs']} dirs",
            fontsize=9, pad=6,
        )
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        strip_spines(ax)

    fig.suptitle(
        f"KernelField variants — 2-D geometry   |   σ = {sigma}",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "09a_variants_geometry")
    plt.close(fig)
    return path


# ---------- Figure: mass conservation ----------

def figure_mass_conservation(D: int = 3):
    apply_style()
    sigmas = np.logspace(-2.5, 0, 25)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for (name, cfg), color in zip(VARIANTS.items(), SERIES):
        masses = [
            KernelField(D=D, sigma=s,
                        radius_multipliers=cfg["radii"],
                        n_directions=cfg["dirs"]).check_mass()
            for s in sigmas
        ]
        ax.plot(sigmas, masses, color=color, lw=1.6, label=name)
    ax.axhline(1.0, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("σ")
    ax.set_ylabel("Σ weights  ≈  ∫ρ dz")
    ax.set_title(f"Mass conservation across σ, D = {D}", pad=6)
    ax.set_ylim(0.85, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    strip_spines(ax)
    fig.suptitle(
        "Extended-radius variants catch more of the Gaussian tail",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"09b_mass_conservation_D{D}")
    plt.close(fig)
    return path


# ---------- Figure: integration accuracy on a known integrand ----------

def _polynomial(Z: np.ndarray) -> np.ndarray:
    """A gentle polynomial integrand in D=2 for testing accuracy."""
    x = Z[..., 0]
    y = Z[..., 1]
    return 0.5 + 0.8 * x - 0.3 * x ** 2 + 0.4 * y


def _reference_mc(sigma: float, z_j: np.ndarray, n: int = 1_000_000) -> float:
    rng = np.random.default_rng(42)
    eps = rng.standard_normal((n, 2)) * sigma
    return float(_polynomial(z_j + eps).mean())


def figure_integration_accuracy():
    apply_style()
    z_j = np.array([0.5, 0.5])
    sigmas = np.logspace(-2, 0, 25)

    truth = np.array([_reference_mc(s, z_j) for s in sigmas])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for (name, cfg), color in zip(VARIANTS.items(), SERIES):
        est = []
        for s in sigmas:
            field = KernelField(D=2, sigma=s,
                                 radius_multipliers=cfg["radii"],
                                 n_directions=cfg["dirs"])
            probes = field.probes_at(z_j)
            vals = _polynomial(probes)
            est.append(float(np.sum(field.weights * vals)))
        est = np.array(est)
        axes[0].plot(sigmas, est, color=color, lw=1.6, label=name)
        rel = 100 * (est - truth) / (np.abs(truth) + 1e-12)
        axes[1].plot(sigmas, rel, color=color, lw=1.6, label=name)

    axes[0].plot(sigmas, truth, color=ZINC[900], lw=2.0, label="Reference (10⁶ MC)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("σ")
    axes[0].set_ylabel("∫ f · ρ dz")
    axes[0].set_title("Integration value (polynomial integrand, D=2)", pad=6)
    axes[0].legend(loc="upper right", fontsize=7)
    strip_spines(axes[0])

    axes[1].axhline(0, color=ZINC[400], lw=0.8, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("σ")
    axes[1].set_ylabel("relative error [%]")
    axes[1].set_title("Relative error per variant", pad=6)
    axes[1].legend(loc="upper left", fontsize=7)
    strip_spines(axes[1])

    fig.suptitle(
        "Accuracy of each variant on a smooth integrand",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "09c_integration_accuracy")
    plt.close(fig)
    return path


# ---------- Figure: D-scaling ----------

def figure_d_scaling(variant: str = "current"):
    apply_style()
    cfg = VARIANTS[variant]
    Ds = [1, 2, 3, 4, 5, 6]
    sigmas = [0.05, 0.10, 0.20]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for sigma, color in zip(sigmas, [STEEL[500], EMERALD[500], RED]):
        masses = []
        n_probes = []
        for D in Ds:
            field = KernelField(D=D, sigma=sigma,
                                 radius_multipliers=cfg["radii"],
                                 n_directions=cfg["dirs"])
            masses.append(field.check_mass())
            n_probes.append(field.n_probes)
        ax.plot(Ds, masses, color=color, lw=1.8, marker="o", label=f"σ = {sigma}")

    ax.axhline(1.0, color=ZINC[300], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("D (dimensions)")
    ax.set_ylabel("Σ weights  ≈  ∫ρ dz")
    ax.set_title(
        f"Mass conservation across D — variant: '{variant}'\n"
        f"(radii {cfg['radii']}, {cfg['dirs']} directions → "
        f"{cfg['dirs'] * len(cfg['radii']) + 1} probes per kernel)",
        pad=6,
    )
    ax.set_ylim(0.85, 1.05)
    ax.legend(fontsize=8)
    strip_spines(ax)

    fig.suptitle(
        "D-scaling — does the field stay well-calibrated in higher dimensions?",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, f"09d_d_scaling_{variant.replace(' ', '_')}")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary():
    print()
    print("=" * 78)
    print("KernelField variants — cost vs accuracy  (Gaussian, D=3)")
    print("=" * 78)
    print(f"{'variant':<16s}{'radii':<30s}{'dirs':>6s}{'n_probes':>12s}{'mass σ=0.1':>14s}")
    print("─" * 78)
    for name, cfg in VARIANTS.items():
        f = KernelField(D=3, sigma=0.10,
                        radius_multipliers=cfg["radii"],
                        n_directions=cfg["dirs"])
        mass = f.check_mass()
        radii_str = str(cfg["radii"])
        print(f"{name:<16s}{radii_str:<30s}{cfg['dirs']:>6d}{f.n_probes:>12d}{mass:>14.4f}")
    print()


if __name__ == "__main__":
    print_summary()
    p1 = figure_geometry_2d()
    p2 = figure_mass_conservation(D=3)
    p3 = figure_integration_accuracy()
    p4 = figure_d_scaling()
    for p in (p1, p2, p3, p4):
        print(f"Saved: {p}")
