"""Concept 10 — estimator comparison, budget-matched per (n_dirs, D) cell.

Parameter sweep:
    n_dirs  ∈ {4, 8, 16, 32, 64, 128}     — angular resolution per shell
    D       ∈ {2, 3, 4, 5}                 — kernel dimensionality
    n_radii = 5                            — D-aware χ² quantile shells
    topology = 3 clustered + 3 scattered   — realistic overlap regime

For each (n_dirs, D) cell:
    per_kernel_budget = n_dirs · n_radii + 1    (the KernelField total)
    total_budget      = per_kernel_budget × N
All three estimators (KernelField, importance sampling, Sobol uniform) are
evaluated at EXACTLY this budget, so the comparison is fair: same cost,
different sampling strategy.

Three figures produced:
    10a  — error vs n_dirs, one panel per D, lines = estimator
    10b  — error vs D, one panel per n_dirs, lines = estimator
    10c  — heatmap of KernelField error on the (n_dirs, D) grid; overlay
           IS and Sobol as "budget needed to beat" reference curves
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.stats import qmc

from _style import (
    apply_style, STEEL, EMERALD, ZINC, YELLOW, RED,
    save, strip_spines,
)
from kernel_field import KernelField
from concept_02_raw_density import raw_density


N_RADII = 5
QUANTILES = (0.02, 0.2, 0.5, 0.8, 0.98)   # D-aware shells via χ² quantiles


# ---------- Test problem ----------

def _test_centers(D: int, N: int = 6, seed: int = 0) -> np.ndarray:
    """Realistic calibration topology: tight cluster + scattered outliers.

    Three kernels spread tightly around the interior (σ-scale separations),
    three kernels placed further out. This reproduces the overlap regime the
    evidence integral actually has to resolve mid-calibration.
    """
    rng = np.random.default_rng(seed)
    cluster_center = np.full(D, 0.5)
    cluster = cluster_center + rng.normal(0.0, 0.08, size=(3, D))
    scattered = rng.uniform(0.12, 0.88, size=(3, D))
    centers = np.vstack([cluster, scattered])
    return np.clip(centers, 0.02, 0.98)


# ---------- Estimators (at matched probe budget) ----------

def estimate_field(centers, sigma, n_dirs: int, D: int):
    field = KernelField(
        D=D, sigma=sigma, n_directions=n_dirs,
        radii_mode="chi2_quantile", radii_quantiles=QUANTILES,
    )
    probes = field.probes_for_batch(centers)
    flat = probes.reshape(-1, D)
    in_domain = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    D_vals = raw_density(flat, centers, np.ones(len(centers)), sigma)
    integrand = 1.0 / (1.0 + D_vals) * in_domain.astype(float)
    M = probes.shape[1]
    per_k = (integrand.reshape(len(centers), M) * field.weights[None, :]).sum(axis=1)
    return float(per_k.sum()), len(centers) * M


def estimate_importance(centers, sigma, budget: int, D: int, seed: int = 0):
    N = len(centers)
    per_kernel = max(2, budget // N)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((N, per_kernel, D)) * sigma
    samples = centers[:, None, :] + eps
    flat = samples.reshape(-1, D)
    in_domain = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    D_vals = raw_density(flat, centers, np.ones(N), sigma)
    integrand = np.where(in_domain, 1.0 / (1.0 + D_vals), 0.0)
    per_k = integrand.reshape(N, per_kernel).mean(axis=1)
    return float(per_k.sum()), N * per_kernel


def estimate_sobol(centers, sigma, budget: int, D: int, seed: int = 0):
    n = 2 ** int(np.ceil(np.log2(max(budget, 2))))
    sobol = qmc.Sobol(d=D, scramble=True, rng=seed).random(n=n)
    D_vals = raw_density(sobol, centers, np.ones(len(centers)), sigma)
    return float((D_vals / (1.0 + D_vals)).mean()), n


# ---------- Reference ----------

def reference(centers, sigma: float, D: int, res: int = 81) -> float:
    if D == 2:
        u = np.linspace(0, 1, res)
        U, V = np.meshgrid(u, u)
        Z = np.stack([U, V], axis=-1)
        D_vals = raw_density(Z, centers, np.ones(len(centers)), sigma)
        E = D_vals / (1.0 + D_vals)
        return float(np.trapezoid(np.trapezoid(E, u, axis=0), u, axis=0))
    if D == 3:
        r3 = min(res, 61)
        u = np.linspace(0, 1, r3)
        U, V, W = np.meshgrid(u, u, u, indexing="ij")
        Z = np.stack([U, V, W], axis=-1)
        D_vals = raw_density(Z.reshape(-1, 3), centers, np.ones(len(centers)), sigma).reshape(r3, r3, r3)
        E = D_vals / (1.0 + D_vals)
        return float(
            np.trapezoid(np.trapezoid(np.trapezoid(E, u, axis=0), u, axis=0), u, axis=0)
        )
    # D >= 4: use a big Sobol batch as the "truth" reference
    n = 2 ** 17  # 131k samples
    sobol = qmc.Sobol(d=D, scramble=True, rng=42).random(n=n)
    D_vals = raw_density(sobol, centers, np.ones(len(centers)), sigma)
    return float((D_vals / (1.0 + D_vals)).mean())


# ---------- Sweep ----------

def sweep(
    sigma: float = 0.10,
    Ds: tuple[int, ...] = (2, 3, 4, 5),
    n_dirs_values: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    N_kernels: int = 6,
) -> dict:
    """Returns nested dict: results[estimator][D][n_dirs] = (est, truth, budget)."""
    out: dict = {
        "KernelField": {D: {} for D in Ds},
        "Importance":  {D: {} for D in Ds},
        "Sobol":       {D: {} for D in Ds},
        "truth":       {D: None for D in Ds},
        "budgets":     {D: {} for D in Ds},
    }
    for D in Ds:
        centers = _test_centers(D, N=N_kernels)
        truth = reference(centers, sigma, D)
        out["truth"][D] = truth
        for n_dirs in n_dirs_values:
            sten_val, sten_n = estimate_field(centers, sigma, n_dirs, D)
            budget = sten_n  # the canonical budget for this (n_dirs, D) cell
            imps_val, _ = estimate_importance(centers, sigma, budget, D)
            sobs_val, sobs_n = estimate_sobol(centers, sigma, budget, D)
            out["budgets"][D][n_dirs] = budget
            out["KernelField"][D][n_dirs] = sten_val
            out["Importance"][D][n_dirs]  = imps_val
            out["Sobol"][D][n_dirs]       = sobs_val
    return out


# ---------- Figure 1: error vs n_dirs, one panel per D ----------

def figure_vs_ndirs(results, Ds, n_dirs_values):
    apply_style()
    fig, axes = plt.subplots(1, len(Ds), figsize=(3.6 * len(Ds), 3.8), constrained_layout=True)
    for ax, D in zip(axes, Ds):
        truth = results["truth"][D]
        for name, color in [("KernelField", STEEL[500]),
                            ("Importance", EMERALD[500]),
                            ("Sobol", RED)]:
            err = [
                100 * (results[name][D][nd] - truth) / (abs(truth) + 1e-12)
                for nd in n_dirs_values
            ]
            ax.plot(n_dirs_values, err, color=color, lw=1.6, marker="o", ms=4, label=name)
        ax.axhline(0, color=ZINC[400], lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_dirs")
        ax.set_ylabel("relative error [%]")
        # Secondary axis: total budget
        budgets = [results["budgets"][D][nd] for nd in n_dirs_values]
        ax.set_title(
            f"D = {D}\nbudget: {budgets[0]}–{budgets[-1]} probes",
            pad=6, fontsize=9,
        )
        if D == Ds[0]:
            ax.legend(fontsize=7, loc="best")
        strip_spines(ax)
    fig.suptitle(
        "Error vs angular resolution at fixed budget, per D   "
        "(σ=0.10, χ² quantile shells, N=6 kernels: 3 clustered + 3 scattered)",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "10a_error_vs_ndirs")
    plt.close(fig)
    return path


# ---------- Figure 2: error vs D, one panel per n_dirs ----------

def figure_vs_D(results, Ds, n_dirs_values):
    apply_style()
    # Show three interesting n_dirs values
    picks = [nd for nd in (8, 32, 128) if nd in n_dirs_values]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.8 * len(picks), 3.8), constrained_layout=True)
    for ax, n_dirs in zip(axes, picks):
        for name, color in [("KernelField", STEEL[500]),
                            ("Importance", EMERALD[500]),
                            ("Sobol", RED)]:
            err = [
                100 * (results[name][D][n_dirs] - results["truth"][D])
                / (abs(results["truth"][D]) + 1e-12)
                for D in Ds
            ]
            ax.plot(Ds, err, color=color, lw=1.6, marker="s", ms=5, label=name)
        ax.axhline(0, color=ZINC[400], lw=0.8, ls="--")
        budgets_this_ndirs = [results["budgets"][D][n_dirs] for D in Ds]
        ax.set_xlabel("D (dimensions)")
        ax.set_ylabel("relative error [%]")
        ax.set_title(
            f"n_dirs = {n_dirs}\nbudget per D: " + ", ".join(str(b) for b in budgets_this_ndirs),
            pad=6, fontsize=9,
        )
        ax.set_xticks(list(Ds))
        if n_dirs == picks[0]:
            ax.legend(fontsize=7, loc="best")
        strip_spines(ax)
    fig.suptitle(
        "Error vs D at fixed angular resolution (KernelField n_dirs = 8, 32, 128)",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "10b_error_vs_D")
    plt.close(fig)
    return path


# ---------- Figure 3: heatmap of KernelField error ----------

def figure_heatmap(results, Ds, n_dirs_values):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, name, cmap in zip(
        axes, ["KernelField", "Importance", "Sobol"], ["RdBu_r", "RdBu_r", "RdBu_r"]
    ):
        err = np.zeros((len(Ds), len(n_dirs_values)))
        for i, D in enumerate(Ds):
            truth = results["truth"][D]
            for j, nd in enumerate(n_dirs_values):
                err[i, j] = 100 * (results[name][D][nd] - truth) / (abs(truth) + 1e-12)
        vmax = max(abs(err.min()), abs(err.max()), 5.0)
        im = ax.imshow(err, cmap=cmap, aspect="auto",
                       vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(n_dirs_values)), [str(n) for n in n_dirs_values])
        ax.set_yticks(range(len(Ds)), [f"D={D}" for D in Ds])
        ax.set_xlabel("n_dirs")
        ax.set_title(f"{name} — relative error [%]", pad=6)
        # Annotate cells
        for i in range(len(Ds)):
            for j in range(len(n_dirs_values)):
                ax.text(j, i, f"{err[i, j]:+.0f}",
                        ha="center", va="center", fontsize=7,
                        color=ZINC[900] if abs(err[i, j]) < vmax * 0.6 else "white")
        fig.colorbar(im, ax=ax, shrink=0.85, label="% error")
    fig.suptitle(
        "Relative error on the (n_dirs, D) grid — budget matched per cell  "
        "(σ=0.10, χ² quantile shells, realistic topology)",
        fontsize=11, color=ZINC[700],
    )
    path = save(fig, "10c_error_heatmap")
    plt.close(fig)
    return path


# ---------- Terminal print ----------

def print_summary(results, Ds, n_dirs_values):
    print()
    print("=" * 90)
    print("Estimator comparison — budget-matched per (n_dirs, D)  (σ=0.10)")
    print("=" * 90)
    header = f"{'':>12s}" + "".join(f"{nd:>10d}" for nd in n_dirs_values)
    print(header)
    for name in ("KernelField", "Importance", "Sobol"):
        print(f"\n-- {name} error [%] --")
        for D in Ds:
            truth = results["truth"][D]
            row = f"D={D:<10d}"
            for nd in n_dirs_values:
                err = 100 * (results[name][D][nd] - truth) / (abs(truth) + 1e-12)
                row += f"{err:>+10.1f}"
            print(row)

    print("\n-- budget per cell (total probes) --")
    for D in Ds:
        row = f"D={D:<10d}"
        for nd in n_dirs_values:
            row += f"{results['budgets'][D][nd]:>10d}"
        print(row)
    print()


def print_radii_diagnostic(Ds, n_dirs_ref: int = 32):
    print()
    print("χ² quantile shells — D-aware radii (in σ units) and mass conservation")
    print("-" * 78)
    for D in Ds:
        field = KernelField(
            D=D, sigma=1.0, n_directions=n_dirs_ref,
            radii_mode="chi2_quantile", radii_quantiles=QUANTILES,
        )
        radii_str = " ".join(f"{r:5.2f}" for r in field.radii)
        print(f"D={D}   radii_σ = [{radii_str}]   Σw = {field.check_mass():.4f}")


if __name__ == "__main__":
    Ds = (2, 3, 4, 5)
    n_dirs_values = (4, 8, 16, 32, 64, 128)
    print_radii_diagnostic(Ds)
    results = sweep(sigma=0.10, Ds=Ds, n_dirs_values=n_dirs_values)
    print_summary(results, Ds, n_dirs_values)
    p1 = figure_vs_ndirs(results, Ds, n_dirs_values)
    p2 = figure_vs_D(results, Ds, n_dirs_values)
    p3 = figure_heatmap(results, Ds, n_dirs_values)
    for p in (p1, p2, p3):
        print(f"Saved: {p}")
