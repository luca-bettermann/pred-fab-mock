#!/usr/bin/env python3
"""
topology_viz.py — visualise evidence, performance and combined acquisition landscapes.

Figure 1: bandwidth comparison (Silverman vs narrow) with sharpness γ, N fixed.
Figure 2: dynamic bandwidth h = c/√N — how the landscape evolves as data accumulates.

Run: uv run python topology_viz.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("./plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 2-D parameter slice: print_speed × water_ratio
#    speed_norm ∈ [0,1] → [20, 60] mm/s
#    wr_norm    ∈ [0,1] → [0.30, 0.50]
# ─────────────────────────────────────────────────────────────────────────────

N_GRID = 60
xs = np.linspace(0, 1, N_GRID)
ys = np.linspace(0, 1, N_GRID)
XS, YS = np.meshgrid(xs, ys)
GRID = np.column_stack([XS.ravel(), YS.ravel()])   # (N_GRID², 2)


def _path_accuracy(sn: float, wrn: float) -> float:
    """Path accuracy score ∈ [0,1] — low speed and low wr preferred."""
    return float(np.clip(0.9 - 0.8 * sn - 0.1 * wrn, 0.0, 1.0))


def _energy_efficiency(sn: float, wrn: float) -> float:
    """Energy efficiency score ∈ [0,1] — peaks around speed_norm ≈ 0.375 (35 mm/s)."""
    return float(np.clip(1.0 - 6.0 * (sn - 0.375) ** 2, 0.0, 1.0))


def perf_score(sn: float, wrn: float) -> float:
    return 0.5 * _path_accuracy(sn, wrn) + 0.5 * _energy_efficiency(sn, wrn)


PERF = np.array([perf_score(x, y) for x, y in GRID]).reshape(N_GRID, N_GRID)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Evidence / uncertainty
#    KDE density:  q(x)  = mean_i exp(-||x - x_i||² / 2h²)
#    NatPN:        n_post = N · (q / q_max)^γ
#    Uncertainty:  u      = 1 / (1 + n_post)
#    Normalised:   u_norm = (u - u_min) / (1 - u_min)   [maps to [0,1]]
# ─────────────────────────────────────────────────────────────────────────────

def _kde(query: np.ndarray, train: np.ndarray, h: float) -> np.ndarray:
    """Mean Gaussian kernel density at each query point. Shape: (M,)."""
    dists_sq = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=2)  # (M, N)
    return np.mean(np.exp(-dists_sq / (2.0 * h ** 2)), axis=1)


def evidence(train: np.ndarray, h: float, gamma: float = 1.0) -> np.ndarray:
    """Normalised uncertainty grid u_norm ∈ [0,1]."""
    N = len(train)
    q_grid  = _kde(GRID,  train, h)
    q_train = _kde(train, train, h)
    q_max   = float(q_train.max())
    ratio   = np.clip(q_grid / (q_max + 1e-12), 0.0, 1.0)
    n_post  = N * (ratio ** gamma)
    u       = 1.0 / (1.0 + n_post)
    u_min   = 1.0 / (1.0 + N)
    return np.clip((u - u_min) / (1.0 - u_min + 1e-12), 0.0, 1.0).reshape(N_GRID, N_GRID)


def silverman_bw(N: int, d: int = 2) -> float:
    """Silverman rule for d-dimensional uniform [0,1]^d data (σ = 1/√12)."""
    return ((N * (d + 2)) / 4.0) ** (-1.0 / (d + 4)) / np.sqrt(12.0)


def narrow_bw(N: int, c: float) -> float:
    """Proposed: h = c / √N.  c is the bubble radius at N=1."""
    return c / np.sqrt(float(N))


def dynamic_gamma(N: int, c: float) -> float:
    """γ = max(1.0, c·√N) — steepens as evidence matures.

    Below c·√N=1 (few points): γ=1, smooth bubbles.
    Above: γ increases, edges sharpen.  Same c governs both h and γ.
    """
    return max(1.0, c * np.sqrt(float(N)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Latin hypercube training points
# ─────────────────────────────────────────────────────────────────────────────

def lhs_points(n: int, d: int = 2, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pts = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.uniform(size=n)) / float(n)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_row(axes, Z_list, train, row_label, col_titles, row_idx, fig):
    for col, (Z, ctitle) in enumerate(zip(Z_list, col_titles)):
        ax = axes[row_idx, col]
        im = ax.contourf(XS, YS, Z, levels=20, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.scatter(train[:, 0], train[:, 1], c="red", s=25, zorder=5, linewidths=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if row_idx == 0:
            ax.set_title(ctitle, fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{row_label}\nwr_norm", fontsize=8)
        ax.set_xlabel("speed_norm", fontsize=8)


COL_TITLES = ["Evidence  u(x)", "Performance  perf(x)", "Combined  α(x)"]
W = 0.7   # κ (exploration weight)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: bandwidth & sharpness comparison (N=12 fixed)
# ─────────────────────────────────────────────────────────────────────────────

N12 = 12
train12 = lhs_points(N12)

h_sil = silverman_bw(N12)
h_nar = narrow_bw(N12, c=0.5)

scenarios = [
    (f"Silverman  h={h_sil:.3f}",         h_sil, 1.0),
    (f"Narrow  c=0.5, γ=1  h={h_nar:.3f}", h_nar, 1.0),
    (f"Narrow  c=0.5, γ=2  h={h_nar:.3f}", h_nar, 2.0),
]

fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12))
fig1.suptitle(
    f"Bandwidth & sharpness comparison  (N={N12}, κ={W})\n"
    "Red = training points (LHS)   |   x = speed_norm [0→60 mm/s]   |   y = wr_norm",
    fontsize=11,
)

for row, (label, h, gamma) in enumerate(scenarios):
    U     = evidence(train12, h, gamma)
    ALPHA = (1.0 - W) * PERF + W * U
    _plot_row(axes1, [U, PERF, ALPHA], train12, label, COL_TITLES, row, fig1)

plt.tight_layout()
plt.savefig("./plots/topology_bandwidth_comparison.png", dpi=120)
print("Saved ./plots/topology_bandwidth_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: dynamic bandwidth h = c/√N — evolution with N
# ─────────────────────────────────────────────────────────────────────────────

C      = 0.5
stages = [4, 12, 24]

fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
fig2.suptitle(
    f"Dynamic  h = {C}/√N,  γ = max(1, {C}·√N)  (κ={W})\n"
    "Both bubble radius and edge steepness adapt as evidence accumulates",
    fontsize=11,
)

for row, n in enumerate(stages):
    train_n = lhs_points(n)
    h       = narrow_bw(n, C)
    g       = dynamic_gamma(n, C)
    U       = evidence(train_n, h, g)
    ALPHA   = (1.0 - W) * PERF + W * U
    label   = f"N={n:2d}  h={h:.3f}  γ={g:.2f}"
    _plot_row(axes2, [U, PERF, ALPHA], train_n, label, COL_TITLES, row, fig2)

plt.tight_layout()
plt.savefig("./plots/topology_dynamic_bandwidth.png", dpi=120)
print("Saved ./plots/topology_dynamic_bandwidth.png")

plt.show()
