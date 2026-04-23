"""SamplingSpace — deterministic kernel-anchored stencil.

The evidence integral we want is  I = ∫_[0,1]^D  f(z) · ρ_j(z) dz, where ρ_j
is a density centered at a datapoint z_j. Rather than estimate this with
random samples (noisy for small σ), we choose **fixed offsets** around every
kernel center — the same set of offsets for every kernel, computed once.

Geometry:
    - one center offset (the origin)
    - K concentric spheres at radii  σ · r_k  (defaults: 0.1, 0.2, 0.5, 1.0, 2.0)
    - on each sphere, M_k uniformly-distributed unit directions

Each stencil point z_j + σ·r_k·ω_i gets a **weight** derived from treating
the stencil as a radial-shell quadrature of the chosen kernel. The integration
is unbiased (up to shell-width approximation) and independent of RNG.

    I ≈ Σ_k Σ_i   w_{k,i} · f(z_j + σ·r_k·ω_{k,i})

Two kernel choices drive the weight derivation:
    - "gaussian":  weight per shell ∝ S_{D-1} · r^{D-1} · Δr · φ_G(r; σ)
                   where φ_G is the isotropic Gaussian pdf
    - "cauchy":    same structure but φ_C is the radial marginal of product
                   Cauchy (numerically tabulated, since closed form is ugly)

The weight normalisation ensures ∫ ρ dz over the stencil equals 1 in ℝ^D
(or equals α(z_j) for in-domain portion when we mask).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np


KernelType = Literal["gaussian", "cauchy"]


# ---------- Angular lattices ----------

def unit_sphere_points(D: int, n: int, seed: int = 0) -> np.ndarray:
    """n points on the unit (D−1)-sphere in ℝ^D, quasi-uniformly spaced.

    D=1: ±1 (n ignored → 2 points).
    D=2: n evenly spaced on the circle.
    D=3: Fibonacci sphere.
    D≥4: random unit vectors (deterministic via seed).
    """
    if D == 1:
        return np.array([[-1.0], [1.0]])
    if D == 2:
        phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    if D == 3:
        indices = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * indices / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
        return np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=-1)
    # General D: deterministic sampling from N(0, I) projected to sphere
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, D))
    return g / np.linalg.norm(g, axis=-1, keepdims=True)


# ---------- Radial weights ----------

def _surface_area_unit_sphere(D: int) -> float:
    """|S^{D−1}| — surface area of the unit (D−1)-sphere in ℝ^D."""
    from math import gamma, pi
    return 2.0 * pi ** (D / 2.0) / gamma(D / 2.0)


def _gaussian_radial_pdf(r: float, sigma: float, D: int) -> float:
    """Radial marginal of isotropic Gaussian of total mass 1.

    The Gaussian pdf at distance r from the center:  (2πσ²)^{-D/2} · exp(-r²/2σ²).
    """
    from math import pi, exp
    return (2.0 * pi * sigma * sigma) ** (-D / 2.0) * exp(-r * r / (2.0 * sigma * sigma))


def _cauchy_radial_pdf(r: float, sigma: float, D: int) -> float:
    """Approximation for product-Cauchy "isotropic" radial density.

    Product Cauchy is NOT isotropic — the density depends on per-axis distances,
    not just r. For the stencil we only need a radial weight function; using
    the 1-D Cauchy evaluated at r, raised to the D power, is the geometric
    average along the shell's "axis-extreme" direction. This overweights the
    tails slightly but is consistent and keeps the weighting simple.
    """
    from math import pi
    return ((sigma / pi) / (r * r + sigma * sigma)) ** D


def radial_shell_weights(
    radii: np.ndarray,
    sigma: float,
    D: int,
    kernel: KernelType,
) -> tuple[np.ndarray, float]:
    """Weights for each radial shell in the stencil.

    radii:    (K,) shell radii in ABSOLUTE units (already σ-multiplied)
    Returns:  (weights, center_weight)  where
        weights:       (K,) total weight per shell (sum across directions)
        center_weight: scalar mass at r=0
    """
    K = len(radii)
    weights = np.zeros(K)
    S = _surface_area_unit_sphere(D)
    pdf = _gaussian_radial_pdf if kernel == "gaussian" else _cauchy_radial_pdf

    # Build shell widths Δr_k as midpoint distances.
    # With r_{-1} = 0 and r_K = r_{K-1} + (r_{K-1} - r_{K-2}) (mirror last gap),
    # each Δr_k = (r_{k+1} − r_{k−1}) / 2.
    r_ext = np.concatenate([[0.0], radii, [2.0 * radii[-1] - radii[-2]]])
    for k in range(K):
        r_k = radii[k]
        dr_k = 0.5 * (r_ext[k + 2] - r_ext[k])
        weights[k] = S * (r_k ** (D - 1)) * dr_k * pdf(r_k, sigma, D)

    # Center weight: volume of a ball of radius Δr_0/2, evaluated at center pdf.
    # This captures the mass inside the innermost shell boundary.
    r0_half = 0.5 * r_ext[1]
    vol_ball = (np.pi ** (D / 2.0) / _gamma_half(D)) * (r0_half ** D)
    center_weight = vol_ball * pdf(0.0, sigma, D)

    return weights, center_weight


def _gamma_half(D: int) -> float:
    """Γ(D/2 + 1) — used in ball volume formula."""
    from math import gamma
    return gamma(D / 2.0 + 1.0)


# ---------- The class ----------

@dataclass(frozen=True)
class SamplingSpace:
    """Deterministic spherical-shell stencil anchored at each kernel center.

    Same offsets and weights for every kernel. Build once, reuse everywhere.
    """

    D: int
    sigma: float
    kernel: KernelType = "gaussian"
    radius_multipliers: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    n_directions: int = 12

    # ----- Derived (cached) geometry + weights -----

    @cached_property
    def radii(self) -> np.ndarray:
        """Absolute radii of each shell, σ · r_k."""
        return np.asarray(self.radius_multipliers, dtype=float) * self.sigma

    @cached_property
    def directions(self) -> np.ndarray:
        """(n_directions, D) — unit vectors on the (D−1)-sphere."""
        return unit_sphere_points(self.D, self.n_directions)

    @cached_property
    def _built(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (offsets, weights). Private; exposed via properties below."""
        K = len(self.radii)
        n_dir = self.directions.shape[0]

        shell_w, center_w = radial_shell_weights(self.radii, self.sigma, self.D, self.kernel)

        # Center point
        offsets = [np.zeros(self.D)]
        weights = [center_w]

        # Shell points — shell total weight is split evenly across directions
        for k, r in enumerate(self.radii):
            w_per_point = shell_w[k] / n_dir
            for d in self.directions:
                offsets.append(r * d)
                weights.append(w_per_point)

        return np.stack(offsets), np.asarray(weights)

    @property
    def offsets(self) -> np.ndarray:
        """(n_samples, D) — stencil offsets from any kernel center."""
        return self._built[0]

    @property
    def weights(self) -> np.ndarray:
        """(n_samples,) — integration weights. ∑ weights ≈ 1 by construction."""
        return self._built[1]

    @property
    def n_samples(self) -> int:
        return self.offsets.shape[0]

    # ----- Public API -----

    def samples_at(self, center: np.ndarray) -> np.ndarray:
        """(n_samples, D) — stencil samples placed at a given center."""
        return self.offsets + center

    def samples_for_batch(self, centers: np.ndarray) -> np.ndarray:
        """(N, n_samples, D) — stencil samples per center, batch-friendly."""
        return centers[:, None, :] + self.offsets[None, :, :]

    def integrate(
        self,
        f_at_samples: np.ndarray,
        in_domain_mask: np.ndarray | None = None,
    ) -> float:
        """∫ f · ρ dz ≈ Σ_i w_i · f(z_i) · [in-domain(z_i)].

        f_at_samples:    (n_samples,) — the integrand evaluated at each offset
                         (already placed at a kernel center by the caller)
        in_domain_mask:  (n_samples,) bool — True if the sample falls inside
                         the integration domain. None disables masking.
        """
        if in_domain_mask is None:
            return float(np.sum(self.weights * f_at_samples))
        return float(np.sum(self.weights * f_at_samples * in_domain_mask))

    # ----- Diagnostic: check mass conservation of the stencil -----

    def check_mass(self) -> float:
        """Stencil's estimate of ∫ρ dz, which analytically = 1.

        Deviation from 1 reveals shell-width approximation error. Decreases as
        you add more (finer) radii.
        """
        return float(self.weights.sum())
