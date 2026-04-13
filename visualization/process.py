"""Process visualizations: 3D printed filament views."""

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sensors.physics import N_LAYERS, N_SEGMENTS, FILAMENT_RADIUS
from .helpers import save_fig


def _make_filament_tube(
    xs: list[float],
    ys: list[float],
    z_center: float,
    radius: float,
    n_circ: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) surface arrays for a cylindrical tube following (xs, ys) at z_center."""
    n_pts = len(xs)
    phi = np.linspace(0, 2.0 * np.pi, n_circ, endpoint=True)
    PHI = phi[:, np.newaxis]                        # (n_circ, 1)
    XS = np.array(xs, dtype=float)[np.newaxis, :]   # (1, n_pts)
    YS = np.array(ys, dtype=float)[np.newaxis, :]   # (1, n_pts)
    X = np.repeat(XS, n_circ, axis=0)               # (n_circ, n_pts)
    Y = YS + radius * np.cos(PHI)                   # (n_circ, n_pts)
    Z = z_center + radius * np.sin(PHI)             # (n_circ, n_pts)
    return X, Y, Z


def plot_path_comparison_3d(
    save_path: str,
    camera: Any,
    params: dict[str, Any],
    exp_code: str = "",
) -> None:
    """3D stacked tube view: designed (blue wireframe) vs as-printed (coloured solid).

    All layers x segments rendered as cylindrical filament tubes.
    Designed path = blue wireframe ghost at y=0.
    Measured path = solid tubes coloured by deviation (green -> red).
    y-axis scaled x3 so lateral offset reads clearly.
    Layer drift gradient (L0 green, L4 dark red) immediately visible.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_layers = int(params.get("n_layers", N_LAYERS))
    n_segments = int(params.get("n_segments", N_SEGMENTS))
    n_pts = 5
    seg_length = (n_pts - 1) * 0.01
    seg_gap = 0.008
    radius = FILAMENT_RADIUS
    layer_step = radius * 2.6

    Y_SCALE = 3.0

    # Pre-fetch all segment data and find global deviation range
    cache: dict[tuple[int, int], dict] = {}
    all_devs: list[float] = []
    for li in range(n_layers):
        for si in range(n_segments):
            data = camera.get_segment_data(params, li, si)
            cache[(li, si)] = data
            for mp, dp in zip(data["measured_path"], data["designed_path"]):
                all_devs.append(abs(mp[1] - dp[1]))

    vmax = max(all_devs) * 1.1 if all_devs else 1e-4
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")

    for li in range(n_layers):
        z_center = li * layer_step
        x_off = 0.0

        for si in range(n_segments):
            data = cache[(li, si)]
            dp = data["designed_path"]
            mp = data["measured_path"]
            xs = [p[0] + x_off for p in dp]
            ys_des = [0.0] * len(dp)
            ys_meas_r = [p[1] for p in mp]
            ys_meas_v = [y * Y_SCALE for y in ys_meas_r]

            mean_dev = float(np.mean([abs(ym) for ym in ys_meas_r]))
            tube_color = cmap(norm(mean_dev))

            # Designed — wireframe ghost at y=0
            Xd, Yd, Zd = _make_filament_tube(xs, ys_des, z_center, radius, n_circ=16)
            ax.plot_wireframe(
                Xd, Yd, Zd,
                color="#6699CC", alpha=0.35, linewidth=0.4,
                rstride=4, cstride=1,
            )
            ax.plot(
                xs, ys_des, [z_center] * len(xs),
                color="#AACCFF", linestyle="--", linewidth=1.0, alpha=0.80, zorder=4,
            )

            # Measured — solid coloured tube (y-scaled)
            Xm, Ym, Zm = _make_filament_tube(xs, ys_meas_v, z_center, radius, n_circ=20)
            ax.plot_surface(
                Xm, Ym, Zm,
                color=tube_color, alpha=0.88, linewidth=0, antialiased=True, shade=True,
            )

            x_off += seg_length + seg_gap

    ax.set_box_aspect([9, 2.5, 2.2])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.09, shrink=0.52, aspect=16)
    cb.set_label("Path Deviation [m]", fontsize=9)

    speed = params.get("print_speed", 0.0)
    water = params.get("water_ratio", 0.0)

    ax.set_xlabel("Along-path [m]", labelpad=10, fontsize=9)
    ax.set_ylabel(f"Lateral offset [m x{Y_SCALE:.0f}]", labelpad=10, fontsize=9)
    ax.set_zticks([i * layer_step for i in range(n_layers)])
    ax.set_zticklabels([f"L{i}" for i in range(n_layers)])

    title = "As-Printed vs As-Designed"
    if exp_code:
        title += f"  ·  {exp_code}"
    title += (
        f"\nwater_ratio={water:.2f}   speed={speed:.1f} mm/s"
        f"\nWireframe = designed   Solid = as-printed   Colour = deviation"
    )
    ax.set_title(title, pad=12, fontsize=10)
    ax.view_init(elev=28, azim=-62)

    save_fig(save_path)
