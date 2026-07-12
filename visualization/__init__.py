"""Domain-specific visualization helpers and 3D process plots.

Generic PFAB plots live in ``pred_fab.plotting``. This module provides only:
- Domain-specific data generation (physics grid evaluation + ground-truth optimum)
- 3D filament visualization (requires live CameraSystem)
- The journey summary plot
"""

from .helpers import save_fig, physics_combined_at, evaluate_physics_grid, get_physics_optimum
from .process import plot_path_comparison_3d
from .journey import plot_journey

__all__ = [
    "save_fig",
    "physics_combined_at",
    "evaluate_physics_grid",
    "get_physics_optimum",
    "plot_path_comparison_3d",
    "plot_journey",
]
