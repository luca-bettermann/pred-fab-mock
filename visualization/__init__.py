"""Domain-specific visualization helpers and 3D process plots.

Generic PFAB plots live in ``pred_fab.plotting``. This module provides only:
- Domain-specific data generation (physics grid evaluation)
- 3D filament visualization (requires live CameraSystem)
"""

from .helpers import save_fig, physics_combined_at, evaluate_physics_grid
from .process import plot_path_comparison_3d

__all__ = [
    "save_fig",
    "physics_combined_at",
    "evaluate_physics_grid",
    "plot_path_comparison_3d",
]
