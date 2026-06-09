# visualization/ — Context

## Purpose
Figures and console output for the showcase. All figures follow the Visual Identity (Zinc/Steel/Emerald palette, softened RdYlGn colormaps, no top/right spines, 600 DPI).

## Modules

| Module | What it provides |
|---|---|
| `_style.py` | Single source of truth for style: palette constants, `PHASE_COLORS`, softened `DEVIATION_CMAP` / `PERFORMANCE_CMAP`, `apply_style()`, `clean_spines`/`clean_3d_panes`/`style_colorbar`/`light_grid`, `save_fig`. |
| `plots.py` | Figure functions: `plot_stage_print` (one stage's average 3-D print, called per stage on a shared scale), `plot_parameter_topology` (sampled points over the true-physics landscape + optimum; additive via `stages_shown`), `plot_performance_timeline`, `plot_feature_heatmaps`, `plot_prediction_accuracy`. |
| `console.py` | Terminal phase headers, experiment rows, summaries. |

## Key Points
- `plots.py` imports the palette/helpers from `_style`; never hardcode colors or DPI in plot functions.
- Path deviation is coloured low=green→high=red (`DEVIATION_CMAP`); stage prints share one `vmax` so the three are comparable.
- `plot_parameter_topology` is called once per stage with a growing `stages_shown` tuple for an additive reveal.
