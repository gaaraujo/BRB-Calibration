"""Shared figure sizes (inches), DPI, and typography for BRB-Calibration plots."""
from __future__ import annotations

import matplotlib.pyplot as plt

# Point size: labels, ticks, titles, legend (single consistent base).
PLOT_FONT_SIZE_PT: float = 9.0

# Larger type for multi-panel montages (specimen grids, combined overlays).
PLOT_FONT_SIZE_GRID_MONTAGE_PT: float = 18.0

# ``results/plots/apparent_b/b_vs_geometry/*``: same grid style, smaller type for denser panels.
B_VS_GEOMETRY_FONT_SCALE: float = 0.7
PLOT_FONT_SIZE_B_VS_GEOMETRY_PT: float = PLOT_FONT_SIZE_GRID_MONTAGE_PT * B_VS_GEOMETRY_FONT_SCALE

# Compact legends for small single-axis figures (e.g. 2.5×2.25 in); keeps in-axes area clear.
LEGEND_FONT_SIZE_SMALL_PT: float = 6.5

# Legend text on grid montages (matches ``legend.fontsize`` pattern vs base plot size).
LEGEND_FONT_SIZE_GRID_MONTAGE_PT: float = PLOT_FONT_SIZE_GRID_MONTAGE_PT - 0.75

# Legend spacing (matplotlib defaults: labelspacing 0.5, columnspacing 2.0, handletextpad 0.8, borderpad 0.5).
# Rows / handle / border: tight vs defaults; column gap is 2× the prior compact value (still below default 2.0).
LEGEND_LABELSPACING: float = 0.125
LEGEND_COLUMNSPACING: float = 1.0
LEGEND_HANDLETEXTPAD: float = 0.4
LEGEND_BORDERPAD: float = 0.25

# Single axes (one specimen loop / one diagnostic panel).
SINGLE_FIGSIZE_IN: tuple[float, float] = (2.5, 2.25)

# Each grid cell in montages / multi-panel figures.
GRID_AX_W_IN: float = 2.5
GRID_AX_H_IN: float = 2.25

# Two rows × one column (time histories, digitized raw/filtered vs resampled drive).
TWO_STACKED_FIG_W_IN: float = 6.0

SAVE_DPI: int = 300

# Experimental / test hysteresis and clouds (experimental overlays solid ``'-'``; numerical ``'--'``).
COLOR_EXPERIMENTAL: str = "#B0B0B0"

# Numerical / simulated in calibration overlays (pair with ``COLOR_EXPERIMENTAL``):
# ``COLOR_NUMERICAL_COHORT`` — train; ``COLOR_NUMERICAL_COHORT_AUX`` — validation
# (zero-weight path rows, scatter-cloud cohort panels).
COLOR_NUMERICAL_COHORT: str = "#001F3F"
COLOR_NUMERICAL_COHORT_AUX: str = "#FF0E00"

# Full data-area rectangle (four spines); separate from ``Axes.patch`` fill outline.
AXES_SPINE_COLOR: str = "black"
AXES_SPINE_LINEWIDTH: float = 0.4


def style_major_tick_lines(
    ax: plt.Axes,
    *,
    linewidth: float | None = None,
    color: str | None = None,
) -> None:
    """Set major tick **line** color and width to match spines (tick labels unchanged)."""
    lw = AXES_SPINE_LINEWIDTH if linewidth is None else linewidth
    col = AXES_SPINE_COLOR if color is None else color
    for axis in (ax.xaxis, ax.yaxis):
        for line in axis.get_ticklines(minor=False):
            line.set_color(col)
            line.set_linewidth(lw)


def style_axes_spines_and_ticks(
    ax: plt.Axes,
    *,
    linewidth: float | None = None,
    color: str | None = None,
) -> None:
    """Spine and tick lines on the data plane: hide axes-patch border, four black spines, inward ticks (labels bottom+left)."""
    lw = AXES_SPINE_LINEWIDTH if linewidth is None else linewidth
    col = AXES_SPINE_COLOR if color is None else color
    ax.patch.set_linewidth(0.0)
    ax.patch.set_edgecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(col)
        spine.set_linewidth(lw)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    style_major_tick_lines(ax, linewidth=lw, color=col)


def configure_matplotlib_style() -> None:
    """Apply consistent font sizes via rcParams (call after ``sns.set_theme()`` if using seaborn)."""
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE_PT,
            "axes.titlesize": PLOT_FONT_SIZE_PT,
            "axes.labelsize": PLOT_FONT_SIZE_PT,
            "xtick.labelsize": PLOT_FONT_SIZE_PT,
            "ytick.labelsize": PLOT_FONT_SIZE_PT,
            # Slightly below axis/tick size so legends read a bit lighter on the page.
            "legend.fontsize": PLOT_FONT_SIZE_PT - 0.75,
            "figure.titlesize": PLOT_FONT_SIZE_PT,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": AXES_SPINE_LINEWIDTH,
            "ytick.major.width": AXES_SPINE_LINEWIDTH,
            "legend.labelspacing": LEGEND_LABELSPACING,
            "legend.columnspacing": LEGEND_COLUMNSPACING,
            "legend.handletextpad": LEGEND_HANDLETEXTPAD,
            "legend.borderpad": LEGEND_BORDERPAD,
        }
    )


def grid_montage_rcparams() -> dict[str, float]:
    """rcParams for multi-panel figures: ~2× base font so subplots stay readable."""
    fs = PLOT_FONT_SIZE_GRID_MONTAGE_PT
    return {
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": LEGEND_FONT_SIZE_GRID_MONTAGE_PT,
        "figure.titlesize": fs,
    }


def b_vs_geometry_rcparams() -> dict[str, float]:
    """rcParams for apparent-b vs geometry montages (``B_VS_GEOMETRY_FONT_SCALE`` × grid montage type)."""
    s = B_VS_GEOMETRY_FONT_SCALE
    fs = PLOT_FONT_SIZE_GRID_MONTAGE_PT * s
    return {
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": LEGEND_FONT_SIZE_GRID_MONTAGE_PT * s,
        "figure.titlesize": fs,
    }


def figsize_for_grid(nrow: int, ncol: int) -> tuple[float, float]:
    """Total figure width/height for an ``nrow`` × ``ncol`` axes grid (use with ``layout='constrained'``)."""
    nr = max(1, int(nrow))
    nc = max(1, int(ncol))
    return (GRID_AX_W_IN * nc, GRID_AX_H_IN * nr)


def figsize_two_stacked_axes() -> tuple[float, float]:
    """Two subplot rows, one column (e.g. force + deformation vs index)."""
    return (TWO_STACKED_FIG_W_IN, GRID_AX_H_IN * 2)
