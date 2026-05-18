"""
ts_style.py
-----------
Economic Order matplotlib stylesheet for the Time Series Analysis textbook.
Import this module at the top of every chapter's first code cell:

    from ts_style import *

This registers the EO color palette, sets global rcParams, and exposes
named colors and convenience helpers for figures.

Typography (EO brand guidelines):
  - Palatino Linotype : body text — tick labels, legends, annotations
  - Calibri           : display text — axis titles, figure suptitles
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# EO Brand Color Palette
# ---------------------------------------------------------------------------

EO_CHARCOAL   = "#36454F"   # Primary: axes, text, spines
EO_COPPER     = "#B87333"   # Accent 1: first data series, highlights
EO_SAGE       = "#87A96B"   # Accent 2: second data series
EO_SKYBLUE    = "#5B9BD5"   # Accent 3: third data series
EO_TERRACOTTA = "#D4745E"   # Accent 4: fourth data series / callouts
EO_LAVENDER   = "#8E7AB5"   # Accent 5: fifth data series

# Ordered cycle for multi-series plots
EO_COLORS = [
    EO_COPPER,
    EO_SKYBLUE,
    EO_SAGE,
    EO_TERRACOTTA,
    EO_LAVENDER,
    EO_CHARCOAL,
]

# ---------------------------------------------------------------------------
# Global rcParams — applied on import
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    # ── Figure ────────────────────────────────────────────────────────────────
    "figure.figsize":        (9, 4),
    "figure.dpi":            150,
    "figure.facecolor":      "white",
    "figure.edgecolor":      "white",

    # ── Axes ──────────────────────────────────────────────────────────────────
    "axes.facecolor":        "white",
    "axes.edgecolor":        EO_CHARCOAL,
    "axes.linewidth":        0.8,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.titlecolor":       EO_CHARCOAL,
    "axes.titlelocation":    "left",          # flush-left titles, editorial feel
    "axes.labelsize":        11,
    "axes.labelcolor":       EO_CHARCOAL,
    "axes.labelweight":      "normal",
    "axes.prop_cycle":       mpl.cycler(color=EO_COLORS),

    # ── Grid ──────────────────────────────────────────────────────────────────
    "grid.color":            "#E5E5E5",
    "grid.linewidth":        0.6,
    "grid.linestyle":        "--",
    "grid.alpha":            0.8,

    # ── Ticks ─────────────────────────────────────────────────────────────────
    "xtick.color":           EO_CHARCOAL,
    "ytick.color":           EO_CHARCOAL,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      4,
    "ytick.major.size":      4,

    # ── Lines ─────────────────────────────────────────────────────────────────
    "lines.linewidth":       1.8,
    "lines.solid_capstyle":  "round",

    # ── Legend ────────────────────────────────────────────────────────────────
    "legend.frameon":        True,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#CCCCCC",
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,

    # ── Typography (EO brand) ─────────────────────────────────────────────────
    # Palatino Linotype for body-level text (ticks, legend, annotations)
    "font.family":           "serif",
    "font.serif":            ["Palatino Linotype", "Palatino", "Georgia",
                              "DejaVu Serif"],
    "font.size":             10,
    "text.color":            EO_CHARCOAL,

    # Calibri for display-level text (titles, axis labels)
    "axes.titlefamily":      "Calibri",
    "axes.labelfamily":      "Calibri",

    # ── Saving ────────────────────────────────────────────────────────────────
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "white",
})


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def eo_fig(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create a figure and axes with EO styling already applied.
    Returns (fig, ax) for single-panel plots, or (fig, axes) for multi-panel.

    Example
    -------
    fig, ax = eo_fig()
    ax.plot(x, y)
    """
    if figsize is None:
        figsize = (9, 4) if (nrows == 1 and ncols == 1) else (9, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def eo_suptitle(fig, title, **kwargs):
    """
    Add a figure-level suptitle in Verdana Bold, EO Charcoal.
    Wraps fig.suptitle() with consistent EO defaults.

    Example
    -------
    eo_suptitle(fig, "US Real GDP, 1960–2024")
    """
    defaults = dict(
        fontsize=13,
        fontweight="bold",
        color=EO_CHARCOAL,
        fontfamily="Calibri",
        y=1.01,
    )
    defaults.update(kwargs)
    fig.suptitle(title, **defaults)


def label_axes(ax, title=None, xlabel=None, ylabel=None):
    """
    Apply title, x-label, and y-label to an axes object in one call.
    All text respects the rcParams font settings.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def despine(ax):
    """
    Remove top and right spines from an axes object.
    Redundant with the rcParams default but useful for axes created
    outside the standard plt.subplots() machinery.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)