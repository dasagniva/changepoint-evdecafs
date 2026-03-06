"""Matplotlib style configuration for publication-quality figures.

Call ``apply_style()`` once at the start of any script or notebook that
generates figures to apply consistent rcParams.
"""

import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt


# Colour palette used consistently across all figures
PALETTE = {
    "series": "#888888",       # raw time-series line
    "mean": "#d62728",         # estimated piecewise-constant mean (red)
    "detected_cp": "#222222",  # detected changepoint (dashed black)
    "true_cp": "#1f77b4",      # ground-truth changepoint (blue)
    "outlier": "#2ca02c",      # outlier marker (green)
    "class0": "#ff7f0e",       # recoiled class
    "class1": "#1f77b4",       # sustained class
}

# Ordered list for multi-line plots
COLOR_CYCLE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def apply_style() -> None:
    """Apply publication-quality rcParams to the current Matplotlib session.

    Sets serif fonts, attempts LaTeX rendering (falls back to mathtext),
    300 DPI, tight layout, and a consistent colour cycle.

    Examples
    --------
    >>> from src.visualization.style import apply_style
    >>> apply_style()
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    """
    # Enable LaTeX rendering only when latex is available on PATH
    if shutil.which("latex") is not None:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        mpl.rcParams["text.usetex"] = False

    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,

            # Figure
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,

            # Lines
            "lines.linewidth": 1.2,
            "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),

            # Grid
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
