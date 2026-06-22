"""Sensitivity heatmap visualisation for the cost-ratio analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Distinct colours for up to 8 detectors
_DETECTOR_COLORS = [
    "#2166AC",  # blue
    "#D32F2F",  # red
    "#388E3C",  # green
    "#F57C00",  # orange
    "#7B1FA2",  # purple
    "#00838F",  # teal
    "#795548",  # brown
    "#546E7A",  # grey-blue
]


def plot_sensitivity_heatmap(
    rankings_df: pd.DataFrame,
    raw_values_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Plot publication-quality cost-ratio sensitivity heatmaps.

    Produces two figures saved with ``_ranks`` / ``_values`` suffixes:

    1. **Best-detector heatmap** — cells colour-coded by winning detector;
       no text overlay.  A legend outside the axes names the colours.
    2. **R-tilde values heatmap** — seaborn heatmap with small numeric
       annotations showing the winning detector's R-tilde and short name.

    Parameters
    ----------
    rankings_df:
        Multi-index ``(cF, cD)``, columns = detector names, values = ranks.
    raw_values_df:
        Same index/columns, values = R-tilde floats.
    save_path:
        Base path.  Suffixes ``_ranks`` / ``_values`` are inserted before
        the extension.  If ``None``, figures are shown interactively.
    """
    cF_vals = sorted(rankings_df.index.get_level_values("cF").unique().tolist())
    cD_vals = sorted(rankings_df.index.get_level_values("cD").unique().tolist())
    detector_names = rankings_df.columns.tolist()

    # ---- Build winner matrix (int index into detector_names) ----
    winner_matrix = np.full((len(cF_vals), len(cD_vals)), -1, dtype=int)
    for i, cF in enumerate(cF_vals):
        for j, cD in enumerate(cD_vals):
            try:
                row = rankings_df.loc[(cF, cD)]
                winner_matrix[i, j] = int(row.values.argmin())
            except KeyError:
                pass

    # Which detectors actually appear as winners?
    winning_indices = sorted(set(winner_matrix.flatten()) - {-1})

    # ---- Figure 1: Categorical best-detector heatmap ----
    colors = _DETECTOR_COLORS[:len(detector_names)]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, len(detector_names) + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_w = max(4.5, len(cD_vals) * 1.4 + 2.5)
    fig_h = max(3.0, len(cF_vals) * 1.0 + 1.2)
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    ax1.imshow(
        winner_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="lower",
    )

    ax1.set_xticks(range(len(cD_vals)))
    ax1.set_xticklabels([str(v) for v in cD_vals], fontsize=11)
    ax1.set_yticks(range(len(cF_vals)))
    ax1.set_yticklabels([str(v) for v in cF_vals], fontsize=11)
    ax1.set_xlabel(r"$c_D$ (delay cost)", fontsize=13)
    ax1.set_ylabel(r"$c_F$ (false positive cost)", fontsize=13)
    ax1.set_title("Best Detector by Cost Ratio", fontsize=14, fontweight="bold")

    # Legend: only show detectors that actually win at least one cell
    legend_elements = [
        mpatches.Patch(facecolor=colors[i], label=detector_names[i])
        for i in winning_indices
    ]
    ax1.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
        title="Detector",
        title_fontsize=9,
    )

    plt.tight_layout()
    _save_or_show(fig1, save_path, suffix="_ranks")

    # ---- Build best-R-tilde matrix for Figure 2 ----
    best_rtilde = np.full((len(cF_vals), len(cD_vals)), np.nan)
    annot_text = np.empty((len(cF_vals), len(cD_vals)), dtype=object)

    for i, cF in enumerate(cF_vals):
        for j, cD in enumerate(cD_vals):
            try:
                row = raw_values_df.loc[(cF, cD)]
                finite_row = row.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite_row) > 0:
                    best_val = float(finite_row.min())
                    best_name = str(finite_row.idxmin())
                    best_rtilde[i, j] = best_val
                    annot_text[i, j] = f"{best_val:.4f}"
                else:
                    annot_text[i, j] = "—"
            except KeyError:
                annot_text[i, j] = "—"

    rtilde_df = pd.DataFrame(
        best_rtilde,
        index=[str(v) for v in cF_vals],
        columns=[str(v) for v in cD_vals],
    )

    # ---- Figure 2: R-tilde values heatmap ----
    fig2_w = max(5.0, len(cD_vals) * 1.6 + 1.5)
    fig2_h = max(3.5, len(cF_vals) * 1.2 + 1.2)
    fig2, ax2 = plt.subplots(figsize=(fig2_w, fig2_h))

    sns.heatmap(
        rtilde_df,
        ax=ax2,
        annot=annot_text,
        fmt="",
        cmap="YlOrRd_r",   # lower R-tilde = better = more yellow/green
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": r"$\tilde{R}$ (lower = better)"},
        annot_kws={"fontsize": 8},
    )
    ax2.set_xlabel(r"$c_D$ (delay cost)", fontsize=13)
    ax2.set_ylabel(r"$c_F$ (false positive cost)", fontsize=13)
    ax2.set_title(r"Best $\tilde{R}$ by Cost Ratio", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()

    _save_or_show(fig2, save_path, suffix="_values")


def _save_or_show(
    fig: plt.Figure,
    save_path: str | Path | None,
    suffix: str = "",
) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        stem = save_path.stem + suffix
        out = save_path.with_name(stem + save_path.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Heatmap saved: %s", out)
        plt.close(fig)
    else:
        try:
            fig.tight_layout()
        except Exception:
            pass
        plt.show()
