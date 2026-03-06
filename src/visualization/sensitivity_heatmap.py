"""Sensitivity heatmap visualisation for the cost-ratio analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def plot_sensitivity_heatmap(
    rankings_df: pd.DataFrame,
    raw_values_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Plot cost-ratio sensitivity heatmaps.

    Produces two figures:

    1. **Best-detector heatmap** — each cell shows the name of the
       rank-1 detector for that ``(cF, cD)`` combination, colour-coded
       by detector identity.
    2. **R-tilde heatmap** — each cell shows the winning detector's
       censored risk value.

    Parameters
    ----------
    rankings_df:
        Multi-index ``(cF, cD)``, columns = detector names, values = ranks.
    raw_values_df:
        Same index/columns, values = R-tilde floats.
    save_path:
        If provided, both figures are saved with ``_ranks`` / ``_values``
        suffixes inserted before the file extension, at 300 DPI.
    """
    cF_vals = rankings_df.index.get_level_values("cF").unique().tolist()
    cD_vals = rankings_df.index.get_level_values("cD").unique().tolist()
    detector_names = rankings_df.columns.tolist()

    # ---- Figure 1: best-detector name per cell ----
    # Encode detector name as integer for colour mapping
    best_detector = rankings_df.idxmin(axis=1)  # name of rank-1 detector per (cF,cD)
    name_to_int = {n: i for i, n in enumerate(detector_names)}

    winner_matrix = np.full((len(cF_vals), len(cD_vals)), np.nan)
    for (cF, cD), winner in best_detector.items():
        ri = cF_vals.index(cF)
        ci = cD_vals.index(cD)
        winner_matrix[ri, ci] = name_to_int[winner]

    fig1, ax1 = plt.subplots(figsize=(max(4, len(cD_vals) * 1.2), max(3, len(cF_vals) * 1.0)))
    cmap = plt.colormaps["tab10"].resampled(len(detector_names))
    im = ax1.imshow(winner_matrix, cmap=cmap, vmin=-0.5, vmax=len(detector_names) - 0.5,
                    aspect="auto")

    # Annotate cells with winner name
    for ri, cF in enumerate(cF_vals):
        for ci, cD in enumerate(cD_vals):
            winner = best_detector.get((cF, cD), "")
            ax1.text(ci, ri, winner, ha="center", va="center", fontsize=8,
                     color="white" if name_to_int.get(winner, 0) % 2 == 0 else "black")

    cbar = fig1.colorbar(im, ax=ax1, ticks=list(range(len(detector_names))))
    cbar.set_ticklabels(detector_names)
    ax1.set_xticks(range(len(cD_vals)))
    ax1.set_xticklabels([str(v) for v in cD_vals])
    ax1.set_yticks(range(len(cF_vals)))
    ax1.set_yticklabels([str(v) for v in cF_vals])
    ax1.set_xlabel("$c_D$ (delay cost)")
    ax1.set_ylabel("$c_F$ (FP cost)")
    ax1.set_title("Best detector by cost-ratio (ranks)")
    _save_or_show(fig1, save_path, suffix="_ranks")

    # ---- Figure 2: winning R-tilde value per cell ----
    rtilde_matrix = np.full((len(cF_vals), len(cD_vals)), np.nan)
    for (cF, cD), winner in best_detector.items():
        ri = cF_vals.index(cF)
        ci = cD_vals.index(cD)
        rtilde_matrix[ri, ci] = raw_values_df.loc[(cF, cD), winner]

    # Replace inf with NaN for display
    rtilde_display = pd.DataFrame(
        rtilde_matrix,
        index=[str(v) for v in cF_vals],
        columns=[str(v) for v in cD_vals],
    )

    fig2, ax2 = plt.subplots(figsize=(max(4, len(cD_vals) * 1.2), max(3, len(cF_vals) * 1.0)))
    sns.heatmap(
        rtilde_display,
        ax=ax2,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": r"$\tilde{R}$ (censored risk)"},
    )
    ax2.set_xlabel("$c_D$ (delay cost)")
    ax2.set_ylabel("$c_F$ (FP cost)")
    ax2.set_title(r"Best-detector censored risk $\tilde{R}$")
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
        plt.tight_layout()
        plt.show()
