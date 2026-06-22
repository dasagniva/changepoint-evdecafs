"""Time-series run charts with changepoint overlays."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging_config import setup_logger
from src.visualization.style import PALETTE

logger = setup_logger(__name__)


def plot_run_chart(
    y: np.ndarray,
    detected_cps: np.ndarray,
    true_cps: np.ndarray | None,
    means: np.ndarray | None,
    title: str = "",
    save_path: str | Path | None = None,
    outlier_indices: np.ndarray | None = None,
) -> None:
    """Plot a time series with detected and true changepoints.

    Parameters
    ----------
    y:
        Observed time series.
    detected_cps:
        Indices of detected changepoints — drawn as dashed black verticals.
    true_cps:
        Indices of true changepoints — drawn as solid blue verticals.
        Pass ``None`` to omit.
    means:
        Piecewise-constant mean estimate — drawn as a red step overlay.
        Pass ``None`` to omit.
    title:
        Figure title.
    save_path:
        If provided, the figure is saved at 300 DPI (format inferred from
        the file extension).
    outlier_indices:
        If provided, drawn as solid green verticals.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    t = np.arange(len(y))
    ax.plot(t, y, color=PALETTE["series"], lw=0.7, alpha=0.85, label="observed", zorder=1)

    if means is not None:
        ax.plot(t, means, color=PALETTE["mean"], lw=1.6, label="mean estimate", zorder=3)

    # True changepoints
    if true_cps is not None and len(true_cps) > 0:
        for i, cp in enumerate(true_cps):
            ax.axvline(
                cp,
                color=PALETTE["true_cp"],
                lw=1.0,
                ls="-",
                alpha=0.8,
                label="true CP" if i == 0 else None,
                zorder=2,
            )

    # Detected changepoints
    if len(detected_cps) > 0:
        for i, cp in enumerate(detected_cps):
            ax.axvline(
                cp,
                color=PALETTE["detected_cp"],
                lw=0.9,
                ls="--",
                alpha=0.7,
                label="detected CP" if i == 0 else None,
                zorder=2,
            )

    # Outlier markers
    if outlier_indices is not None and len(outlier_indices) > 0:
        for i, idx in enumerate(outlier_indices):
            ax.axvline(
                idx,
                color=PALETTE["outlier"],
                lw=0.8,
                ls=":",
                alpha=0.6,
                label="outlier" if i == 0 else None,
                zorder=2,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

    _save_or_show(fig, save_path)


def plot_changepoint_comparison(
    y: np.ndarray,
    detectors_dict: dict[str, np.ndarray],
    true_cps: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Grid of subplots comparing multiple detectors on the same series.

    Each subplot shows the full series with true changepoints (solid blue)
    and one detector's detected changepoints (dashed black).

    Parameters
    ----------
    y:
        Observed time series.
    detectors_dict:
        ``{detector_name: detected_changepoints_array}``
    true_cps:
        Ground-truth changepoint indices.
    save_path:
        If provided, the figure is saved at 300 DPI.
    """
    n_det = len(detectors_dict)
    n_cols = min(n_det, 2)
    n_rows = math.ceil(n_det / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    t = np.arange(len(y))

    for ax_idx, (name, cps) in enumerate(detectors_dict.items()):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]

        ax.plot(t, y, color=PALETTE["series"], lw=0.7, alpha=0.85)

        for i, cp in enumerate(true_cps):
            ax.axvline(cp, color=PALETTE["true_cp"], lw=1.0, ls="-", alpha=0.7,
                       label="true CP" if i == 0 else None)
        for i, cp in enumerate(np.asarray(cps)):
            ax.axvline(cp, color=PALETTE["detected_cp"], lw=0.9, ls="--", alpha=0.6,
                       label="detected" if i == 0 else None)

        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
        ax.set_xlabel("Time")

    # Hide any unused subplots
    for ax_idx in range(n_det, n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Changepoint detector comparison", fontsize=12)
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
