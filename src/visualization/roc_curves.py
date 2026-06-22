"""ROC curve plots for Phase II classifier comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.utils.logging_config import setup_logger
from src.visualization.style import COLOR_CYCLE

logger = setup_logger(__name__)


def plot_roc_curves(
    classifiers_results: dict[str, dict],
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC curves for multiple classifiers on shared axes.

    Parameters
    ----------
    classifiers_results:
        ``{classifier_name: {'y_true': array, 'y_proba': array}}``
        where ``y_proba`` is the probability of the **positive** class
        (shape ``(n,)``).
    save_path:
        If provided, the figure is saved at 300 DPI.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, res), color in zip(
        classifiers_results.items(), COLOR_CYCLE * 10
    ):
        y_true = np.asarray(res["y_true"])
        y_proba = np.asarray(res["y_proba"])
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        if len(np.unique(y_true)) < 2:
            logger.warning("Skipping ROC for %s: only one class present.", name)
            continue

        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{name} (AUC = {roc_auc:.3f})")
        except Exception as exc:
            logger.warning("ROC curve failed for %s: %s", name, exc)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", label="random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("ROC figure saved: %s", save_path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
