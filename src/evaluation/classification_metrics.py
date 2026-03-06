"""Classification metrics for Phase II evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute a full suite of binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels, shape ``(n,)``.
    y_pred:
        Predicted binary labels, shape ``(n,)``.
    y_proba:
        Predicted probabilities.  Accepts shape ``(n,)`` (positive-class
        probability) or ``(n, 2)`` (full probability matrix).  Required for
        AUC-ROC.  If ``None`` or only one class is present, AUC-ROC is
        returned as ``np.nan``.

    Returns
    -------
    dict with keys:
        ``balanced_accuracy``, ``mcc``, ``auc_roc``,
        ``confusion_matrix``, ``classification_report``,
        ``f1_class0``, ``f1_class1``,
        ``precision_class0``, ``precision_class1``,
        ``recall_class0``, ``recall_class1``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report_str = classification_report(y_true, y_pred, zero_division=0)

    # Per-class metrics from the report dict
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    def _cls_metric(label: int, metric: str) -> float:
        """Look up a per-class metric; try both str and int keys."""
        row = report_dict.get(str(label), report_dict.get(label, {}))
        return float(row.get(metric, 0.0))

    f1_0 = _cls_metric(0, "f1-score")
    f1_1 = _cls_metric(1, "f1-score")
    prec_0 = _cls_metric(0, "precision")
    prec_1 = _cls_metric(1, "precision")
    rec_0 = _cls_metric(0, "recall")
    rec_1 = _cls_metric(1, "recall")

    # AUC-ROC
    auc_roc = np.nan
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            y_score = np.asarray(y_proba)
            if y_score.ndim == 2:
                y_score = y_score[:, 1]
            auc_roc = float(roc_auc_score(y_true, y_score))
        except Exception as exc:
            logger.warning("AUC-ROC computation failed: %s", exc)

    metrics = {
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "classification_report": report_str,
        "f1_class0": f1_0,
        "f1_class1": f1_1,
        "precision_class0": prec_0,
        "precision_class1": prec_1,
        "recall_class0": rec_0,
        "recall_class1": rec_1,
    }

    logger.info(
        "Metrics — Bal.Acc=%.4f  MCC=%.4f  AUC-ROC=%s",
        bal_acc,
        mcc,
        f"{auc_roc:.4f}" if not np.isnan(auc_roc) else "N/A",
    )
    return metrics
