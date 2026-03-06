"""Baseline classifiers for Phase II comparison.

Provides Logistic Regression, Isolation Forest, One-Class SVM, and a
Feedforward NN (sklearn MLPClassifier), along with a unified training and
evaluation harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM

from src.evaluation.classification_metrics import compute_classification_metrics
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Classifiers that output anomaly scores instead of class probabilities
_ANOMALY_DETECTORS = {"Isolation Forest", "One-Class SVM"}


def get_baselines(params: dict) -> dict:
    """Instantiate all baseline classifiers.

    Parameters
    ----------
    params:
        The ``baselines`` sub-dict from ``config/params.yaml``.

    Returns
    -------
    dict mapping classifier name → unfitted sklearn-compatible estimator.
        Keys: ``'Logistic Regression'``, ``'Isolation Forest'``,
        ``'One-Class SVM'``, ``'Feedforward NN'``.
    """
    lr_C_range = params.get("lr_C_range", [0.01, 0.1, 1, 10])
    if_contamination = params.get("if_contamination", "auto")
    ocsvm_kernel = params.get("ocsvm_kernel", "rbf")
    fnn_hidden = tuple(params.get("fnn_hidden", [64, 32]))
    fnn_epochs = int(params.get("fnn_epochs", 100))

    lr_base = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    lr = GridSearchCV(
        lr_base,
        param_grid={"C": lr_C_range},
        cv=5,
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": lr,
        "Isolation Forest": IsolationForest(
            contamination=if_contamination, random_state=42
        ),
        "One-Class SVM": OneClassSVM(kernel=ocsvm_kernel),
        "Feedforward NN": MLPClassifier(
            hidden_layer_sizes=fnn_hidden,
            max_iter=fnn_epochs * 5,  # allow enough iterations
            early_stopping=True,
            random_state=42,
        ),
    }


def train_and_evaluate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baselines_dict: dict,
    fpnn,
) -> pd.DataFrame:
    """Train all classifiers and evaluate on the test set.

    For Isolation Forest and One-Class SVM, raw anomaly labels (−1/+1) are
    remapped: inlier (+1) → sustained (1), outlier (−1) → recoiled (0).
    Their ``decision_function`` output is used as a probability proxy for
    AUC-ROC.

    Parameters
    ----------
    X_train, y_train:
        SMOTE-balanced training data.
    X_test, y_test:
        Held-out test data with ground-truth labels.
    baselines_dict:
        Dict of unfitted estimators from :func:`get_baselines`.
    fpnn:
        Fitted :class:`~src.phase2.fpnn.FourierPNN` instance.

    Returns
    -------
    pd.DataFrame
        Rows = classifiers, columns = metric names.
        Metrics: ``Balanced Accuracy``, ``MCC``, ``F1 (class 0)``,
        ``F1 (class 1)``, ``AUC-ROC``.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    rows = {}

    # ---- FPNN (already fitted) ----
    logger.info("Evaluating FPNN...")
    y_proba_fpnn = fpnn.predict_proba(X_test)
    y_pred_fpnn = fpnn.predict(X_test)
    m = compute_classification_metrics(y_test, y_pred_fpnn, y_proba_fpnn)
    rows["FPNN"] = _metrics_to_row(m)

    # ---- Baselines ----
    for name, clf in baselines_dict.items():
        logger.info("Training %s...", name)
        try:
            if name in _ANOMALY_DETECTORS:
                # One-class detectors: fit on all training data (unsupervised)
                clf.fit(X_train)
                raw_pred = clf.predict(X_test)
                # Remap: +1 (inlier) → 1 (sustained), -1 (outlier) → 0 (recoiled)
                y_pred = (raw_pred == 1).astype(int)
                # Use decision_function as probability proxy
                scores = clf.decision_function(X_test)
                # Normalise to [0, 1] as a rough probability substitute
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    y_proba = (scores - s_min) / (s_max - s_min)
                else:
                    y_proba = np.full(len(scores), 0.5)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                # Get probability estimates if available
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_test)
                elif hasattr(clf, "decision_function"):
                    scores = clf.decision_function(X_test)
                    s_min, s_max = scores.min(), scores.max()
                    y_proba = (
                        (scores - s_min) / (s_max - s_min)
                        if s_max > s_min
                        else np.full(len(scores), 0.5)
                    )
                else:
                    y_proba = None

            m = compute_classification_metrics(y_test, y_pred, y_proba)
            rows[name] = _metrics_to_row(m)
            logger.info(
                "%s — Bal.Acc=%.4f  MCC=%.4f  AUC-ROC=%s",
                name,
                m["balanced_accuracy"],
                m["mcc"],
                f"{m['auc_roc']:.4f}" if not np.isnan(m["auc_roc"]) else "N/A",
            )
        except Exception as exc:
            logger.error("Failed to train/evaluate %s: %s", name, exc)
            rows[name] = {
                "Balanced Accuracy": np.nan,
                "MCC": np.nan,
                "F1 (class 0)": np.nan,
                "F1 (class 1)": np.nan,
                "AUC-ROC": np.nan,
            }

    return pd.DataFrame(rows).T


def _metrics_to_row(m: dict) -> dict:
    return {
        "Balanced Accuracy": m["balanced_accuracy"],
        "MCC": m["mcc"],
        "F1 (class 0)": m["f1_class0"],
        "F1 (class 1)": m["f1_class1"],
        "AUC-ROC": m["auc_roc"],
    }
