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


def build_gru_classifier(
    input_dim: int,
    hidden_dim: int = 32,
    dropout: float = 0.3,
    epochs: int = 100,
    lr: float = 0.001,
):
    """Build a GRU-based classifier (sklearn-compatible).

    Uses PyTorch if available; falls back to MLPClassifier otherwise.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        GRU hidden units.
    dropout:
        Dropout probability.
    epochs:
        Training epochs.
    lr:
        Adam learning rate.
    """
    try:
        import torch
        import torch.nn as nn
        from sklearn.base import BaseEstimator, ClassifierMixin

        class GRUClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.hidden_dim = hidden_dim
                self.dropout = dropout
                self.epochs = epochs
                self.lr = lr
                self.input_dim = input_dim
                self.model_ = None
                self.classes_ = np.array([0, 1])

            def _build_model(self):
                return nn.Sequential(
                    nn.GRU(self.input_dim, self.hidden_dim,
                           batch_first=True, dropout=0.0),
                )

            def fit(self, X, y):
                X = np.asarray(X, dtype=np.float32)
                y = np.asarray(y, dtype=np.int64)
                # Shape: (n, 1, d) — treat feature vector as 1-step sequence
                X_t = torch.tensor(X[:, None, :])
                y_t = torch.tensor(y)

                class GRUNet(nn.Module):
                    def __init__(self, in_d, hid, drop):
                        super().__init__()
                        self.gru = nn.GRU(in_d, hid, batch_first=True)
                        self.drop = nn.Dropout(drop)
                        self.fc = nn.Linear(hid, 2)

                    def forward(self, x):
                        out, _ = self.gru(x)
                        out = self.drop(out[:, -1, :])
                        return self.fc(out)

                self.model_ = GRUNet(self.input_dim, self.hidden_dim, self.dropout)
                counts = np.bincount(y, minlength=2)
                weights = torch.tensor(
                    1.0 / (counts + 1e-6), dtype=torch.float32
                )
                criterion = nn.CrossEntropyLoss(weight=weights)
                optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

                self.model_.train()
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    logits = self.model_(X_t)
                    loss = criterion(logits, y_t)
                    loss.backward()
                    optimizer.step()
                return self

            def predict_proba(self, X):
                X = np.asarray(X, dtype=np.float32)
                X_t = torch.tensor(X[:, None, :])
                self.model_.eval()
                with torch.no_grad():
                    logits = self.model_(X_t)
                    proba = torch.softmax(logits, dim=1).numpy()
                return proba

            def predict(self, X):
                return self.predict_proba(X).argmax(axis=1)

        return GRUClassifier()

    except ImportError:
        logger.warning(
            "PyTorch not available — GRU classifier falls back to MLPClassifier."
        )
        return MLPClassifier(hidden_layer_sizes=(hidden_dim,), max_iter=200)


def get_baselines(params: dict, input_dim: int = 4) -> dict:
    """Instantiate all baseline classifiers.

    Parameters
    ----------
    params:
        The ``baselines`` sub-dict from ``config/params.yaml``.
    input_dim:
        Number of input features (passed to GRU classifier).

    Returns
    -------
    dict mapping classifier name → unfitted sklearn-compatible estimator.
        Keys: ``'Logistic Regression'``, ``'Isolation Forest'``,
        ``'One-Class SVM'``, ``'Feedforward NN'``, ``'GRU (RNN)'``.
    """
    lr_C_range = params.get("lr_C_range", [0.01, 0.1, 1, 10])
    if_contamination = params.get("if_contamination", "auto")
    ocsvm_kernel = params.get("ocsvm_kernel", "rbf")
    fnn_hidden = tuple(params.get("fnn_hidden", [64, 32]))
    fnn_epochs = int(params.get("fnn_epochs", 100))
    gru_hidden = int(params.get("gru_hidden_dim", 32))
    gru_dropout = float(params.get("gru_dropout", 0.3))
    gru_epochs = int(params.get("gru_epochs", 100))
    gru_lr = float(params.get("gru_lr", 0.001))

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
        "GRU (RNN)": build_gru_classifier(
            input_dim=input_dim,
            hidden_dim=gru_hidden,
            dropout=gru_dropout,
            epochs=gru_epochs,
            lr=gru_lr,
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
