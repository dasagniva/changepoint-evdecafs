"""Tests for FPNN, baselines, and classification metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase2.fpnn import FourierPNN
from src.phase2.baselines import get_baselines, train_and_evaluate_all
from src.evaluation.classification_metrics import compute_classification_metrics

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_dataset(n=120, n_features=4, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n0, n1 = n // 2, n - n // 2
    X0 = rng.normal(0, 1, (n0, n_features))
    X1 = rng.normal(sep, 1, (n1, n_features))
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


def _split(X, y, frac=0.7):
    n = int(len(X) * frac)
    return X[:n], y[:n], X[n:], y[n:]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:
    def setup_method(self):
        self.y_true = np.array([0, 0, 1, 1, 0, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0])

    def test_returns_all_keys(self):
        m = compute_classification_metrics(self.y_true, self.y_pred)
        for key in [
            "balanced_accuracy", "mcc", "auc_roc", "confusion_matrix",
            "classification_report", "f1_class0", "f1_class1",
            "precision_class0", "precision_class1", "recall_class0", "recall_class1",
        ]:
            assert key in m

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1])
        m = compute_classification_metrics(y, y)
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)
        assert m["f1_class0"] == pytest.approx(1.0)
        assert m["f1_class1"] == pytest.approx(1.0)

    def test_auc_roc_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_classification_metrics(y_true, y_pred, y_proba)
        assert not np.isnan(m["auc_roc"])
        assert 0.0 <= m["auc_roc"] <= 1.0

    def test_auc_roc_with_2d_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.column_stack([
            np.array([0.9, 0.8, 0.2, 0.1]),
            np.array([0.1, 0.2, 0.8, 0.9]),
        ])
        m = compute_classification_metrics(y_true, y_pred, y_proba)
        assert not np.isnan(m["auc_roc"])

    def test_auc_roc_none_when_proba_not_given(self):
        m = compute_classification_metrics(self.y_true, self.y_pred, y_proba=None)
        assert np.isnan(m["auc_roc"])

    def test_single_class_auc_is_nan(self):
        y = np.zeros(10, dtype=int)
        m = compute_classification_metrics(y, y, np.zeros(10))
        assert np.isnan(m["auc_roc"])

    def test_confusion_matrix_shape(self):
        m = compute_classification_metrics(self.y_true, self.y_pred)
        assert m["confusion_matrix"].shape == (2, 2)

    def test_balanced_accuracy_range(self):
        m = compute_classification_metrics(self.y_true, self.y_pred)
        assert 0.0 <= m["balanced_accuracy"] <= 1.0

    def test_f1_range(self):
        m = compute_classification_metrics(self.y_true, self.y_pred)
        assert 0.0 <= m["f1_class0"] <= 1.0
        assert 0.0 <= m["f1_class1"] <= 1.0


# ---------------------------------------------------------------------------
# FourierPNN
# ---------------------------------------------------------------------------

class TestFourierPNN:
    def setup_method(self):
        self.X, self.y = _make_binary_dataset(n=100, sep=3.0)
        self.X_tr, self.y_tr, self.X_te, self.y_te = _split(self.X, self.y)

    def _fit(self, J=5):
        fpnn = FourierPNN(J=J, scaling_range=(-0.5, 0.5))
        fpnn.fit(self.X_tr, self.y_tr)
        return fpnn

    def test_fit_returns_self(self):
        fpnn = FourierPNN(J=5)
        result = fpnn.fit(self.X_tr, self.y_tr)
        assert result is fpnn

    def test_attributes_after_fit(self):
        fpnn = self._fit()
        assert hasattr(fpnn, "classes_")
        assert hasattr(fpnn, "scaler_")
        assert hasattr(fpnn, "coef_cos_")
        assert hasattr(fpnn, "coef_sin_")
        assert hasattr(fpnn, "class_counts_")
        assert hasattr(fpnn, "n_samples_")

    def test_coef_shapes(self):
        fpnn = self._fit(J=8)
        for c in [0, 1]:
            assert fpnn.coef_cos_[c].shape == (4, 8)
            assert fpnn.coef_sin_[c].shape == (4, 8)

    def test_predict_proba_shape(self):
        fpnn = self._fit()
        proba = fpnn.predict_proba(self.X_te)
        assert proba.shape == (len(self.X_te), 2)

    def test_predict_proba_sums_to_one(self):
        fpnn = self._fit()
        proba = fpnn.predict_proba(self.X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_nonneg(self):
        fpnn = self._fit()
        proba = fpnn.predict_proba(self.X_te)
        assert np.all(proba >= 0)

    def test_predict_shape(self):
        fpnn = self._fit()
        y_pred = fpnn.predict(self.X_te)
        assert y_pred.shape == (len(self.X_te),)

    def test_predict_labels_are_valid(self):
        fpnn = self._fit()
        y_pred = fpnn.predict(self.X_te)
        assert set(y_pred).issubset({0, 1})

    def test_predict_consistent_with_proba(self):
        fpnn = self._fit()
        proba = fpnn.predict_proba(self.X_te)
        y_pred = fpnn.predict(self.X_te)
        expected = fpnn.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(y_pred, expected)

    def test_separable_data_decent_accuracy(self):
        """With well-separated classes, FPNN should exceed 70% balanced accuracy."""
        from sklearn.metrics import balanced_accuracy_score
        fpnn = self._fit(J=10)
        y_pred = fpnn.predict(self.X_te)
        bal_acc = balanced_accuracy_score(self.y_te, y_pred)
        assert bal_acc >= 0.70, f"Balanced accuracy too low: {bal_acc:.3f}"

    def test_get_coefficients_structure(self):
        fpnn = self._fit()
        coefs = fpnn.get_coefficients()
        assert "cos" in coefs and "sin" in coefs
        for c in [0, 1]:
            assert c in coefs["cos"]
            assert coefs["cos"][c].shape == (4, fpnn.J)
            assert coefs["sin"][c].shape == (4, fpnn.J)

    def test_get_coefficients_is_copy(self):
        """Modifying returned coefficients should not affect the model."""
        fpnn = self._fit()
        coefs = fpnn.get_coefficients()
        original = fpnn.coef_cos_[0].copy()
        coefs["cos"][0] *= 999
        np.testing.assert_array_equal(fpnn.coef_cos_[0], original)

    def test_unfitted_predict_raises(self):
        fpnn = FourierPNN(J=5)
        with pytest.raises(RuntimeError):
            fpnn.predict(self.X_te)

    def test_different_J_values(self):
        for J in [1, 5, 20]:
            fpnn = FourierPNN(J=J).fit(self.X_tr, self.y_tr)
            proba = fpnn.predict_proba(self.X_te)
            assert proba.shape == (len(self.X_te), 2)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_single_feature(self):
        X = self.X_tr[:, :1]
        fpnn = FourierPNN(J=5).fit(X, self.y_tr)
        proba = fpnn.predict_proba(self.X_te[:, :1])
        assert proba.shape == (len(self.X_te), 2)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

DUMMY_PARAMS = {
    "lr_C_range": [0.1, 1.0],
    "if_contamination": "auto",
    "ocsvm_kernel": "rbf",
    "fnn_hidden": [16, 8],
    "fnn_epochs": 10,
}


class TestGetBaselines:
    def test_returns_four_classifiers(self):
        b = get_baselines(DUMMY_PARAMS)
        assert len(b) == 5  # LR, IF, OC-SVM, FNN, GRU

    def test_expected_keys(self):
        b = get_baselines(DUMMY_PARAMS)
        assert "Logistic Regression" in b
        assert "Isolation Forest" in b
        assert "One-Class SVM" in b
        assert "Feedforward NN" in b

    def test_all_have_fit(self):
        b = get_baselines(DUMMY_PARAMS)
        for name, clf in b.items():
            assert hasattr(clf, "fit"), f"{name} missing fit"


class TestTrainAndEvaluateAll:
    def setup_method(self):
        self.X, self.y = _make_binary_dataset(n=120, sep=4.0)
        X_tr, y_tr, X_te, y_te = _split(self.X, self.y)
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_te, self.y_te = X_te, y_te
        self.fpnn = FourierPNN(J=5).fit(X_tr, y_tr)
        self.baselines = get_baselines(DUMMY_PARAMS)

    def test_returns_dataframe(self):
        import pandas as pd
        df = train_and_evaluate_all(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.baselines, self.fpnn
        )
        assert isinstance(df, pd.DataFrame)

    def test_rows_include_fpnn_and_baselines(self):
        df = train_and_evaluate_all(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.baselines, self.fpnn
        )
        assert "FPNN" in df.index
        assert "Logistic Regression" in df.index
        assert "Isolation Forest" in df.index
        assert "One-Class SVM" in df.index
        assert "Feedforward NN" in df.index

    def test_expected_columns(self):
        df = train_and_evaluate_all(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.baselines, self.fpnn
        )
        for col in ["Balanced Accuracy", "MCC", "F1 (class 0)", "F1 (class 1)"]:
            assert col in df.columns

    def test_balanced_accuracy_in_range(self):
        df = train_and_evaluate_all(
            self.X_tr, self.y_tr, self.X_te, self.y_te,
            self.baselines, self.fpnn
        )
        ba = df["Balanced Accuracy"].dropna()
        assert (ba >= 0.0).all()
        assert (ba <= 1.0).all()
