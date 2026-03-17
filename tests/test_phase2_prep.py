"""Tests for feature extraction, labelling, and SMOTE balancing."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase1.feature_extract import extract_features, FEATURE_NAMES
from src.phase2.labelling import compute_kappa_mu, label_changepoints
from src.phase2.smote_balance import balance_training_data

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _piecewise(changepoints, levels, n=300, sigma=0.3):
    """Piecewise-constant signal with Gaussian noise."""
    y = np.empty(n)
    mu = np.empty(n)
    boundaries = [0] + list(changepoints) + [n]
    for start, end, level in zip(boundaries[:-1], boundaries[1:], levels):
        y[start:end] = level + RNG.normal(0, sigma, size=end - start)
        mu[start:end] = level
    return y, mu


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def setup_method(self):
        self.y, self.mu = _piecewise([50, 150, 220], [0, 5, 0, 5], n=300)
        self.cps = np.array([50, 150, 220])

    def test_shape(self):
        X, names = extract_features(self.y, self.cps, self.mu, L=10)
        assert X.shape == (3, 4)

    def test_feature_names(self):
        _, names = extract_features(self.y, self.cps, self.mu, L=10)
        assert names == FEATURE_NAMES

    def test_no_nan(self):
        X, _ = extract_features(self.y, self.cps, self.mu, L=10)
        assert np.all(np.isfinite(X))

    def test_delta_mu_sign(self):
        """Upward shift at tau=50 should give positive delta_mu."""
        X, _ = extract_features(self.y, np.array([50]), self.mu, L=10)
        assert X[0, 0] > 0  # mean goes from 0 → 5

    def test_delta_mu_negative_for_downward_shift(self):
        X, _ = extract_features(self.y, np.array([150]), self.mu, L=10)
        assert X[0, 0] < 0  # mean goes from 5 → 0

    def test_persistence_in_unit_interval(self):
        X, _ = extract_features(self.y, self.cps, self.mu, L=10)
        assert np.all(X[:, 1] >= 0.0)
        assert np.all(X[:, 1] <= 1.0)

    def test_phi_local_clipped(self):
        X, _ = extract_features(self.y, self.cps, self.mu, L=10)
        assert np.all(np.abs(X[:, 2]) < 1.0)

    def test_variance_ratio_nonneg(self):
        X, _ = extract_features(self.y, self.cps, self.mu, L=10)
        assert np.all(X[:, 3] >= 0.0)

    def test_empty_changepoints(self):
        X, _ = extract_features(self.y, np.array([], dtype=int), self.mu, L=5)
        assert X.shape == (0, 4)

    def test_epsilon_provided(self):
        """User-supplied epsilon should be used rather than computed."""
        X1, _ = extract_features(self.y, self.cps, self.mu, L=10, epsilon=0.01)
        X2, _ = extract_features(self.y, self.cps, self.mu, L=10, epsilon=100.0)
        # Very large epsilon: all post-change points are within tolerance → S ≈ 1
        assert np.all(X2[:, 1] == pytest.approx(1.0))
        # Very small epsilon: very few points within tolerance → S ≈ 0
        assert X1[:, 1].mean() < X2[:, 1].mean()

    def test_boundary_changepoint(self):
        """Changepoint near the start/end should not raise."""
        X, _ = extract_features(self.y, np.array([2, 297]), self.mu, L=10)
        assert X.shape == (2, 4)
        assert np.all(np.isfinite(X))

    def test_custom_epsilon_none_computed(self):
        """When epsilon=None, it should be derived from median(|y-mu|)."""
        X, _ = extract_features(self.y, self.cps, self.mu, L=10, epsilon=None)
        assert np.all(np.isfinite(X))

    def test_large_shift_higher_persistence(self):
        """A large, clean shift should produce higher persistence than a small one."""
        y_large, mu_large = _piecewise([100], [0, 20], n=200, sigma=0.1)
        y_small, mu_small = _piecewise([100], [0, 0.2], n=200, sigma=0.1)
        X_large, _ = extract_features(y_large, np.array([100]), mu_large, L=15, epsilon=1.0)
        X_small, _ = extract_features(y_small, np.array([100]), mu_small, L=15, epsilon=1.0)
        assert X_large[0, 1] >= X_small[0, 1]


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

class TestComputeKappaMu:
    def setup_method(self):
        self.X = np.column_stack([
            np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
            np.zeros(8),
            np.zeros(8),
            np.zeros(8),
        ])

    def test_returns_float(self):
        k = compute_kappa_mu(self.X, percentile=75)
        assert isinstance(k, float)

    def test_75th_percentile_of_abs_delta_mu(self):
        k = compute_kappa_mu(self.X, percentile=75)
        expected = float(np.percentile(np.abs(self.X[:, 0]), 75))
        assert k == pytest.approx(expected)

    def test_50th_percentile(self):
        k = compute_kappa_mu(self.X, percentile=50)
        expected = float(np.percentile(np.abs(self.X[:, 0]), 50))
        assert k == pytest.approx(expected)

    def test_negative_delta_mu_uses_abs(self):
        X_neg = self.X.copy()
        X_neg[:, 0] *= -1
        k_pos = compute_kappa_mu(self.X, 75)
        k_neg = compute_kappa_mu(X_neg, 75)
        assert k_pos == pytest.approx(k_neg)


class TestLabelChangepoints:
    def _make_X(self, delta_mus, S_vals):
        m = len(delta_mus)
        return np.column_stack([
            np.array(delta_mus, dtype=float),
            np.array(S_vals, dtype=float),
            np.zeros(m),
            np.zeros(m),
        ])

    def test_all_sustained(self):
        X = self._make_X([10, 20, 15], [0.8, 0.9, 0.7])
        labels = label_changepoints(X, kappa_mu=5.0, kappa_S=0.5)
        assert np.all(labels == 1)

    def test_all_recoiled_low_magnitude(self):
        X = self._make_X([1, 2, 0.5], [0.9, 0.9, 0.9])
        labels = label_changepoints(X, kappa_mu=5.0, kappa_S=0.5)
        assert np.all(labels == 0)

    def test_all_recoiled_low_persistence(self):
        X = self._make_X([10, 20, 15], [0.1, 0.2, 0.3])
        labels = label_changepoints(X, kappa_mu=5.0, kappa_S=0.5)
        assert np.all(labels == 0)

    def test_mixed(self):
        # Row 0: large shift, high persistence → sustained
        # Row 1: large shift, low persistence → recoiled
        # Row 2: small shift, high persistence → recoiled
        X = self._make_X([10, 10, 1], [0.8, 0.2, 0.9])
        labels = label_changepoints(X, kappa_mu=5.0, kappa_S=0.5)
        assert labels[0] == 1
        assert labels[1] == 0
        assert labels[2] == 0

    def test_output_shape(self):
        X = self._make_X([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5])
        labels = label_changepoints(X, kappa_mu=2.0, kappa_S=0.4)
        assert labels.shape == (4,)

    def test_dtype_int(self):
        X = self._make_X([5], [0.8])
        labels = label_changepoints(X, kappa_mu=2.0, kappa_S=0.5)
        assert labels.dtype == int or np.issubdtype(labels.dtype, np.integer)

    def test_boundary_exactly_kappa(self):
        """Exactly at threshold should be recoiled (strictly greater than required)."""
        X = self._make_X([5.0], [0.5])
        labels = label_changepoints(X, kappa_mu=5.0, kappa_S=0.5)
        assert labels[0] == 0  # |5| > 5 is False; S > 0.5 is False

    def test_negative_delta_mu_uses_abs(self):
        X_pos = self._make_X([10], [0.9])
        X_neg = self._make_X([-10], [0.9])
        assert label_changepoints(X_pos, 5.0, 0.5)[0] == 1
        assert label_changepoints(X_neg, 5.0, 0.5)[0] == 1


# ---------------------------------------------------------------------------
# SMOTE balancing
# ---------------------------------------------------------------------------

class TestBalanceTrainingData:
    def _imbalanced(self, n_majority=80, n_minority=20, n_features=4, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_majority + n_minority, n_features))
        y = np.array([0] * n_majority + [1] * n_minority)
        return X, y

    def test_output_shapes_consistent(self):
        X, y = self._imbalanced()
        X_res, y_res = balance_training_data(X, y)
        assert X_res.shape[1] == X.shape[1]
        assert len(X_res) == len(y_res)

    def test_classes_balanced(self):
        X, y = self._imbalanced(80, 20)
        X_res, y_res = balance_training_data(X, y)
        counts = {c: int((y_res == c).sum()) for c in [0, 1]}
        assert counts[0] == counts[1]

    def test_majority_unchanged(self):
        """SMOTE only synthesises minority samples; majority count stays the same."""
        X, y = self._imbalanced(80, 20)
        X_res, y_res = balance_training_data(X, y)
        assert int((y_res == 0).sum()) == 80

    def test_minority_increased(self):
        X, y = self._imbalanced(80, 20)
        X_res, y_res = balance_training_data(X, y)
        assert int((y_res == 1).sum()) == 80

    def test_k_neighbors_reduction(self):
        """Minority class of 3 samples with k=5 should reduce k to 2 without error."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(13, 4))
        y = np.array([0] * 10 + [1] * 3)
        X_res, y_res = balance_training_data(X, y, k_neighbors=5)
        assert len(X_res) == len(y_res)

    def test_single_class_returns_original(self):
        X = np.ones((10, 4))
        y = np.zeros(10, dtype=int)
        X_res, y_res = balance_training_data(X, y)
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)

    def test_reproducible(self):
        X, y = self._imbalanced()
        X1, y1 = balance_training_data(X, y, random_state=42)
        X2, y2 = balance_training_data(X, y, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X, y = self._imbalanced()
        X1, _ = balance_training_data(X, y, random_state=1)
        X2, _ = balance_training_data(X, y, random_state=2)
        assert not np.allclose(X1, X2)


# ---------------------------------------------------------------------------
# relabel_with_hypersensitive (Change 3 / Change 7b)
# ---------------------------------------------------------------------------

class TestRelabelWithHypersensitive:
    """Unit tests for the 4-class X-algorithm relabelling."""

    def setup_method(self):
        from src.phase2.labelling import relabel_with_hypersensitive
        self.relabel = relabel_with_hypersensitive

    def _make_flags(self, n, flag_positions):
        """Create a boolean flags array of length n with True at given positions."""
        flags = np.zeros(n, dtype=bool)
        for p in flag_positions:
            flags[p] = True
        return flags

    def test_x_hit_and_true_hit_gives_sustained(self):
        """x_hit + true_hit → Sustained."""
        flags = self._make_flags(100, [50])
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=np.array([50]),
            existing_labels=np.array([0]),
            tolerance=5,
        )
        assert labels[0] == "Sustained"

    def test_x_hit_no_true_hit_gives_abrupt(self):
        """x_hit + no true_hit → Abrupt."""
        flags = self._make_flags(100, [50])
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=np.array([80]),  # far away
            existing_labels=np.array([1]),
            tolerance=5,
        )
        assert labels[0] == "Abrupt"

    def test_no_x_hit_true_hit_gives_abrupt_preceded(self):
        """no x_hit + true_hit → Abrupt-Preceded."""
        flags = self._make_flags(100, [80])   # flag far from CP
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=np.array([50]),
            existing_labels=np.array([1]),
            tolerance=5,
        )
        assert labels[0] == "Abrupt-Preceded"

    def test_no_x_hit_no_true_hit_keeps_existing(self):
        """no x_hit + no true_hit → keep existing label."""
        flags = self._make_flags(100, [80])
        existing = np.array([0])
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=np.array([20]),   # far
            existing_labels=existing,
            tolerance=5,
        )
        assert labels[0] == "Recoiled"   # existing 0 → "Recoiled"

    def test_no_label_mode_empty_true_cps(self):
        """No-label mode (empty true_cp_indices): only x_hit drives label."""
        flags = self._make_flags(100, [50])
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=np.array([]),
            existing_labels=np.array([1]),
            tolerance=5,
        )
        # x_hit=True, true_hit=False → Abrupt
        assert labels[0] == "Abrupt"

    def test_no_label_mode_none_true_cps(self):
        """No-label mode (None true_cp_indices): only x_hit drives label."""
        flags = self._make_flags(100, [])  # no x flags either
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=None,
            existing_labels=np.array([1]),
            tolerance=5,
        )
        # x_hit=False, true_hit=False → keep existing (1 → "Sustained")
        assert labels[0] == "Sustained"

    def test_output_length_matches_input(self):
        """Output length must equal number of CPs."""
        flags = self._make_flags(200, [30, 70, 120])
        labels = self.relabel(
            cp_indices=np.array([30, 70, 120]),
            x_flags=flags,
            true_cp_indices=np.array([30, 70]),
            existing_labels=np.array([1, 0, 1]),
            tolerance=5,
        )
        assert len(labels) == 3

    def test_tolerance_window(self):
        """x_flag at position 45 should hit CP at 50 with tolerance=10."""
        flags = self._make_flags(100, [45])
        labels = self.relabel(
            cp_indices=np.array([50]),
            x_flags=flags,
            true_cp_indices=np.array([]),
            existing_labels=np.array([0]),
            tolerance=10,
        )
        assert labels[0] == "Abrupt"  # x_hit=True (within 10), true_hit=False
