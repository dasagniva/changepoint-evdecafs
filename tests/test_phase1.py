"""Unit tests for Phase I: AR(1) estimation, EVT penalty, EV-DeCAFS."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase1.ar1_model import estimate_ar1_params
from src.phase1.evt_penalty import (
    compute_adaptive_penalty,
    compute_evi_field,
    compute_exceedance_count_penalty,
)
from src.phase1.decafs import ev_decafs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _ar1_series(n: int, phi: float = 0.5, sigma_v: float = 1.0) -> np.ndarray:
    """Stationary AR(1) series with zero mean."""
    v = RNG.normal(0, sigma_v, size=n)
    y = np.empty(n)
    y[0] = v[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + v[t]
    return y


def _piecewise_ar1(changepoints, means, n: int = 400, phi: float = 0.4, sigma_v: float = 1.0) -> np.ndarray:
    """Piecewise-constant mean plus AR(1) noise."""
    y = np.empty(n)
    boundaries = [0] + list(changepoints) + [n]
    for start, end, mu in zip(boundaries[:-1], boundaries[1:], means):
        seg = np.empty(end - start)
        seg[0] = mu + RNG.normal(0, sigma_v)
        for i in range(1, len(seg)):
            seg[i] = mu + phi * (seg[i - 1] - mu) + RNG.normal(0, sigma_v)
        y[start:end] = seg
    return y


# ---------------------------------------------------------------------------
# AR(1) estimation
# ---------------------------------------------------------------------------

class TestEstimateAR1Params:
    def test_returns_expected_keys(self):
        y = _ar1_series(500)
        result = estimate_ar1_params(y)
        assert set(result.keys()) == {"phi", "sigma_v_sq", "sigma_eta_sq"}

    def test_phi_stationarity(self):
        y = _ar1_series(1000, phi=0.7)
        r = estimate_ar1_params(y)
        assert abs(r["phi"]) < 1.0

    def test_phi_sign_correct(self):
        """Positive phi series should yield positive phi estimate."""
        y = _ar1_series(2000, phi=0.8, sigma_v=0.5)
        r = estimate_ar1_params(y)
        assert r["phi"] > 0

    def test_phi_negative_correct(self):
        y = _ar1_series(2000, phi=-0.6, sigma_v=0.5)
        r = estimate_ar1_params(y)
        assert r["phi"] < 0

    def test_sigma_v_sq_positive(self):
        y = _ar1_series(500)
        r = estimate_ar1_params(y)
        assert r["sigma_v_sq"] > 0

    def test_sigma_eta_sq_at_least_floor(self):
        y = _ar1_series(500)
        r = estimate_ar1_params(y)
        assert r["sigma_eta_sq"] >= 1e-8

    def test_phi_rough_accuracy(self):
        """With n=5000 the Yule-Walker estimate should be within 0.15 of the truth."""
        y = _ar1_series(5000, phi=0.5, sigma_v=1.0)
        r = estimate_ar1_params(y)
        assert abs(r["phi"] - 0.5) < 0.15

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            estimate_ar1_params(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# EVT penalty
# ---------------------------------------------------------------------------

class TestComputeEviField:
    def setup_method(self):
        self.n = 200
        self.y = _ar1_series(self.n, phi=0.3, sigma_v=1.0)

    def test_output_shape(self):
        xi = compute_evi_field(self.y, w=20, q0=0.90)
        assert xi.shape == (self.n,)

    def test_dtype_float(self):
        xi = compute_evi_field(self.y, w=20, q0=0.90)
        assert xi.dtype == float

    def test_no_nan(self):
        xi = compute_evi_field(self.y, w=20, q0=0.90)
        assert np.all(np.isfinite(xi))

    def test_heavy_tail_series_higher_xi(self):
        """A series with injected outliers should produce higher mean xi."""
        y_clean = _ar1_series(300, phi=0.3, sigma_v=1.0)
        y_heavy = y_clean.copy()
        y_heavy[::15] += RNG.choice([-10, 10], size=len(y_heavy[::15]))
        xi_clean = compute_evi_field(y_clean, w=20, q0=0.90)
        xi_heavy = compute_evi_field(y_heavy, w=20, q0=0.90)
        assert xi_heavy.mean() >= xi_clean.mean()


class TestComputeAdaptivePenalty:
    def test_output_shape(self):
        xi = np.random.default_rng(1).uniform(-0.1, 0.5, 100)
        alpha = compute_adaptive_penalty(xi, alpha_0=5.0, lambda_ev=1.0)
        assert alpha.shape == (100,)

    def test_all_at_least_alpha_0(self):
        xi = np.random.default_rng(1).uniform(-0.5, 0.5, 100)
        alpha = compute_adaptive_penalty(xi, alpha_0=5.0, lambda_ev=1.0)
        assert np.all(alpha >= 5.0)

    def test_negative_xi_increases_penalty(self):
        # With |xi|, negative xi now raises penalty (both directions of tail
        # irregularity are captured)
        xi = np.full(50, -1.0)
        alpha = compute_adaptive_penalty(xi, alpha_0=3.0, lambda_ev=2.0)
        np.testing.assert_allclose(alpha, 3.0 * (1 + 2.0 * 1.0))

    def test_positive_xi_increases_penalty(self):
        xi = np.full(10, 0.5)
        alpha = compute_adaptive_penalty(xi, alpha_0=4.0, lambda_ev=1.0)
        np.testing.assert_allclose(alpha, 4.0 * (1 + 0.5))


class TestExceedanceCountPenalty:
    def test_output_shape(self):
        n = 100
        y = _ar1_series(n)
        mu = np.zeros(n)
        alpha = compute_exceedance_count_penalty(y, mu, sigma_v=1.0, w=10, c=2.0, alpha_0=5.0)
        assert alpha.shape == (n,)

    def test_all_at_least_alpha_0(self):
        n = 100
        y = _ar1_series(n)
        mu = np.zeros(n)
        alpha = compute_exceedance_count_penalty(y, mu, sigma_v=1.0, w=10, c=2.0, alpha_0=5.0)
        assert np.all(alpha >= 5.0)

    def test_more_outliers_more_penalty(self):
        """Series with many outliers should have higher mean penalty."""
        n = 200
        y_clean = _ar1_series(n, sigma_v=0.1)
        y_noisy = y_clean.copy()
        y_noisy[::5] += 10.0
        mu = np.zeros(n)
        a_clean = compute_exceedance_count_penalty(y_clean, mu, 0.1, w=10, c=2.0, alpha_0=5.0)
        a_noisy = compute_exceedance_count_penalty(y_noisy, mu, 0.1, w=10, c=2.0, alpha_0=5.0)
        assert a_noisy.mean() > a_clean.mean()


# ---------------------------------------------------------------------------
# EV-DeCAFS
# ---------------------------------------------------------------------------

class TestEvDecafs:
    def _run(self, y, alpha_0=2.0, lambda_param=1.0, gamma=1.0, phi=0.3, n_grid=100):
        alpha_t = np.full(len(y), alpha_0)
        return ev_decafs(y, alpha_t, lambda_param, gamma, phi, n_grid=n_grid)

    def test_returns_expected_keys(self):
        y = _ar1_series(50)
        result = self._run(y)
        assert "changepoints" in result
        assert "means" in result
        assert "cost" in result

    def test_means_shape(self):
        n = 60
        y = _ar1_series(n)
        result = self._run(y)
        assert result["means"].shape == (n,)

    def test_cost_is_finite(self):
        y = _ar1_series(60)
        result = self._run(y)
        assert np.isfinite(result["cost"])

    def test_cost_nonnegative(self):
        y = _ar1_series(60)
        result = self._run(y)
        assert result["cost"] >= 0.0

    def test_changepoints_in_range(self):
        y = _piecewise_ar1([50, 100, 150], [0, 5, 0, 5], n=200)
        result = self._run(y, alpha_0=5.0, lambda_param=0.5, gamma=1.0)
        cps = result["changepoints"]
        assert np.all(cps >= 1)
        assert np.all(cps < 200)

    def test_detects_large_shift(self):
        """A very large mean shift should always be detected."""
        y = _piecewise_ar1([100], [0, 50], n=200)
        result = self._run(y, alpha_0=2.0, lambda_param=0.1, gamma=1.0, n_grid=200)
        cps = result["changepoints"]
        # At least one changepoint near t=100
        assert len(cps) > 0
        assert np.any(np.abs(cps - 100) < 20)

    def test_constant_series_no_changepoints(self):
        """A perfectly constant series should produce no changepoints."""
        y = np.ones(100)
        alpha_t = np.full(100, 10.0)
        result = ev_decafs(y, alpha_t, lambda_param=1.0, gamma=1.0, phi=0.0, n_grid=50)
        assert len(result["changepoints"]) == 0

    def test_n1_series(self):
        """Edge case: single observation."""
        y = np.array([3.0])
        alpha_t = np.array([1.0])
        result = ev_decafs(y, alpha_t, lambda_param=1.0, gamma=1.0, phi=0.0)
        assert result["means"].shape == (1,)
        assert len(result["changepoints"]) == 0

    def test_means_within_data_range(self):
        y = _ar1_series(80)
        result = self._run(y)
        # Means should not wildly extrapolate (grid is min-2std to max+2std)
        assert result["means"].min() >= y.min() - 3 * np.std(y)
        assert result["means"].max() <= y.max() + 3 * np.std(y)


# ---------------------------------------------------------------------------
# BOCPD and CUSUM detectors (Change 2 / Change 7a)
# ---------------------------------------------------------------------------

class TestBOCPD:
    """Unit tests for run_bocpd."""

    def test_returns_bool_array_same_length(self):
        from src.phase1.hypersensitive_cpd import run_bocpd
        y = _ar1_series(100)
        flags = run_bocpd(y, phi=0.5, sigma_v=1.0)
        assert flags.dtype == bool
        assert len(flags) == 100

    def test_step_change_produces_flags(self):
        """A large step change should trigger at least one BOCPD flag."""
        from src.phase1.hypersensitive_cpd import run_bocpd
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 0.5, 50), rng.normal(10, 0.5, 50)])
        flags = run_bocpd(y, phi=0.0, sigma_v=0.5, threshold=0.3)
        assert flags.any(), "Expected at least one BOCPD flag on step-change series"

    def test_constant_series_low_flags(self):
        """A constant series should produce no flags (posterior stays at prior)."""
        from src.phase1.hypersensitive_cpd import run_bocpd
        y = np.ones(100)
        flags = run_bocpd(y, phi=0.0, sigma_v=1.0, threshold=0.9)
        assert not flags.any(), "Constant series should not trigger BOCPD"

    def test_short_series(self):
        """Series of length 1 should return all-False flags."""
        from src.phase1.hypersensitive_cpd import run_bocpd
        y = np.array([1.0])
        flags = run_bocpd(y, phi=0.0, sigma_v=1.0)
        assert not flags.any()

    def test_threshold_effect(self):
        """Higher threshold should produce fewer (or equal) flags."""
        from src.phase1.hypersensitive_cpd import run_bocpd
        rng = np.random.default_rng(7)
        y = np.concatenate([rng.normal(0, 1, 60), rng.normal(5, 1, 60)])
        flags_low = run_bocpd(y, phi=0.0, sigma_v=1.0, threshold=0.1)
        flags_high = run_bocpd(y, phi=0.0, sigma_v=1.0, threshold=0.9)
        assert flags_low.sum() >= flags_high.sum()


class TestCUSUM:
    """Unit tests for run_cusum."""

    def test_returns_bool_array_same_length(self):
        from src.phase1.hypersensitive_cpd import run_cusum
        y = _ar1_series(100)
        flags = run_cusum(y, phi=0.5, sigma_v=1.0)
        assert flags.dtype == bool
        assert len(flags) == 100

    def test_step_change_produces_flags(self):
        """A large step change should trigger CUSUM flags."""
        from src.phase1.hypersensitive_cpd import run_cusum
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0, 0.2, 50), rng.normal(10, 0.2, 50)])
        flags = run_cusum(y, phi=0.0, sigma_v=0.2, h_multiplier=1.0)
        assert flags.any(), "Expected CUSUM flags on step-change series"

    def test_constant_series_no_flags(self):
        """A perfectly constant series should produce no CUSUM flags."""
        from src.phase1.hypersensitive_cpd import run_cusum
        y = np.zeros(200)
        flags = run_cusum(y, phi=0.0, sigma_v=1.0)
        assert not flags.any()

    def test_short_series(self):
        """Series of length 1 should return all-False flags."""
        from src.phase1.hypersensitive_cpd import run_cusum
        y = np.array([5.0])
        flags = run_cusum(y, phi=0.0, sigma_v=1.0)
        assert not flags.any()

    def test_higher_h_fewer_flags(self):
        """A stricter control limit should produce fewer or equal flags."""
        from src.phase1.hypersensitive_cpd import run_cusum
        rng = np.random.default_rng(13)
        y = rng.normal(0, 1, 300)
        # Inject a shift
        y[150:] += 3.0
        flags_sens = run_cusum(y, phi=0.0, sigma_v=1.0, h_multiplier=0.5)
        flags_cons = run_cusum(y, phi=0.0, sigma_v=1.0, h_multiplier=5.0)
        assert flags_sens.sum() >= flags_cons.sum()
