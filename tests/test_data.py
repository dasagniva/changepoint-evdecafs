"""Unit tests for src/data/loader.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_welllog_data, _generate_synthetic_welllog, _WELLLOG_TRUE_CPS


# ---------------------------------------------------------------------------
# Synthetic well-log generator
# ---------------------------------------------------------------------------

class TestGenerateSyntheticWelllog:
    def setup_method(self):
        self.y, self.outliers = _generate_synthetic_welllog(
            n=4050,
            changepoints=_WELLLOG_TRUE_CPS,
            random_seed=0,
        )

    def test_output_shape(self):
        assert self.y.shape == (4050,)

    def test_dtype_is_float(self):
        assert self.y.dtype == float

    def test_no_nan_or_inf(self):
        assert np.all(np.isfinite(self.y))

    def test_outlier_count(self):
        assert len(self.outliers) == 20

    def test_outlier_indices_in_range(self):
        assert self.outliers.min() >= 10
        assert self.outliers.max() < 4040

    def test_outlier_indices_sorted(self):
        assert np.all(np.diff(self.outliers) > 0)

    def test_segment_means_in_range(self):
        # Sample a few mid-segment points away from changepoints and outliers
        safe = np.ones(4050, dtype=bool)
        for cp in _WELLLOG_TRUE_CPS:
            safe[max(0, cp - 10): cp + 10] = False
        safe[self.outliers] = False
        segment_vals = self.y[safe]
        # AR(1) noise std is 2000; segment means are in [70k, 140k]
        # Values should mostly lie within ±4*sigma of segment bounds
        assert segment_vals.min() > 70_000 - 10_000
        assert segment_vals.max() < 140_000 + 10_000


# ---------------------------------------------------------------------------
# load_welllog_data (synthetic path, no real CSV)
# ---------------------------------------------------------------------------

class TestLoadWelllogData:
    def setup_method(self, tmp_path=None):
        import tempfile, os
        self.tmp_dir = tempfile.mkdtemp()
        self.cache = Path(self.tmp_dir) / "welllog_test.csv"

    def _load(self, train_fraction=0.75):
        return load_welllog_data(
            cache_path=self.cache,
            train_fraction=train_fraction,
            random_seed=42,
        )

    def test_returns_four_arrays(self):
        result = self._load()
        assert len(result) == 4

    def test_train_test_shapes(self):
        y_train, y_test, cps, outliers = self._load(train_fraction=0.75)
        total = len(y_train) + len(y_test)
        assert total == 4050
        assert len(y_train) == int(4050 * 0.75)

    def test_dtypes(self):
        y_train, y_test, cps, outliers = self._load()
        assert y_train.dtype == float
        assert y_test.dtype == float
        assert cps.dtype == int or np.issubdtype(cps.dtype, np.integer)

    def test_no_nan(self):
        y_train, y_test, cps, outliers = self._load()
        assert np.all(np.isfinite(y_train))
        assert np.all(np.isfinite(y_test))

    def test_ground_truth_cps_match_expected(self):
        _, _, cps, _ = self._load()
        np.testing.assert_array_equal(cps, _WELLLOG_TRUE_CPS)

    def test_outlier_array_returned(self):
        _, _, _, outliers = self._load()
        assert isinstance(outliers, np.ndarray)
        assert len(outliers) == 20

    def test_cache_created(self):
        self._load()
        assert self.cache.exists()

    def test_cache_reload_identical(self):
        # Second call reads from CSV; allow tiny floating-point rounding from round-trip
        y_train_1, y_test_1, cps_1, _ = self._load()
        y_train_2, y_test_2, cps_2, _ = self._load()
        np.testing.assert_allclose(y_train_1, y_train_2, rtol=1e-10)
        np.testing.assert_allclose(y_test_1, y_test_2, rtol=1e-10)

    def test_different_train_fractions(self):
        y_train, y_test, _, _ = load_welllog_data(
            cache_path=Path(self.tmp_dir) / "wl_80.csv",
            train_fraction=0.80,
        )
        assert len(y_train) == int(4050 * 0.80)
        assert len(y_train) + len(y_test) == 4050


# ---------------------------------------------------------------------------
# load_bitcoin_data (network-optional)
# ---------------------------------------------------------------------------

class TestLoadBitcoinData:
    """Tests that do not require a network connection."""

    def test_cached_load(self, tmp_path):
        """If a valid CSV exists, data is loaded without hitting the network."""
        import pandas as pd

        cache = tmp_path / "btc.csv"
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 100)))
        df = pd.DataFrame(
            {"log_price": np.log(prices)},
            index=dates,
        )
        df.index.name = "date"
        df.to_csv(cache)

        from src.data.loader import load_bitcoin_data

        y_train, y_test, d_train, d_test = load_bitcoin_data(
            start_date="2020-01-01",
            end_date="2020-04-09",
            train_end_date="2020-02-29",
            cache_path=cache,
        )
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        assert len(y_train) + len(y_test) == 100
        assert np.all(np.isfinite(y_train))
        assert np.all(np.isfinite(y_test))
        assert len(d_train) == len(y_train)
        assert len(d_test) == len(y_test)

    def test_train_test_no_overlap(self, tmp_path):
        import pandas as pd

        cache = tmp_path / "btc2.csv"
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame(
            {"log_price": np.zeros(200)},
            index=dates,
        )
        df.index.name = "date"
        df.to_csv(cache)

        from src.data.loader import load_bitcoin_data

        _, _, d_train, d_test = load_bitcoin_data(
            train_end_date="2020-04-30",
            cache_path=cache,
        )
        if len(d_train) > 0 and len(d_test) > 0:
            assert d_train[-1] < d_test[0]
