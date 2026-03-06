"""Data loading for EV-DeCAFS experiments.

Provides loaders for:
- Bitcoin daily log-prices (via yfinance)
- Well-log nuclear response signal (or synthetic surrogate)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Synthetic well-log ground-truth changepoint positions (0-indexed)
_WELLLOG_TRUE_CPS = np.array(
    [400, 820, 1210, 1320, 1540, 1790, 2050, 2380, 2690, 2990, 3300, 3590],
    dtype=int,
)
_WELLLOG_N = 4050


def load_bitcoin_data(
    start_date: str = "2014-01-01",
    end_date: str = "2024-12-31",
    train_end_date: str = "2022-12-31",
    cache_path: str | Path = "data/raw/btc_usd.csv",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Download (or load from cache) daily BTC-USD log-prices and split.

    Parameters
    ----------
    start_date:
        Start of the download range (inclusive), e.g. ``"2014-01-01"``.
    end_date:
        End of the download range (inclusive).
    train_end_date:
        Inclusive end of the training split.  Observations after this date
        form the test set.
    cache_path:
        Local CSV path.  If the file exists it is read directly; otherwise
        data are downloaded from Yahoo Finance and cached.

    Returns
    -------
    y_train : np.ndarray, shape (n_train,)
        Log-prices in the training period.
    y_test : np.ndarray, shape (n_test,)
        Log-prices in the test period.
    dates_train : pd.DatetimeIndex
    dates_test : pd.DatetimeIndex
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Loading Bitcoin data from cache: %s", cache_path)
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
    else:
        logger.info(
            "Downloading BTC-USD from Yahoo Finance (%s to %s)...",
            start_date,
            end_date,
        )
        import yfinance as yf

        ticker = yf.Ticker("BTC-USD")
        raw = ticker.history(start=start_date, end=end_date, interval="1d")
        if raw.empty:
            raise ValueError("yfinance returned no data for BTC-USD.")

        df = pd.DataFrame({"log_price": np.log(raw["Close"].values)}, index=raw.index)
        df.index.name = "date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache_path)
        logger.info("Cached %d observations to %s", len(df), cache_path)

    log_price = df["log_price"].values.astype(float)
    dates = df.index

    # Remove any NaN / Inf arising from zero prices
    valid = np.isfinite(log_price)
    log_price = log_price[valid]
    dates = dates[valid]

    train_mask = dates <= pd.Timestamp(train_end_date)
    y_train = log_price[train_mask]
    y_test = log_price[~train_mask]
    dates_train = dates[train_mask]
    dates_test = dates[~train_mask]

    logger.info(
        "Bitcoin split — train: %d obs (%s – %s), test: %d obs (%s – %s)",
        len(y_train),
        dates_train[0].date() if len(dates_train) else "N/A",
        dates_train[-1].date() if len(dates_train) else "N/A",
        len(y_test),
        dates_test[0].date() if len(dates_test) else "N/A",
        dates_test[-1].date() if len(dates_test) else "N/A",
    )
    return y_train, y_test, dates_train, dates_test


def load_welllog_data(
    cache_path: str | Path = "data/raw/welllog.csv",
    train_fraction: float = 0.75,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (or synthesise) the well-log nuclear response dataset.

    Attempts to read a local CSV first.  If it does not exist, generates a
    synthetic surrogate with known changepoints and outliers that mirrors
    the statistical properties of Ruanaidh & Fitzgerald (2012).

    Parameters
    ----------
    cache_path:
        Path to a CSV with a single numeric column of observations.
        If the file does not exist, a synthetic dataset is generated and
        saved here for reproducibility.
    train_fraction:
        Fraction of observations used for the training split (chronological).
    random_seed:
        Seed for the synthetic data generator.

    Returns
    -------
    y_train : np.ndarray
    y_test : np.ndarray
    ground_truth_changepoints : np.ndarray
        Indices of true changepoints in the *full* (pre-split) signal.
    ground_truth_outliers : np.ndarray
        Indices of injected outlier spikes in the full signal.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    true_cps = _WELLLOG_TRUE_CPS.copy()

    if cache_path.exists():
        logger.info("Loading well-log data from: %s", cache_path)
        df = pd.read_csv(cache_path)
        col = df.columns[0]
        y = df[col].values.astype(float)
        # When loading real data we don't have ground-truth outliers
        ground_truth_outliers = np.array([], dtype=int)
        logger.info("Loaded %d well-log observations", len(y))
    else:
        logger.info(
            "Well-log CSV not found at %s — generating synthetic surrogate.", cache_path
        )
        y, ground_truth_outliers = _generate_synthetic_welllog(
            n=_WELLLOG_N,
            changepoints=true_cps,
            random_seed=random_seed,
        )
        pd.DataFrame({"welllog": y}).to_csv(cache_path, index=False)
        logger.info(
            "Synthetic well-log saved to %s (%d obs, %d changepoints, %d outliers)",
            cache_path,
            len(y),
            len(true_cps),
            len(ground_truth_outliers),
        )

    n_train = int(len(y) * train_fraction)
    y_train = y[:n_train]
    y_test = y[n_train:]

    logger.info(
        "Well-log split — train: %d obs, test: %d obs", len(y_train), len(y_test)
    )
    return y_train, y_test, true_cps, ground_truth_outliers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_welllog(
    n: int,
    changepoints: np.ndarray,
    random_seed: int = 42,
    n_outliers: int = 20,
    outlier_magnitude: float = 30_000.0,
    ar1_phi: float = 0.5,
    ar1_sigma_v: float = 2_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic well-log surrogate.

    Constructs a piecewise-constant signal whose segment means are drawn
    uniformly from [70 000, 140 000], adds AR(1) noise, and injects outlier
    spikes at random positions.

    Parameters
    ----------
    n:
        Total number of observations.
    changepoints:
        Sorted array of changepoint indices (segment boundaries).
    random_seed:
        NumPy random seed for reproducibility.
    n_outliers:
        Number of outlier spikes to inject.
    outlier_magnitude:
        Absolute spike size (sign is chosen randomly).
    ar1_phi:
        AR(1) autocorrelation for the noise process.
    ar1_sigma_v:
        Innovation standard deviation for the AR(1) noise.

    Returns
    -------
    y : np.ndarray, shape (n,)
    outlier_indices : np.ndarray of int
    """
    rng = np.random.default_rng(random_seed)

    # Build piecewise-constant mean
    boundaries = np.concatenate([[0], changepoints, [n]])
    mu = np.empty(n)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        level = rng.uniform(70_000, 140_000)
        mu[start:end] = level

    # AR(1) noise: epsilon_t = phi * epsilon_{t-1} + v_t
    v = rng.normal(0, ar1_sigma_v, size=n)
    epsilon = np.empty(n)
    epsilon[0] = v[0]
    for t in range(1, n):
        epsilon[t] = ar1_phi * epsilon[t - 1] + v[t]

    y = mu + epsilon

    # Inject outlier spikes at random positions (avoid the first and last 10)
    all_positions = np.arange(10, n - 10)
    outlier_indices = rng.choice(all_positions, size=n_outliers, replace=False)
    outlier_indices.sort()
    signs = rng.choice([-1, 1], size=n_outliers)
    y[outlier_indices] += signs * outlier_magnitude

    return y, outlier_indices
