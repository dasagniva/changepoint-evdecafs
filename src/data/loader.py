"""Data loading for EV-DeCAFS experiments.

Provides loaders for:
- Well-log nuclear response signal (or synthetic surrogate)
- TCPD benchmark series (brent_spot, scanline_42049, etc.)
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import Counter
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
        "Well-log split — n=%d, train=%d (%.0f%%), test=%d (%.0f%%)",
        len(y),
        len(y_train), 100 * len(y_train) / len(y),
        len(y_test), 100 * len(y_test) / len(y),
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


# =============================================================================
# TCPD benchmark loaders
# =============================================================================

_TCPD_BASE_URL = (
    "https://raw.githubusercontent.com/alan-turing-institute/TCPD/master"
)


def load_tcpd_series(
    name: str,
    cache_dir: str = "data/raw/tcpd",
) -> dict:
    """Load a TCPD series by name, downloading and caching if needed.

    Parameters
    ----------
    name:
        TCPD dataset name, e.g. ``"brent_spot"`` or ``"scanline_42049"``.
    cache_dir:
        Directory for cached JSON files.

    Returns
    -------
    dict with keys:
        - ``'y'`` : np.ndarray — raw observation values.
        - ``'annotations'`` : dict annotator → list of CP indices.
        - ``'consensus_cps'`` : np.ndarray — majority-vote CP indices
          (>= 3 out of N annotators, with 2%-tolerance clustering).
        - ``'n_annotators'`` : int
        - ``'name'`` : str
        - ``'longname'`` : str
    """
    os.makedirs(cache_dir, exist_ok=True)

    # --- Download dataset JSON ---
    data_path = os.path.join(cache_dir, f"{name}.json")
    if not os.path.exists(data_path):
        url = f"{_TCPD_BASE_URL}/datasets/{name}/{name}.json"
        logger.info("Downloading TCPD series '%s' from %s", name, url)
        urllib.request.urlretrieve(url, data_path)

    with open(data_path) as f:
        data = json.load(f)

    # Extract observation values (first column if multidimensional)
    raw = data["series"][0]["raw"]
    y = np.array(
        [row[0] if isinstance(row, list) else row for row in raw],
        dtype=float,
    )

    # --- Download annotations ---
    ann_path = os.path.join(cache_dir, "annotations.json")
    if not os.path.exists(ann_path):
        ann_url = f"{_TCPD_BASE_URL}/annotations.json"
        logger.info("Downloading TCPD annotations from %s", ann_url)
        urllib.request.urlretrieve(ann_url, ann_path)

    with open(ann_path) as f:
        all_annotations = json.load(f)

    annotations: dict[str, list[int]] = all_annotations.get(name, {})

    # --- Consensus CPs: majority vote with tolerance clustering ---
    all_cps: list[int] = []
    for annotator_cps in annotations.values():
        all_cps.extend(annotator_cps)

    cp_counts = Counter(all_cps)
    tolerance = max(1, int(0.02 * len(y)))
    sorted_cps = sorted(cp_counts.keys())
    consensus_cps: list[int] = []
    used: set[int] = set()

    for cp in sorted_cps:
        if cp in used:
            continue
        cluster = [c for c in sorted_cps if abs(c - cp) <= tolerance and c not in used]
        total_votes = sum(cp_counts[c] for c in cluster)
        # Majority: >= half the annotators (rounded up)
        n_ann = max(len(annotations), 1)
        if total_votes >= max(3, n_ann // 2 + 1):
            consensus_cps.append(int(np.median(cluster)))
            used.update(cluster)

    consensus_arr = np.array(sorted(consensus_cps), dtype=int)

    logger.info(
        "TCPD '%s': n=%d, %d annotators, %d consensus CPs at %s",
        name, len(y), len(annotations), len(consensus_arr), consensus_arr.tolist(),
    )

    return {
        "y": y,
        "annotations": annotations,
        "consensus_cps": consensus_arr,
        "n_annotators": len(annotations),
        "name": name,
        "longname": data.get("longname", name),
    }


def load_oilwell_data(
    path: str | Path = "data/raw/oilwell.csv",
    train_fraction: float = 0.75,
    random_seed: int = 42,
) -> dict:
    """Load (or generate) oil-well drilling rate dataset.

    If ``path`` exists, loads it directly with no ground-truth changepoints.
    Otherwise generates a synthetic surrogate with 8 changepoints.

    Returns
    -------
    dict with keys: ``y_train``, ``y_test``, ``true_cps_train``,
    ``true_cps_test``, ``split_index``, ``name``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _OIL_N = 4000
    _OIL_TRUE_CPS = np.array([400, 800, 1200, 1600, 2000, 2500, 3000, 3500], dtype=int)

    if path.exists():
        logger.info("Loading oil-well data from: %s", path)
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values.astype(float)
        known_cps = np.array([], dtype=int)
        logger.info("Loaded %d oil-well observations", len(y))
    else:
        logger.info("Oil-well CSV not found at %s — generating synthetic surrogate.", path)
        rng = np.random.default_rng(random_seed)
        boundaries = np.concatenate([[0], _OIL_TRUE_CPS, [_OIL_N]])
        mu = np.empty(_OIL_N)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            mu[start:end] = rng.uniform(50_000, 120_000)
        v = rng.normal(0, 2000.0, size=_OIL_N)
        eps = np.empty(_OIL_N)
        eps[0] = v[0]
        for t in range(1, _OIL_N):
            eps[t] = 0.7 * eps[t - 1] + v[t]
        y = mu + eps
        # Inject 12 outliers
        outlier_idx = rng.choice(np.arange(10, _OIL_N - 10), size=12, replace=False)
        signs = rng.choice([-1, 1], size=12)
        y[outlier_idx] += signs * 20_000.0
        pd.DataFrame({"oilwell": y}).to_csv(path, index=False)
        known_cps = _OIL_TRUE_CPS.copy()
        logger.info("Synthetic oil-well data saved to %s", path)

    split = int(len(y) * train_fraction)
    y_train = y[:split]
    y_test = y[split:]

    train_mask = known_cps < split
    test_mask = known_cps >= split
    true_cps_train = known_cps[train_mask]
    true_cps_test = known_cps[test_mask] - split

    logger.info(
        "Oil-well split — n=%d, train=%d (%.0f%%), test=%d (%.0f%%)",
        len(y), len(y_train), 100 * len(y_train) / len(y),
        len(y_test), 100 * len(y_test) / len(y),
    )
    return {
        "y_train": y_train,
        "y_test": y_test,
        "true_cps_train": true_cps_train,
        "true_cps_test": true_cps_test,
        "split_index": split,
        "name": "oilwell",
    }


def load_brent_crude(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    train_end_date: str = "2024-12-31",
    cache_path: str | Path = "data/raw/brent_crude.csv",
) -> dict:
    """Load Brent crude oil futures as log-returns, split chronologically.

    Downloads via yfinance (ticker ``BZ=F``) and caches to ``cache_path``.
    The signal used for changepoint detection is the log-return series
    ``r_t = log(P_t) - log(P_{t-1})``.

    Returns
    -------
    dict with keys: ``y_train``, ``y_test``, ``true_cps_train``,
    ``true_cps_test``, ``split_index``, ``dates_train``, ``dates_test``,
    ``price_train``, ``price_test``, ``is_financial``, ``name``.
    """
    import yfinance as yf

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Loading Brent crude from cache: %s", cache_path)
        # yfinance >= 0.2 writes extra ticker/metadata rows; skip them
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True, skiprows=lambda i: i in (1, 2))
        # Drop any non-numeric rows that slipped through
        df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
    else:
        logger.info("Downloading Brent crude futures (BZ=F) from yfinance...")
        raw = yf.download("BZ=F", start=start_date, end=end_date, auto_adjust=True)
        # Flatten multi-level columns if present (yfinance >= 0.2)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close"]].dropna()
        df.to_csv(cache_path)
        logger.info("Cached Brent crude data to %s (%d rows)", cache_path, len(df))

    prices = df.iloc[:, 0].values.astype(float)
    dates = pd.to_datetime(df.index, errors="coerce")
    # Drop any rows where dates couldn't be parsed (stray metadata rows)
    valid = ~pd.isnull(dates)
    prices = prices[valid]
    dates = dates[valid]

    # Log returns
    log_returns = np.diff(np.log(prices))
    dates_ret = dates[1:]  # one shorter after diff

    # Split at train_end_date
    train_end = pd.Timestamp(train_end_date)
    train_mask = dates_ret <= train_end
    split = int(train_mask.sum())

    y_train = log_returns[:split]
    y_test = log_returns[split:]
    dates_train = dates_ret[:split]
    dates_test = dates_ret[split:]
    # Prices aligned with log-return dates (offset by 1 from raw prices)
    price_train = prices[1:split + 1]
    price_test = prices[split + 1:]

    logger.info(
        "Brent crude log-returns: n=%d, train=%d (up to %s), test=%d",
        len(log_returns), split, train_end_date, len(y_test),
    )
    return {
        "y_train": y_train,
        "y_test": y_test,
        "true_cps_train": np.array([], dtype=int),
        "true_cps_test": np.array([], dtype=int),
        "split_index": split,
        "dates_train": dates_train,
        "dates_test": dates_test,
        "price_train": price_train,
        "price_test": price_test,
        "is_financial": True,
        "name": "brent_crude",
    }


def load_us_ip_growth(series_id="INDPRO", start_date="2000-01-01",
                       end_date="2026-01-01", train_end_date="2023-12-01",
                       cache_path="data/raw"):
    """Load US Industrial Production Index and compute monthly growth rate.

    Source: FRED (Federal Reserve Economic Data)
    Series: INDPRO — Industrial Production: Total Index (SA, 2017=100)

    The pipeline operates on MONTH-OVER-MONTH GROWTH RATES (%), not the
    index level. Growth rates are approximately stationary within regimes,
    with regime shifts at recession onsets/recoveries.

    NBER recession dates serve as ground-truth changepoints:
    - Mar 2001 (recession start)
    - Nov 2001 (recession end / recovery start)
    - Dec 2007 (Great Recession start)
    - Jun 2009 (recovery start)
    - Feb 2020 (COVID recession start)
    - Apr 2020 (recovery start)

    Returns dict compatible with the pipeline.
    """
    import os as _os

    _os.makedirs(cache_path, exist_ok=True)
    csv_path = _os.path.join(cache_path, "us_indpro.csv")

    if _os.path.exists(csv_path):
        # Handle multiple possible column name conventions in cached file
        raw = pd.read_csv(csv_path, nrows=0)
        date_col = next((c for c in raw.columns if c.lower() in ('date', 'observation_date')), None)
        if date_col:
            df = pd.read_csv(csv_path, parse_dates=[date_col], index_col=date_col)
            df.index.name = 'date'
        else:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = 'date'
        logger.info("US IP loaded from cache: %d observations", len(df))
    else:
        # Try fredapi first, fall back to direct CSV download from FRED
        try:
            from fredapi import Fred
            api_key = _os.environ.get('FRED_API_KEY', None)
            if api_key:
                fred = Fred(api_key=api_key)
                series = fred.get_series(series_id, observation_start=start_date,
                                          observation_end=end_date)
                df = pd.DataFrame({'value': series})
                df.index.name = 'date'
            else:
                raise ValueError("No FRED API key")
        except (ImportError, ValueError, Exception) as e:
            logger.info("fredapi not available (%s), downloading CSV from FRED...", e)
            import urllib.request as _ur
            url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?"
                   f"id={series_id}&cosd={start_date}&coed={end_date}")
            _ur.urlretrieve(url, csv_path)
            raw_dl = pd.read_csv(csv_path, nrows=0)
            date_col = next(
                (c for c in raw_dl.columns if c.lower() in ('date', 'observation_date')), None
            )
            if date_col is None:
                raise ValueError(f"Cannot find date column in FRED CSV; columns={list(raw_dl.columns)}")
            df = pd.read_csv(csv_path, parse_dates=[date_col])
            df = df.rename(columns={date_col: 'date', series_id: 'value'})
            df = df.set_index('date')

        df.to_csv(csv_path)
        logger.info("US IP downloaded and cached: %d observations", len(df))

    if 'value' not in df.columns and len(df.columns) == 1:
        df.columns = ['value']

    # Compute month-over-month growth rate (%)
    df['growth_rate'] = df['value'].pct_change() * 100
    df = df.dropna()

    # NBER recession-based ground truth changepoints
    nber_dates = [
        '2001-03-01',  # recession start
        '2001-11-01',  # recovery
        '2007-12-01',  # Great Recession start
        '2009-06-01',  # recovery
        '2020-02-01',  # COVID start
        '2020-04-01',  # COVID recovery
    ]

    # Convert dates to integer indices in the growth rate series
    nber_indices = []
    for date_str in nber_dates:
        target = pd.Timestamp(date_str)
        if target >= df.index[0] and target <= df.index[-1]:
            idx = df.index.searchsorted(target)
            if idx < len(df):
                nber_indices.append(idx)
    nber_indices = np.array(sorted(set(nber_indices)))

    # Chronological split
    train_end = pd.Timestamp(train_end_date)
    train_mask = df.index <= train_end

    y_train = df.loc[train_mask, 'growth_rate'].values
    y_test = df.loc[~train_mask, 'growth_rate'].values
    dates_train = df.index[train_mask]
    dates_test = df.index[~train_mask]
    index_train = df.loc[train_mask, 'value'].values
    index_test = df.loc[~train_mask, 'value'].values

    # Split ground truth CPs
    split_idx = len(y_train)
    cps_train = nber_indices[nber_indices < split_idx]
    cps_test_abs = nber_indices[nber_indices >= split_idx]
    cps_test_rel = cps_test_abs - split_idx

    logger.info(
        "US IP growth rate: train=%d months, test=%d months",
        len(y_train), len(y_test),
    )
    logger.info("NBER ground truth: %d CPs total, %d in train, %d in test",
                len(nber_indices), len(cps_train), len(cps_test_rel))

    return {
        'y_train': y_train,
        'y_test': y_test,
        'true_cps_train': cps_train,
        'true_cps_test': cps_test_rel,
        'dates_train': dates_train,
        'dates_test': dates_test,
        'index_train': index_train,
        'index_test': index_test,
        'split_index': split_idx,
        'name': 'us_ip_growth',
        'longname': 'US Industrial Production (monthly growth rate)',
        'nber_dates': nber_dates,
        'is_financial': False,
    }


def load_tcpd_for_pipeline(
    name: str,
    train_fraction: float = 0.75,
    cache_dir: str = "data/raw/tcpd",
) -> dict:
    """Load a TCPD series and split chronologically for the pipeline.

    Parameters
    ----------
    name:
        TCPD dataset name.
    train_fraction:
        Fraction of observations used for training (chronological).

    Returns
    -------
    dict with keys: ``y_train``, ``y_test``, ``true_cps_train``,
    ``true_cps_test`` (test-relative), ``true_cps_test_abs``,
    ``split_index``, ``metadata``.
    """
    data = load_tcpd_series(name, cache_dir=cache_dir)
    y = data["y"]
    split = int(len(y) * train_fraction)

    y_train = y[:split]
    y_test = y[split:]

    cps = data["consensus_cps"]
    train_mask = cps < split
    test_mask = cps >= split

    true_cps_train = cps[train_mask]
    true_cps_test_abs = cps[test_mask]
    true_cps_test_rel = true_cps_test_abs - split

    logger.info(
        "TCPD '%s' split — n=%d, train=%d (%.0f%%), test=%d (%.0f%%); "
        "train CPs: %d, test CPs: %d",
        name, len(y),
        len(y_train), 100 * len(y_train) / len(y),
        len(y_test), 100 * len(y_test) / len(y),
        len(true_cps_train), len(true_cps_test_rel),
    )

    return {
        "y_train": y_train,
        "y_test": y_test,
        "true_cps_train": true_cps_train,
        "true_cps_test": true_cps_test_rel,
        "true_cps_test_abs": true_cps_test_abs,
        "split_index": split,
        "metadata": data,
    }
