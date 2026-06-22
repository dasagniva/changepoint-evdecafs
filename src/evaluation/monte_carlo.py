"""Monte Carlo coverage simulation for robust FPNN evaluation.

Generates B synthetic series with known changepoints and outliers,
runs the full pipeline on each, and reports the distribution of
classification metrics across replications.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np

logger = logging.getLogger(__name__)


def generate_synthetic_series(
    n: int = 2000,
    n_changepoints: int = 8,
    n_outliers: int = 15,
    phi: float = 0.5,
    sigma_v: float = 2000.0,
    sigma_eta: float = 100.0,
    jump_magnitude_range: tuple = (5000, 30000),
    outlier_magnitude_range: tuple = (20000, 40000),
    seed: int | None = None,
) -> Dict:
    """Generate a single synthetic series with known ground truth.

    Model: y_t = mu_t + epsilon_t
           mu_t = mu_{t-1} + eta_t + delta_t  (delta_t = jump at changepoint)
           epsilon_t = phi * epsilon_{t-1} + v_t,  v_t ~ N(0, sigma_v^2)

    Parameters
    ----------
    n:
        Series length.
    n_changepoints:
        Number of sustained mean-shift changepoints.
    n_outliers:
        Number of single-point outlier spikes (recoiled events).
    phi:
        AR(1) autocorrelation coefficient.
    sigma_v:
        AR(1) innovation standard deviation.
    sigma_eta:
        Standard deviation of the random drift term.
    jump_magnitude_range:
        (min, max) for uniform jump magnitudes.
    outlier_magnitude_range:
        (min, max) for uniform outlier spike magnitudes.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'y': observed series (n,)
        'mu': true mean signal with drift (n,)
        'true_changepoints': array of CP indices (sustained shifts)
        'true_outlier_indices': array of outlier spike indices
        'all_event_positions': sorted union of CPs and outlier indices
        'all_event_labels': 1=sustained, 0=recoiled for each event
        'params': dict of generating parameters
    """
    rng = np.random.default_rng(seed)

    # --- Mean signal with changepoints ---
    mu = np.zeros(n)
    cp_positions = np.sort(
        rng.choice(range(100, n - 100), size=n_changepoints, replace=False)
    )

    current_mean = float(rng.uniform(70000, 140000))
    mu[: cp_positions[0]] = current_mean
    for i, cp in enumerate(cp_positions):
        jump = float(rng.uniform(*jump_magnitude_range)) * float(
            rng.choice([-1, 1])
        )
        current_mean += jump
        next_cp = int(cp_positions[i + 1]) if i + 1 < len(cp_positions) else n
        mu[cp:next_cp] = current_mean

    # --- Add random drift ---
    eta = rng.normal(0, sigma_eta, n)
    eta[0] = 0.0
    mu_with_drift = mu + np.cumsum(eta)

    # --- AR(1) noise ---
    epsilon = np.zeros(n)
    v = rng.normal(0, sigma_v, n)
    for t in range(1, n):
        epsilon[t] = phi * epsilon[t - 1] + v[t]

    y = mu_with_drift + epsilon

    # --- Outlier spikes (recoiled shifts) ---
    forbidden = set(int(cp) for cp in cp_positions)
    candidates = [
        i for i in range(50, n - 50)
        if not any(abs(i - cp) < 20 for cp in cp_positions)
    ]
    n_outliers_actual = min(n_outliers, len(candidates), n // 20)
    outlier_positions = np.sort(
        rng.choice(candidates, size=n_outliers_actual, replace=False)
    )
    for op in outlier_positions:
        spike = float(rng.uniform(*outlier_magnitude_range)) * float(
            rng.choice([-1, 1])
        )
        y[op] += spike

    # --- Combine all events with labels ---
    all_event_positions = np.concatenate([cp_positions, outlier_positions])
    all_event_labels = np.concatenate(
        [
            np.ones(len(cp_positions), dtype=int),       # sustained
            np.zeros(len(outlier_positions), dtype=int),  # recoiled
        ]
    )
    sort_idx = np.argsort(all_event_positions)
    all_event_positions = all_event_positions[sort_idx]
    all_event_labels = all_event_labels[sort_idx]

    return {
        "y": y,
        "mu": mu_with_drift,
        "true_changepoints": cp_positions,
        "true_outlier_indices": outlier_positions,
        "all_event_positions": all_event_positions,
        "all_event_labels": all_event_labels,
        "params": {
            "n": n,
            "phi": phi,
            "sigma_v": sigma_v,
            "sigma_eta": sigma_eta,
            "n_changepoints": n_changepoints,
            "n_outliers": n_outliers_actual,
        },
    }


def assign_nearest_labels(
    detected_cps: np.ndarray,
    true_cps: np.ndarray,
    true_labels: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """For each detected CP, inherit the label of the nearest true event.

    If no true event falls within ``tolerance`` steps, the detected CP is
    considered spurious and assigned label 0 (recoiled).

    Parameters
    ----------
    detected_cps:
        Detected CP indices (relative to eval segment).
    true_cps:
        True event positions (same coordinate system).
    true_labels:
        Labels corresponding to true_cps (1=sustained, 0=recoiled).
    tolerance:
        Maximum distance for a match.

    Returns
    -------
    np.ndarray of int, shape ``(len(detected_cps),)``
    """
    labels = np.zeros(len(detected_cps), dtype=int)
    if len(true_cps) == 0:
        return labels
    for i, dcp in enumerate(detected_cps):
        distances = np.abs(true_cps - dcp)
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] <= tolerance:
            labels[i] = int(true_labels[nearest_idx])
    return labels


def run_monte_carlo(
    pipeline_func: Callable,
    B: int = 500,
    train_fraction: float = 0.75,
    series_params: dict | None = None,
    seed: int = 42,
) -> Dict:
    """Run B replications of the full pipeline on synthetic data.

    Parameters
    ----------
    pipeline_func:
        Callable with signature::

            pipeline_func(y_train, y_test,
                          true_cps_train, true_labels_train,
                          true_cps_test, true_labels_test)
            -> dict of metrics {'balanced_accuracy': float, ...}

    B:
        Number of Monte Carlo replications.
    train_fraction:
        Chronological train/test split fraction.
    series_params:
        Keyword arguments forwarded to :func:`generate_synthetic_series`.
    seed:
        Base random seed; replication ``i`` uses ``seed + i``.

    Returns
    -------
    dict
        Keys = metric names, values = dicts with keys:
        'values' (array length B), 'mean', 'std', 'ci_lower', 'ci_upper',
        'median'.
    """
    if series_params is None:
        series_params = {}

    all_metrics: list[dict] = []

    for i in range(B):
        if (i + 1) % 50 == 0:
            logger.info("Monte Carlo replication %d/%d", i + 1, B)

        data = generate_synthetic_series(seed=seed + i, **series_params)
        y = data["y"]
        n = len(y)
        split = int(n * train_fraction)

        y_train = y[:split]
        y_test = y[split:]

        all_pos = data["all_event_positions"]
        all_lab = data["all_event_labels"]

        train_mask = all_pos < split
        test_mask = all_pos >= split

        true_cps_train = all_pos[train_mask]
        true_labels_train = all_lab[train_mask]
        true_cps_test = all_pos[test_mask] - split  # test-relative
        true_labels_test = all_lab[test_mask]

        try:
            metrics = pipeline_func(
                y_train,
                y_test,
                true_cps_train,
                true_labels_train,
                true_cps_test,
                true_labels_test,
            )
            if metrics is not None:
                all_metrics.append(metrics)
        except Exception as e:
            logger.warning("Replication %d failed: %s", i + 1, e)
            continue

    if not all_metrics:
        raise RuntimeError("All Monte Carlo replications failed")

    metric_names = list(all_metrics[0].keys())
    results: Dict = {}
    for name in metric_names:
        values = np.array(
            [m[name] for m in all_metrics if m is not None and name in m], dtype=float
        )
        results[name] = {
            "values": values,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "ci_lower": float(np.nanpercentile(values, 2.5)),
            "ci_upper": float(np.nanpercentile(values, 97.5)),
            "median": float(np.nanmedian(values)),
        }

    logger.info(
        "Monte Carlo complete: %d/%d replications successful",
        len(all_metrics),
        B,
    )
    for name in metric_names:
        r = results[name]
        logger.info(
            "  %s: mean=%.4f [%.4f, %.4f]",
            name,
            r["mean"],
            r["ci_lower"],
            r["ci_upper"],
        )

    return results
