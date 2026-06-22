"""EVT-based adaptive penalty computation for EV-DeCAFS.

Implements the EVI (Extreme Value Index) field and two variants of the
adaptive penalty schedule: GPD-based and exceedance-count-based.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto
from tqdm import tqdm

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_evi_field(
    y: np.ndarray,
    w: int = 50,
    q0: float = 0.90,
) -> np.ndarray:
    """Compute the local Extreme Value Index (xi_t) at each time point.

    For each time ``t``, a local window ``W_t = y[max(0, t-w) : min(n, t+w+1)]``
    is formed.  Deviations ``|y_s - mean(W_t)|`` are thresholded at their
    ``q0``-th percentile; a GPD is fitted to the exceedances; the shape
    parameter ``xi_t`` is extracted.

    If a window contains fewer than 5 exceedances, ``xi_t`` is set to 0.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.
    w:
        Half-width of the local window.
    q0:
        Percentile threshold for identifying GPD exceedances (0–1).

    Returns
    -------
    xi_field : np.ndarray, shape ``(n,)``
        Local EVI estimates.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    xi_field = np.zeros(n)
    n_successful = 0

    for t in tqdm(range(n), desc="EVI field", unit="step", leave=False):
        lo = max(0, t - w)
        hi = min(n, t + w + 1)
        W = y[lo:hi]

        mu_W = np.mean(W)
        deviations = np.abs(W - mu_W)
        threshold = np.percentile(deviations, q0 * 100.0)

        exceedances = deviations[deviations > threshold] - threshold
        if len(exceedances) < 5:
            xi_field[t] = 0.0
            continue

        try:
            # floc=0: location is fixed at 0 (exceedances are already threshold-subtracted)
            xi, _loc, _scale = genpareto.fit(exceedances, floc=0)
            xi_field[t] = float(xi)
            n_successful += 1
        except Exception:
            xi_field[t] = 0.0

    logger.info(
        "EVI field — %d/%d windows with successful GPD fits, mean xi=%.4f",
        n_successful,
        n,
        float(np.mean(xi_field)),
    )
    return xi_field


def compute_adaptive_penalty(
    xi_field: np.ndarray,
    alpha_0: float,
    lambda_ev: float = 1.0,
) -> np.ndarray:
    """Compute GPD-based adaptive penalty schedule.

    ``alpha_t = alpha_0 * (1 + lambda_ev * max(xi_t, 0))``

    Parameters
    ----------
    xi_field:
        EVI field from :func:`compute_evi_field`, shape ``(n,)``.
    alpha_0:
        Base penalty value.
    lambda_ev:
        EVT sensitivity multiplier.

    Returns
    -------
    alpha_t : np.ndarray, shape ``(n,)``
    """
    xi_field = np.asarray(xi_field, dtype=float)
    alpha_t = alpha_0 * (1.0 + lambda_ev * np.maximum(xi_field, 0.0))
    logger.debug(
        "Adaptive penalty — min=%.4f, max=%.4f, mean=%.4f",
        alpha_t.min(),
        alpha_t.max(),
        alpha_t.mean(),
    )
    return alpha_t


def compute_exceedance_count_penalty(
    y: np.ndarray,
    mu_est: np.ndarray,
    sigma_v: float,
    w: int,
    c: float,
    alpha_0: float,
) -> np.ndarray:
    """Compute the exceedance-count variant of the adaptive penalty.

    ``alpha_t = alpha_0 * (1 + E_t / (2w+1))``

    where ``E_t`` is the number of observations in ``W_t`` that deviate from
    ``mu_est`` by more than ``c * sigma_v``.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.
    mu_est:
        Preliminary piecewise-constant mean estimate (e.g. from a standard
        PELT run), shape ``(n,)``.
    sigma_v:
        Noise standard deviation estimate.
    w:
        Half-width of the local window.
    c:
        Multiplier for the exceedance threshold (``c * sigma_v``).
    alpha_0:
        Base penalty value.

    Returns
    -------
    alpha_t : np.ndarray, shape ``(n,)``
    """
    y = np.asarray(y, dtype=float)
    mu_est = np.asarray(mu_est, dtype=float)
    n = len(y)
    nominal_window = 2 * w + 1
    threshold = c * sigma_v
    alpha_t = np.zeros(n)

    for t in range(n):
        lo = max(0, t - w)
        hi = min(n, t + w + 1)
        residuals = np.abs(y[lo:hi] - mu_est[lo:hi])
        E_t = int(np.sum(residuals > threshold))
        alpha_t[t] = alpha_0 * (1.0 + E_t / nominal_window)

    logger.debug(
        "Exceedance-count penalty — min=%.4f, max=%.4f, mean=%.4f",
        alpha_t.min(),
        alpha_t.max(),
        alpha_t.mean(),
    )
    return alpha_t
