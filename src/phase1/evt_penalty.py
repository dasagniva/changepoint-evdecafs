"""EVT-based adaptive penalty computation for EV-DeCAFS.

Implements the EVI (Extreme Value Index) field and two variants of the
adaptive penalty schedule: GPD-based and exceedance-count-based.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_evi_field(
    y: np.ndarray,
    w: int = 50,
    q0: float = 0.90,
    min_exceedances: int = 5,
) -> np.ndarray:
    """Compute the local Extreme Value Index (xi_t) at each time point.

    For each time ``t``, a local window ``W_t`` of half-width ``w`` is formed.
    Deviations ``|y_s - mean(W_t)|`` are thresholded at their ``q0``-th
    percentile; the GPD shape parameter ``xi_t`` is estimated via the
    method-of-moments estimator (closed form, ~100x faster than MLE):

        xi_MOM = 0.5 * (1 - 1 / (Var / Mean^2))

    where Mean and Var are the sample mean and variance of the exceedances.

    If a window contains fewer than ``min_exceedances`` points, ``xi_t = 0``.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.
    w:
        Half-width of the local window.
    q0:
        Percentile threshold for identifying GPD exceedances (0–1).
    min_exceedances:
        Minimum number of exceedances required to estimate xi.

    Returns
    -------
    xi_field : np.ndarray, shape ``(n,)``
        Local EVI estimates.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    xi_field = np.zeros(n)
    n_successful = 0
    t_start = time.perf_counter()

    # Pad array to avoid per-step boundary checks
    y_padded = np.pad(y, w, mode="reflect")

    for t in tqdm(range(n), desc="EVI field", unit="step", leave=False):
        window = y_padded[t : t + 2 * w + 1]  # length 2w+1
        mu_W = np.mean(window)
        deviations = np.abs(window - mu_W)
        threshold = np.percentile(deviations, q0 * 100.0)

        exceedances = deviations[deviations > threshold] - threshold
        if len(exceedances) < min_exceedances:
            xi_field[t] = 0.0
            continue

        # Method-of-moments GPD estimator (closed form):
        # For GPD(xi, beta): Var/Mean^2 = 1/(1-2*xi)  =>  xi = 0.5*(1 - 1/ratio)
        m = np.mean(exceedances)
        if m <= 0.0:
            xi_field[t] = 0.0
            continue
        v = np.var(exceedances)
        ratio = v / (m * m)
        xi_mom = 0.5 * (1.0 - 1.0 / max(ratio, 1e-10))
        xi_field[t] = float(np.clip(xi_mom, -1.0, 2.0))
        n_successful += 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "EVI field computed via method-of-moments in %.2fs (n=%d, w=%d); "
        "%d/%d windows estimated, mean xi=%.4f",
        elapsed,
        n,
        w,
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

    ``alpha_t = alpha_0 * (1 + lambda_ev * |xi_t|)``

    The EVI xi measures departure from exponential-tail behavior (xi = 0).
    Heavy tails (xi > 0) indicate potential outlier clusters; sharply bounded
    tails (xi < 0) near spike outliers indicate abrupt distributional
    truncation, which is itself a signature of regime disturbance. Using |xi|
    means ANY non-exponential local tail structure raises the penalty, making
    the algorithm more conservative in regions of unusual distributional
    behavior regardless of tail direction.

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
    alpha_t = alpha_0 * (1.0 + lambda_ev * np.abs(xi_field))
    logger.info(
        "Adaptive penalty: mean |xi|=%.4f, alpha_t range=[%.2f, %.2f], "
        "mean=%.2f, fraction elevated above alpha_0=%.2f%%",
        float(np.mean(np.abs(xi_field))),
        float(alpha_t.min()),
        float(alpha_t.max()),
        float(alpha_t.mean()),
        float(np.mean(alpha_t > alpha_0 * 1.01)) * 100,
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
