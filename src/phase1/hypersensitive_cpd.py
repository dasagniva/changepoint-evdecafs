"""Hypersensitive online changepoint detectors (X Algorithm).

Implements two complementary online detectors used for post-hoc relabelling
of EV-DeCAFS detected changepoints:

  - BOCPD: Bayesian Online Changepoint Detection with Normal-Gamma conjugate prior
  - CUSUM: Two-sided CUSUM with a 1-sigma control limit (hypersensitive)

Both detectors are consistent with the AR(1) noise model used in Phase I
(parameters phi and sigma_v are passed in).
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# BOCPD — Bayesian Online Changepoint Detection
# ---------------------------------------------------------------------------

def run_bocpd(
    y: np.ndarray,
    phi: float,
    sigma_v: float,
    threshold: float = 0.5,
    *,
    # Normal-Gamma prior hyperparameters (weakly informative defaults)
    mu0: float | None = None,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float | None = None,
) -> np.ndarray:
    """Run Bayesian Online Changepoint Detection on a univariate series.

    Uses a Gaussian likelihood with Normal-Gamma conjugate prior.  The AR(1)
    autocorrelation is incorporated by pre-whitening the series:

        z_t = y_t - phi * y_{t-1}   (for t >= 1)

    so the whitened residuals are approximately i.i.d. N(mu, sigma_v^2).

    The algorithm maintains a run-length distribution P(r_t | y_{1:t}) where
    r_t is the number of observations since the last changepoint.  At each
    step the predictive probability under the current run is computed and the
    posterior is updated; the probability of a CP at t is:

        P(CP at t) = sum_{r} P(r_{t-1} = r) * H

    where H is the hazard rate (set to 1/n for a uniform-duration prior).

    Parameters
    ----------
    y:
        Univariate time series, shape (n,).
    phi:
        AR(1) autocorrelation coefficient (used for pre-whitening).
    sigma_v:
        AR(1) innovation standard deviation.
    threshold:
        Flag index t if P(CP at t) > threshold.  Default 0.5.
    mu0:
        Prior mean for the Normal component.  Defaults to mean(y).
    kappa0:
        Prior pseudo-count for the mean.  Default 1.0.
    alpha0:
        Prior shape for the Gamma (precision) component.  Default 1.0.
    beta0:
        Prior rate for the Gamma component.  Defaults to var(y)/2.

    Returns
    -------
    bocpd_flags : np.ndarray of bool, shape (n,)
        True at each index where P(CP) > threshold.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 2:
        return np.zeros(n, dtype=bool)

    # Pre-whiten: remove AR(1) autocorrelation
    z = np.empty(n)
    z[0] = y[0]
    z[1:] = y[1:] - phi * y[:-1]

    # Prior defaults
    if mu0 is None:
        mu0 = float(np.mean(z))
    if beta0 is None:
        beta0 = max(float(np.var(z)) / 2.0, 1e-6)

    # Hazard: constant rate H = 1/n (geometric run-length prior)
    H = 1.0 / n

    # Run-length distribution: R[r] = P(r_t = r | y_{1:t})
    # Start with P(r_0 = 0) = 1 (certain changepoint at t=0)
    R = np.array([1.0])

    # Sufficient statistics for Normal-Gamma conjugate update per run length r:
    # mu_r, kappa_r, alpha_r, beta_r
    # For r=0 (just started): use the prior
    kappa_r = np.array([kappa0])
    mu_r = np.array([mu0])
    alpha_r = np.array([alpha0])
    beta_r = np.array([beta0])

    cp_probs = np.zeros(n)

    for t in range(1, n):
        x = z[t]

        # Predictive probability under each run length: Student-t
        # p(x | r) = StudentT(2*alpha_r, mu_r, beta_r*(kappa_r+1)/(alpha_r*kappa_r))
        df = 2.0 * alpha_r
        scale_sq = beta_r * (kappa_r + 1.0) / (alpha_r * kappa_r + 1e-12)
        scale_sq = np.maximum(scale_sq, 1e-12)

        # Log Student-t pdf (vectorised over run lengths)
        log_pred = _log_student_t(x, df, mu_r, scale_sq)
        # Shift for numerical stability before exponentiation
        log_pred_shifted = log_pred - log_pred.max()
        pred = np.exp(log_pred_shifted)

        # Prior predictive for the new run (CP case) — uses hyperprior params,
        # not the run-specific posteriors.  Computed in log-space then shifted
        # by the same offset so probabilities are on a comparable scale.
        df0 = 2.0 * alpha0
        scale_sq0 = max(float(beta0 * (kappa0 + 1.0) / (alpha0 * kappa0 + 1e-12)), 1e-12)
        log_pred_prior = float(_log_student_t(x, np.array([df0]), np.array([mu0]),
                                               np.array([scale_sq0]))[0])
        pred_prior = float(np.exp(log_pred_prior - log_pred.max()))

        # Growth: existing runs extend by one step
        R_growth = R * pred * (1.0 - H)

        # Changepoint: new run r=0 starts — use the prior predictive
        cp_prob = H * pred_prior

        # Append the new run r=0 at the start
        R_new = np.concatenate([[cp_prob], R_growth])

        # Normalise — R_new[0] is now the posterior P(r_t=0 | y_{1:t})
        Z = R_new.sum()
        if Z > 1e-300:
            R_new = R_new / Z
        else:
            R_new = np.zeros_like(R_new)
            R_new[0] = 1.0

        cp_probs[t] = float(R_new[0])

        # Update sufficient statistics for the new run (prior)
        mu_r_new = np.concatenate([[mu0], _update_mu(mu_r, kappa_r, x)])
        kappa_r_new = np.concatenate([[kappa0], kappa_r + 1.0])
        alpha_r_new = np.concatenate([[alpha0], alpha_r + 0.5])
        beta_r_new = np.concatenate([[beta0],
            beta_r + 0.5 * kappa_r / (kappa_r + 1.0) * (x - mu_r) ** 2])

        R = R_new
        mu_r = mu_r_new
        kappa_r = kappa_r_new
        alpha_r = alpha_r_new
        beta_r = beta_r_new

        # Truncate to keep memory bounded (keep top-probability run lengths)
        if len(R) > 500:
            keep = np.argsort(R)[-500:]
            keep = np.sort(keep)
            R = R[keep]
            R /= R.sum()
            mu_r = mu_r[keep]
            kappa_r = kappa_r[keep]
            alpha_r = alpha_r[keep]
            beta_r = beta_r[keep]

    bocpd_flags = cp_probs > threshold

    n_flagged = int(bocpd_flags.sum())
    logger.debug(
        "BOCPD — %d flags (threshold=%.2f, n=%d)",
        n_flagged, threshold, n,
    )
    return bocpd_flags


def _log_student_t(
    x: float,
    df: np.ndarray,
    mu: np.ndarray,
    scale_sq: np.ndarray,
) -> np.ndarray:
    """Log pdf of a Student-t distribution (vectorised over parameters)."""
    import math
    from scipy.special import gammaln

    z = (x - mu) ** 2 / (scale_sq + 1e-12)
    log_p = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(np.pi * df * scale_sq + 1e-300)
        - 0.5 * (df + 1.0) * np.log1p(z / (df + 1e-12))
    )
    return log_p


def _update_mu(
    mu_r: np.ndarray,
    kappa_r: np.ndarray,
    x: float,
) -> np.ndarray:
    """Bayesian update for the Normal mean given a new observation."""
    return (kappa_r * mu_r + x) / (kappa_r + 1.0)


# ---------------------------------------------------------------------------
# CUSUM — Two-sided CUSUM
# ---------------------------------------------------------------------------

def run_cusum(
    y: np.ndarray,
    phi: float,
    sigma_v: float,
    h_multiplier: float = 1.0,
) -> np.ndarray:
    """Run two-sided CUSUM on a univariate series (hypersensitive variant).

    The control limit is ``h = h_multiplier * sigma_v``.  With the default
    ``h_multiplier=1.0`` this is a 1-sigma threshold, making the detector
    highly sensitive (intended for relabelling, not primary detection).

    AR(1) autocorrelation is removed by pre-whitening:

        z_t = y_t - phi * y_{t-1}

    The CUSUM accumulates standardised deviations ``(z_t - mu_z) / sigma_v``
    and resets after each crossing of the control limit.

    Parameters
    ----------
    y:
        Univariate time series, shape (n,).
    phi:
        AR(1) autocorrelation coefficient.
    sigma_v:
        AR(1) innovation standard deviation.
    h_multiplier:
        Control limit as a multiple of sigma_v.  Default 1.0 (1-sigma).

    Returns
    -------
    cusum_flags : np.ndarray of bool, shape (n,)
        True at each index where a CUSUM crossing is declared.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 2:
        return np.zeros(n, dtype=bool)

    sigma_v = max(float(sigma_v), 1e-12)
    h = h_multiplier * sigma_v

    # Pre-whiten
    z = np.empty(n)
    z[0] = y[0]
    z[1:] = y[1:] - phi * y[:-1]

    mu_z = float(np.mean(z))

    cusum_flags = np.zeros(n, dtype=bool)
    S_pos = 0.0  # upper CUSUM statistic
    S_neg = 0.0  # lower CUSUM statistic

    for t in range(n):
        dev = (z[t] - mu_z) / sigma_v
        S_pos = max(0.0, S_pos + dev)
        S_neg = max(0.0, S_neg - dev)

        if S_pos > h or S_neg > h:
            cusum_flags[t] = True
            S_pos = 0.0
            S_neg = 0.0

    n_flagged = int(cusum_flags.sum())
    logger.debug(
        "CUSUM — %d flags (h=%.2f*sigma_v=%.2f, n=%d)",
        n_flagged, h_multiplier, h, n,
    )
    return cusum_flags
