"""Mean Run Length (MRL) index and composite risk metrics.

Provides functions to compute the MRL, the cost-weighted risk R, and
the censored variant R-tilde used in the sensitivity analysis.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_mrl(
    detected_cps: np.ndarray,
    true_cp: int | float,
    tolerance: int | None = None,
) -> dict:
    """Compute false positives and Mean Run Length for a single changepoint.

    A detected changepoint is a **true detection** if it is >= ``true_cp``
    (and within ``tolerance`` if supplied).  All detections strictly before
    ``true_cp`` (and outside the tolerance window) are **false positives**.

    Parameters
    ----------
    detected_cps:
        Array of detected changepoint indices (need not be sorted).
    true_cp:
        Index of the true (single) changepoint.
    tolerance:
        If given, detections in ``[true_cp - tolerance, true_cp + tolerance]``
        are counted as true detections rather than false positives.

    Returns
    -------
    dict with keys:
        - ``'FP'``: int — number of false positives.
        - ``'MRL'``: float — ``T_first - true_cp``, or ``np.inf`` if nothing
          was detected at or after ``true_cp``.
        - ``'T_first'``: float — index of the first true detection, or
          ``np.inf``.
    """
    detected_cps = np.asarray(detected_cps, dtype=float)
    true_cp = float(true_cp)

    if tolerance is not None:
        tol = float(tolerance)
        lo = true_cp - tol
        hi = true_cp + tol
        true_detections = detected_cps[(detected_cps >= lo) & (detected_cps <= hi)]
    else:
        lo = true_cp
        # No upper limit: any detection at or after true_cp counts
        true_detections = detected_cps[detected_cps >= lo]

    # False positives: detected before the (tolerance-adjusted) lower bound
    fp_mask = detected_cps < lo
    FP = int(fp_mask.sum())

    if len(true_detections) > 0:
        T_first = float(true_detections.min())
        MRL = T_first - true_cp
    else:
        T_first = np.inf
        MRL = np.inf

    logger.debug("MRL — FP=%d, MRL=%s, T_first=%s", FP, MRL, T_first)
    return {"FP": FP, "MRL": MRL, "T_first": T_first}


def compute_risk(
    FP: int,
    MRL: float,
    cF: float,
    cD: float,
) -> float:
    """Compute the cost-weighted risk R = (cF * FP) / (cD * MRL).

    Parameters
    ----------
    FP:
        Number of false positives.
    MRL:
        Mean Run Length (detection delay).
    cF:
        Cost per false positive.
    cD:
        Cost per unit of detection delay.

    Returns
    -------
    float
        Risk R.  Returns ``np.inf`` if ``MRL`` is 0 or ``np.inf``.
    """
    MRL = float(MRL)
    if MRL == 0.0 or not np.isfinite(MRL):
        return np.inf
    return float(cF * FP) / float(cD * MRL)


def compute_censored_mrl(
    MRL: float,
    epsilon: float,
    Tmax: float,
) -> float:
    """Censor the MRL into a finite, bounded range.

    - ``MRL == 0``   → ``epsilon``
    - ``MRL == inf`` → ``Tmax``
    - Otherwise      → ``MRL``

    Parameters
    ----------
    MRL:
        Raw MRL value.
    epsilon:
        Lower censoring bound (replaces zero delays).
    Tmax:
        Upper censoring bound (replaces infinite delays / missed detections).

    Returns
    -------
    float
    """
    MRL = float(MRL)
    if MRL == 0.0:
        return float(epsilon)
    if not np.isfinite(MRL):
        return float(Tmax)
    return MRL


def compute_censored_risk(
    FP: int,
    MRL: float,
    cF: float,
    cD: float,
    epsilon: float,
    Tmax: float,
) -> float:
    """Compute the censored risk R-tilde = (cF * FP) / (cD * MRL_censored).

    Parameters
    ----------
    FP, MRL, cF, cD:
        See :func:`compute_risk`.
    epsilon, Tmax:
        See :func:`compute_censored_mrl`.

    Returns
    -------
    float
    """
    mrl_c = compute_censored_mrl(MRL, epsilon, Tmax)
    return float(cF * FP) / float(cD * mrl_c)
