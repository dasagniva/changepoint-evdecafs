"""Mean Run Length (MRL) index and composite risk metrics.

Provides functions to compute the MRL, the cost-weighted risk R, and
the censored variant R-tilde used in the sensitivity analysis.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def convert_to_relative(
    true_cps_absolute: np.ndarray,
    split_index: int,
) -> tuple[np.ndarray, int]:
    """Convert absolute true CP indices to test-relative coordinates.

    Parameters
    ----------
    true_cps_absolute:
        Array of true CP indices in the full series.
    split_index:
        The index where the test set begins (e.g., ``len(y_train)``).

    Returns
    -------
    true_cps_relative:
        Array of true CPs that fall in the test set, converted to
        test-relative coordinates (i.e., index - split_index).
    n_excluded:
        Number of true CPs that fall before the split (excluded).
    """
    true_cps_absolute = np.asarray(true_cps_absolute)
    mask = true_cps_absolute >= split_index
    true_cps_relative = true_cps_absolute[mask] - split_index
    n_excluded = int((~mask).sum())
    return true_cps_relative, n_excluded


def compute_mrl(
    detected_cps: np.ndarray,
    true_cps: np.ndarray,
    tolerance: int | None = None,
    boundary_exclusion_window: int = 0,
) -> dict:
    """Compute false positives and Mean Run Length across multiple changepoints.

    Both ``detected_cps`` and ``true_cps`` must be in the **same coordinate
    system** (e.g., both test-relative). Use :func:`convert_to_relative` to
    convert absolute true CP indices before calling this function.

    FP: number of detected CPs not matched to any true CP within tolerance.
    MRL: mean detection delay across true CPs (averaging only finite delays).

    Parameters
    ----------
    detected_cps:
        Array of detected changepoint indices (relative to eval segment).
    true_cps:
        Array of true CP indices (same coordinate system as detected_cps).
    tolerance:
        Matching window half-width.  A detected CP within ``tolerance`` steps
        of a true CP is considered a true detection.  Default: exact matching
        (tolerance = 0).
    boundary_exclusion_window:
        Detected CPs that fall within the first ``boundary_exclusion_window``
        observations of the evaluation segment are excluded from FP counting
        (treated as boundary artifacts).  Default 0 (disabled).

    Returns
    -------
    dict with keys:
        - ``'FP'``: int — detected CPs not matched to any true CP.
        - ``'MRL'``: float — mean delay to first detection at/after each true CP;
          ``np.inf`` if all true CPs are missed.
        - ``'delays'``: list[float] — per-true-CP delay (``np.inf`` if missed).
        - ``'n_missed'``: int — number of true CPs with no nearby detection.
        - ``'n_true_cps'``: int — total number of true CPs.
        - ``'n_detected'``: int — total number of detected CPs.
    """
    tol = float(tolerance) if tolerance is not None else 0.0
    detected_all = np.sort(np.asarray(detected_cps, dtype=float))
    true = np.sort(np.asarray(true_cps, dtype=float))

    # --- Boundary exclusion: remove detections near the start of the segment ---
    if boundary_exclusion_window > 0:
        boundary_mask = detected_all < boundary_exclusion_window
        n_excluded_boundary = int(boundary_mask.sum())
        if n_excluded_boundary > 0:
            logger.debug(
                "Boundary exclusion: %d detection(s) within first %d obs "
                "excluded from FP counting.",
                n_excluded_boundary, boundary_exclusion_window,
            )
        detected = detected_all[~boundary_mask]
    else:
        detected = detected_all

    # --- False Positives: detected CPs not near any true CP ---
    matched_detected: set[int] = set()
    for tcp in true:
        for i, dcp in enumerate(detected):
            if abs(dcp - tcp) <= tol:
                matched_detected.add(i)
                break  # each true CP matches at most one detected CP
    FP = len(detected) - len(matched_detected)

    # --- MRL: per true CP, delay to first detection at or after ---
    delays: list[float] = []
    for tcp in true:
        post = detected[detected >= tcp - tol]
        if len(post) > 0:
            delay = float(post[0] - tcp)
            delays.append(max(delay, 0.0))
        else:
            delays.append(np.inf)

    finite_delays = [d for d in delays if np.isfinite(d)]
    MRL = float(np.mean(finite_delays)) if finite_delays else np.inf
    n_missed = sum(1 for d in delays if np.isinf(d))

    logger.debug(
        "MRL — FP=%d, MRL=%s, n_missed=%d/%d, n_detected=%d",
        FP, MRL, n_missed, len(true), len(detected),
    )
    return {
        "FP": FP,
        "MRL": MRL,
        "delays": delays,
        "n_missed": n_missed,
        "n_true_cps": len(true),
        "n_detected": len(detected),
        "n_detected_total": len(detected_all),
    }


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
