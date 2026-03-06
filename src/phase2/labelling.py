"""Changepoint labelling (Algorithm 3).

Labels each detected changepoint as 'sustained' (1) or 'recoiled' (0)
based on magnitude and persistence thresholds.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_kappa_mu(X: np.ndarray, percentile: float = 75) -> float:
    """Compute the magnitude threshold kappa_mu.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(m, 4)``.  The first column is ``delta_mu``.
    percentile:
        Percentile of ``|delta_mu|`` used as threshold.

    Returns
    -------
    float
        kappa_mu threshold.
    """
    X = np.asarray(X, dtype=float)
    kappa_mu = float(np.percentile(np.abs(X[:, 0]), percentile))
    logger.debug("kappa_mu (%.0fth pct of |delta_mu|) = %.4f", percentile, kappa_mu)
    return kappa_mu


def label_changepoints(
    X: np.ndarray,
    kappa_mu: float,
    kappa_S: float = 0.5,
) -> np.ndarray:
    """Assign binary labels to changepoints (Algorithm 3).

    A changepoint is labelled **sustained** (1) if::

        |delta_mu| > kappa_mu  AND  S > kappa_S

    Otherwise it is labelled **recoiled** (0).

    Parameters
    ----------
    X:
        Feature matrix, shape ``(m, 4)``.  Columns: [delta_mu, S, ...].
    kappa_mu:
        Magnitude threshold (from :func:`compute_kappa_mu`).
    kappa_S:
        Persistence threshold.

    Returns
    -------
    labels : np.ndarray of int, shape ``(m,)``
        Binary class labels (0 = recoiled, 1 = sustained).
    """
    X = np.asarray(X, dtype=float)
    sustained = (np.abs(X[:, 0]) > kappa_mu) & (X[:, 1] > kappa_S)
    labels = sustained.astype(int)

    n_sustained = int(labels.sum())
    n_recoiled = len(labels) - n_sustained
    logger.info(
        "Labelling — %d sustained, %d recoiled (kappa_mu=%.4f, kappa_S=%.4f)",
        n_sustained,
        n_recoiled,
        kappa_mu,
        kappa_S,
    )
    return labels
