"""Feature extraction from detected changepoints (Algorithm 2).

Extracts a 4-dimensional feature vector for each detected changepoint,
capturing the magnitude, persistence, local AR(1) coefficient, and
variance ratio of the shift.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

FEATURE_NAMES = ["delta_mu", "S", "phi_local", "V"]


def extract_features(
    y: np.ndarray,
    changepoints: np.ndarray,
    means: np.ndarray,
    L: int = 5,
    epsilon: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract a feature matrix for all detected changepoints (Algorithm 2).

    For each changepoint ``tau_i``:

    - ``delta_mu`` : ``mean(D_plus) - mean(D_minus)`` — signed magnitude of
      the level shift.
    - ``S`` (persistence) : fraction of post-change observations in ``D_plus``
      within ``epsilon`` of ``mean(D_plus)``.
    - ``phi_local`` : local AR(1) residual coefficient estimated from the
      post-change window residuals ``r_t = y_t - mu_t``.
    - ``V`` (variance ratio) : ``var(D_minus) / (var(D_plus) + 1e-10)``.

    Parameters
    ----------
    y:
        Univariate time series (training portion), shape ``(n,)``.
    changepoints:
        Indices of detected changepoints, shape ``(m,)``.
    means:
        Estimated piecewise-constant mean for each time point, shape ``(n,)``.
    L:
        Half-width of the feature extraction window.
        ``D_minus = y[tau-L : tau]``,  ``D_plus = y[tau+1 : tau+L+1]``.
    epsilon:
        Persistence tolerance.  If ``None``, computed as
        ``median(|y - means|)`` over the full series.

    Returns
    -------
    X : np.ndarray, shape ``(m, 4)``
        Feature matrix.  Columns: [delta_mu, S, phi_local, V].
    feature_names : list[str]
        Column labels ``['delta_mu', 'S', 'phi_local', 'V']``.
    """
    y = np.asarray(y, dtype=float)
    means = np.asarray(means, dtype=float)
    changepoints = np.asarray(changepoints, dtype=int)
    n = len(y)
    m = len(changepoints)

    if epsilon is None:
        epsilon = float(np.median(np.abs(y - means)))
        if epsilon < 1e-10:
            epsilon = 1e-10

    X = np.empty((m, 4), dtype=float)

    for i, tau in enumerate(changepoints):
        # Pre-change window D_minus = y[tau-L : tau]
        lo_minus = max(0, tau - L)
        D_minus = y[lo_minus:tau]

        # Post-change window D_plus = y[tau+1 : tau+L+1]
        hi_plus = min(n, tau + L + 1)
        D_plus = y[tau + 1 : hi_plus]

        # --- delta_mu ---
        mu_minus = np.mean(D_minus) if len(D_minus) > 0 else 0.0
        mu_plus = np.mean(D_plus) if len(D_plus) > 0 else 0.0
        delta_mu = mu_plus - mu_minus

        # --- S (persistence) ---
        if len(D_plus) > 0:
            S = float(np.mean(np.abs(D_plus - mu_plus) <= epsilon))
        else:
            S = 0.0

        # --- phi_local (local AR(1) from post-change residuals) ---
        # r_t = y_t - mu_hat_t  for t in D_plus window
        tau_plus_start = tau + 1
        tau_plus_end = hi_plus
        r = y[tau_plus_start:tau_plus_end] - means[tau_plus_start:tau_plus_end]
        if len(r) >= 2:
            num = np.sum(r[:-1] * r[1:])
            den = np.sum(r[:-1] ** 2)
            phi_local = float(num / den) if abs(den) > 1e-10 else 0.0
            phi_local = float(np.clip(phi_local, -0.999, 0.999))
        else:
            phi_local = 0.0

        # --- V (variance ratio) ---
        var_minus = float(np.var(D_minus)) if len(D_minus) > 1 else 0.0
        var_plus = float(np.var(D_plus)) if len(D_plus) > 1 else 0.0
        V = var_minus / (var_plus + 1e-10)

        X[i] = [delta_mu, S, phi_local, V]

    logger.info(
        "Feature extraction — %d changepoints, shape=%s, "
        "delta_mu in [%.3g, %.3g], S mean=%.3f, epsilon=%.4g",
        m,
        X.shape,
        X[:, 0].min() if m > 0 else 0,
        X[:, 0].max() if m > 0 else 0,
        X[:, 1].mean() if m > 0 else 0,
        epsilon,
    )
    return X, FEATURE_NAMES
