"""AR(1) parameter estimation for EV-DeCAFS Phase I.

Estimates phi, sigma_v^2, and sigma_eta^2 from a univariate time series.
"""

from __future__ import annotations

import numpy as np
from statsmodels.regression.linear_model import yule_walker

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def estimate_ar1_params(y: np.ndarray) -> dict[str, float]:
    """Estimate AR(1) model parameters from a univariate time series.

    Fits the model::

        y_t = mu_t + epsilon_t,   epsilon_t = phi * epsilon_{t-1} + v_t

    where ``v_t ~ N(0, sigma_v^2)`` and ``mu_t`` is a piecewise-constant
    level that changes with variance ``sigma_eta^2``.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.

    Returns
    -------
    dict with keys:
        - ``'phi'`` : float  — AR(1) autocorrelation coefficient, clipped to (-1, 1).
        - ``'sigma_v_sq'`` : float — Innovation variance.
        - ``'sigma_eta_sq'`` : float — Level-change variance (>= 1e-8).

    Notes
    -----
    ``phi`` is estimated via Yule-Walker equations (statsmodels) on the
    mean-centred series.  ``sigma_v_sq`` is the variance of the AR(1)
    residuals ``y_t - phi * y_{t-1}``.  ``sigma_eta_sq`` is estimated from
    the variance of the first-differenced signal minus the theoretical
    contribution of the AR(1) noise; clipped to ``1e-8`` if negative.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        raise ValueError("Need at least 3 observations to estimate AR(1) params.")

    # --- Step 1: phi via Yule-Walker on the mean-centred series ---
    y_centred = y - np.mean(y)
    rho, _ = yule_walker(y_centred, order=1, method="mle")
    phi = float(np.clip(rho[0], -0.999, 0.999))

    # --- Step 2: sigma_v^2 from AR(1) residuals ---
    # Residuals: e_t = y_t - phi * y_{t-1}  (absorbs the unknown level into the mean)
    residuals = y[1:] - phi * y[:-1]
    sigma_v_sq = float(np.var(residuals, ddof=1))

    # --- Step 3: sigma_eta^2 ---
    # Under the AR(1) noise model, the variance of first differences at
    # non-changepoint times is:
    #   Var(dy_t) = Var(epsilon_t - epsilon_{t-1}) = 2 * sigma_v^2 / (1 + phi)
    # Any excess variance is attributed to the piecewise-constant level process.
    dy = np.diff(y)
    var_dy = float(np.var(dy, ddof=1))
    noise_contribution = 2.0 * sigma_v_sq / (1.0 + abs(phi) + 1e-12)
    sigma_eta_sq = max(var_dy - noise_contribution, 1e-8)

    logger.info(
        "AR(1) estimates — phi=%.4f, sigma_v^2=%.4e, sigma_eta^2=%.4e",
        phi,
        sigma_v_sq,
        sigma_eta_sq,
    )
    return {"phi": phi, "sigma_v_sq": sigma_v_sq, "sigma_eta_sq": sigma_eta_sq}
