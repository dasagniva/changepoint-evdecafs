"""GPD goodness-of-fit diagnostics for EV-DeCAFS tail analysis.

Samples local windows from a time series, fits GPD to threshold exceedances,
runs KS tests, and classifies tails as Weibull / Gumbel / Frechet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Tail-type boundary on the shape parameter xi
_WEIBULL_THRESHOLD = -0.05   # xi < -0.05 → bounded tail
_FRECHET_THRESHOLD = 0.05    # xi >  0.05 → heavy tail


def _fit_gpd_window(
    y_window: np.ndarray,
    q0: float,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Fit GPD to threshold exceedances in a single window.

    Returns a dict with ``xi``, ``beta``, ``ks_stat``, ``ks_pvalue``,
    ``n_exceedances``, ``xi_ci_lower``, ``xi_ci_upper``, and
    ``tail_type``, or ``None`` if fewer than 5 exceedances are found.
    """
    if rng is None:
        rng = np.random.default_rng()

    threshold = float(np.percentile(np.abs(y_window - np.mean(y_window)), q0 * 100))
    exceedances = np.abs(y_window - np.mean(y_window))
    exceedances = exceedances[exceedances > threshold] - threshold

    if len(exceedances) < 5:
        return None

    # Fit GPD: genpareto with floc=0
    try:
        xi, loc, beta = stats.genpareto.fit(exceedances, floc=0)
    except Exception:
        return None

    if beta <= 0:
        return None

    # KS test
    ks_stat, ks_pvalue = stats.kstest(
        exceedances, stats.genpareto(c=xi, loc=0, scale=beta).cdf
    )

    # Bootstrap 95% CI for xi
    xi_boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(exceedances, size=len(exceedances), replace=True)
        try:
            xi_b, _, _ = stats.genpareto.fit(sample, floc=0)
            xi_boot.append(xi_b)
        except Exception:
            continue
    if len(xi_boot) >= 10:
        xi_ci_lower = float(np.percentile(xi_boot, 2.5))
        xi_ci_upper = float(np.percentile(xi_boot, 97.5))
    else:
        xi_ci_lower = xi_ci_upper = float(xi)

    # Classify tail type
    if xi < _WEIBULL_THRESHOLD:
        tail_type = "Weibull"
    elif xi > _FRECHET_THRESHOLD:
        tail_type = "Frechet"
    else:
        tail_type = "Gumbel"

    return {
        "xi": float(xi),
        "beta": float(beta),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "n_exceedances": int(len(exceedances)),
        "xi_ci_lower": xi_ci_lower,
        "xi_ci_upper": xi_ci_upper,
        "tail_type": tail_type,
    }


def run_tail_diagnostics(
    y: np.ndarray,
    w: int = 50,
    q0: float = 0.90,
    n_windows: int = 100,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Run GPD goodness-of-fit diagnostics on randomly sampled windows.

    Parameters
    ----------
    y:
        Univariate time series.
    w:
        Half-width of the local window (same as Phase I parameter).
    q0:
        Quantile used to threshold local deviations before GPD fit.
    n_windows:
        Number of randomly sampled window positions.
    n_bootstrap:
        Bootstrap resamples for xi confidence interval.
    seed:
        Random seed.

    Returns
    -------
    summary : dict
        Aggregate diagnostics.
    per_window : pd.DataFrame
        Row per successfully fitted window.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # Sample window centres (avoid edges where window would be truncated)
    min_centre = w
    max_centre = n - w - 1
    if max_centre <= min_centre:
        raise ValueError(f"Series too short (n={n}) for window half-width w={w}.")

    centres = rng.choice(
        np.arange(min_centre, max_centre + 1),
        size=min(n_windows, max_centre - min_centre + 1),
        replace=False,
    )

    rows = []
    for centre in centres:
        window = y[centre - w: centre + w + 1]
        result = _fit_gpd_window(window, q0, n_bootstrap=n_bootstrap, rng=rng)
        if result is not None:
            result["centre"] = int(centre)
            rows.append(result)

    per_window = pd.DataFrame(rows)

    if len(per_window) == 0:
        logger.warning("Tail diagnostics: no windows yielded enough exceedances.")
        summary = {
            "n_windows_tested": 0,
            "n_windows_fitted": 0,
            "mean_xi": float("nan"),
            "std_xi": float("nan"),
            "median_xi": float("nan"),
            "fraction_weibull": float("nan"),
            "fraction_gumbel": float("nan"),
            "fraction_frechet": float("nan"),
            "ks_not_rejected_fraction": float("nan"),
            "ks_median_p_value": float("nan"),
            "overall_classification": "unknown",
        }
        return summary, per_window

    xi_vals = per_window["xi"].values
    tail_types = per_window["tail_type"].values
    ks_pvals = per_window["ks_pvalue"].values

    n_fitted = len(per_window)
    frac_weibull = float((tail_types == "Weibull").mean())
    frac_gumbel = float((tail_types == "Gumbel").mean())
    frac_frechet = float((tail_types == "Frechet").mean())
    ks_not_rejected = float((ks_pvals >= 0.05).mean())

    # Overall classification by majority vote
    counts = {
        "Weibull": frac_weibull,
        "Gumbel": frac_gumbel,
        "Frechet": frac_frechet,
    }
    overall = max(counts, key=counts.get)

    summary = {
        "n_windows_tested": int(len(centres)),
        "n_windows_fitted": n_fitted,
        "mean_xi": float(np.mean(xi_vals)),
        "std_xi": float(np.std(xi_vals)),
        "median_xi": float(np.median(xi_vals)),
        "fraction_weibull": round(frac_weibull, 4),
        "fraction_gumbel": round(frac_gumbel, 4),
        "fraction_frechet": round(frac_frechet, 4),
        "ks_not_rejected_fraction": round(ks_not_rejected, 4),
        "ks_median_p_value": float(np.median(ks_pvals)),
        "overall_classification": overall,
    }

    logger.info(
        "Tail diagnostics (%d/%d windows fitted):\n"
        "  Mean xi = %.3f ± %.3f  (median=%.3f)\n"
        "  Weibull: %.0f%%  Gumbel: %.0f%%  Frechet: %.0f%%\n"
        "  KS: GPD not rejected at 5%% in %.0f%% of windows  (median p=%.3f)\n"
        "  Overall classification: %s",
        n_fitted, len(centres),
        summary["mean_xi"], summary["std_xi"], summary["median_xi"],
        frac_weibull * 100, frac_gumbel * 100, frac_frechet * 100,
        ks_not_rejected * 100, summary["ks_median_p_value"],
        overall,
    )

    return summary, per_window
