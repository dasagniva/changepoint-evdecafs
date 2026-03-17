"""Phase I baseline changepoint detectors for multi-detector comparison.

Wraps ruptures algorithms (PELT, BinSeg, BottomUp, Window) and a vanilla
DeCAFS ablation (constant penalty, no EVT modulation).

Each ruptures detector computes its own native L2-BIC penalty
    pen_L2 = 2 * Var(y) * log(n)
so that comparisons are fair on each cost scale.  Vanilla DeCAFS still
receives the same normalised penalty as EV-DeCAFS.
"""

from __future__ import annotations

import numpy as np

from src.phase1.decafs import ev_decafs
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _native_l2_bic(y: np.ndarray) -> float:
    """Native BIC penalty for the L2 (squared-error) cost used by ruptures.

    The L2 cost sums squared residuals from the segment mean, so the
    per-observation cost scale is Var(y).  The BIC penalty is therefore::

        pen = 2 * Var(y) * log(n)

    This gives approximately the same expected number of detections as
    the DeCAFS BIC on the normalised cost scale.
    """
    return 2.0 * float(np.var(y)) * float(np.log(len(y)))


def _ruptures_detect(algo_cls, y: np.ndarray, pen: float, **kwargs) -> np.ndarray:
    """Run a ruptures algorithm and return detected CP indices (0-based).

    Ruptures returns breakpoints as 1-based exclusive end indices. We convert
    to 0-based start indices of new segments (subtract 1, drop the final ``n``
    entry that ruptures always appends).
    """
    try:
        import ruptures as rpt
    except ImportError:
        logger.error("ruptures not installed — run: pip install ruptures")
        return np.array([], dtype=int)

    try:
        signal = y.reshape(-1, 1)
        algo = algo_cls(model="l2", **kwargs).fit(signal)
        bkps = algo.predict(pen=pen)
        cps = np.array([b - 1 for b in bkps[:-1]], dtype=int)
        logger.info("%s: %d changepoints (pen=%.2e)", algo_cls.__name__, len(cps), pen)
        return cps
    except Exception as exc:
        logger.warning("%s failed: %s", algo_cls.__name__, exc)
        return np.array([], dtype=int)


def run_pelt(y: np.ndarray, pen: float | None = None) -> np.ndarray:
    """PELT with L2 cost (ruptures).  Uses native L2-BIC when pen=None."""
    try:
        import ruptures as rpt
        if pen is None:
            pen = _native_l2_bic(y)
        return _ruptures_detect(rpt.Pelt, y, pen)
    except ImportError:
        return np.array([], dtype=int)


def run_binseg(y: np.ndarray, pen: float | None = None) -> np.ndarray:
    """Binary Segmentation with L2 cost (ruptures).  Uses native L2-BIC when pen=None."""
    try:
        import ruptures as rpt
        if pen is None:
            pen = _native_l2_bic(y)
        return _ruptures_detect(rpt.Binseg, y, pen)
    except ImportError:
        return np.array([], dtype=int)


def run_bottomup(y: np.ndarray, pen: float | None = None) -> np.ndarray:
    """Bottom-Up segmentation with L2 cost (ruptures).  Uses native L2-BIC when pen=None."""
    try:
        import ruptures as rpt
        if pen is None:
            pen = _native_l2_bic(y)
        return _ruptures_detect(rpt.BottomUp, y, pen)
    except ImportError:
        return np.array([], dtype=int)


def run_window(y: np.ndarray, pen: float | None = None, width: int = 100) -> np.ndarray:
    """Window-based segmentation with L2 cost (ruptures).  Uses native L2-BIC when pen=None."""
    try:
        import ruptures as rpt
        if pen is None:
            pen = _native_l2_bic(y)
        return _ruptures_detect(rpt.Window, y, pen, width=width)
    except ImportError:
        return np.array([], dtype=int)


def run_vanilla_decafs(
    y: np.ndarray,
    alpha_0: float,
    lambda_param: float,
    gamma: float,
    phi: float,
) -> np.ndarray:
    """Vanilla DeCAFS ablation: constant penalty (no EVT modulation).

    Identical to EV-DeCAFS but with a flat penalty schedule
    ``alpha_t = alpha_0`` for all t.  Isolates the contribution of the
    adaptive EVT penalty.
    """
    try:
        alpha_flat = np.full(len(y), alpha_0)
        result = ev_decafs(y, alpha_flat, lambda_param, gamma, phi)
        cps = np.asarray(result["changepoints"], dtype=int)
        logger.info("Vanilla DeCAFS (flat): %d changepoints (alpha_0=%.2f)", len(cps), alpha_0)
        return cps
    except Exception as exc:
        logger.warning("Vanilla DeCAFS failed: %s", exc)
        return np.array([], dtype=int)


def run_all_baseline_detectors(
    y: np.ndarray,
    pen: float | None = None,
    decafs_params: dict | None = None,
) -> dict[str, np.ndarray]:
    """Run all Phase I baseline detectors and vanilla DeCAFS.

    Ruptures detectors compute their own native L2-BIC penalty
    (``pen`` is ignored for them; pass ``pen`` only for vanilla DeCAFS
    if you want to override the DeCAFS-normalised penalty).

    Parameters
    ----------
    y:
        Time series to segment.
    pen:
        Unused for ruptures (each uses native L2-BIC).  Passed as
        ``alpha_0`` to vanilla DeCAFS when ``decafs_params`` is provided.
    decafs_params:
        Dict with keys ``alpha_0``, ``lambda_param``, ``gamma``, ``phi``
        for the vanilla DeCAFS ablation.

    Returns
    -------
    dict mapping detector name → array of detected CP indices.
    """
    results: dict[str, np.ndarray] = {}

    # Ruptures detectors — each uses its own native L2-BIC penalty
    results["PELT"] = run_pelt(y)
    results["BinSeg"] = run_binseg(y)
    results["BottomUp"] = run_bottomup(y)
    results["Window (w=100)"] = run_window(y, width=100)

    # Vanilla DeCAFS — uses the same normalised penalty as EV-DeCAFS
    if decafs_params is not None:
        results["Vanilla DeCAFS"] = run_vanilla_decafs(
            y,
            alpha_0=decafs_params["alpha_0"],
            lambda_param=decafs_params["lambda_param"],
            gamma=decafs_params["gamma"],
            phi=decafs_params["phi"],
        )

    return results
