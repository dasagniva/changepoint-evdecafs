"""Hausdorff distance between sets of changepoint indices."""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def directed_hausdorff(set_A: np.ndarray, set_B: np.ndarray) -> float:
    """Compute the directed Hausdorff distance from set_A to set_B.

    ``sup_{a in A} inf_{b in B} |a - b|``

    Returns 0.0 if either set is empty (no point in A has a finite
    nearest-neighbour distance to check).

    Parameters
    ----------
    set_A, set_B:
        1-D arrays of changepoint indices (need not be sorted).

    Returns
    -------
    float
    """
    A = np.asarray(set_A, dtype=float).ravel()
    B = np.asarray(set_B, dtype=float).ravel()

    if len(A) == 0 or len(B) == 0:
        return 0.0

    # For each a in A, find min |a - b| over b in B
    # Broadcasting: (|A|, 1) - (1, |B|) → (|A|, |B|)
    dists = np.abs(A[:, np.newaxis] - B[np.newaxis, :])  # (|A|, |B|)
    min_dists = dists.min(axis=1)  # (|A|,) — nearest B for each a
    return float(min_dists.max())


def symmetric_hausdorff(set_A: np.ndarray, set_B: np.ndarray) -> float:
    """Compute the symmetric Hausdorff distance between two sets.

    ``max(directed_hausdorff(A, B), directed_hausdorff(B, A))``

    Parameters
    ----------
    set_A, set_B:
        1-D arrays of changepoint indices.

    Returns
    -------
    float
    """
    d_AB = directed_hausdorff(set_A, set_B)
    d_BA = directed_hausdorff(set_B, set_A)
    h = max(d_AB, d_BA)
    logger.debug("Hausdorff — d(A,B)=%.2f, d(B,A)=%.2f, symmetric=%.2f", d_AB, d_BA, h)
    return h
