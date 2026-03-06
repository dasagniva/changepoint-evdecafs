"""SMOTE-based training data balancing for Phase II.

Wraps imbalanced-learn's SMOTE with logging and an edge-case guard for
very small minority classes.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance the training set.

    If the minority class has fewer than ``k_neighbors`` samples,
    ``k_neighbors`` is reduced to ``n_minority - 1`` to avoid an error.

    If only one class is present, the original data is returned unchanged
    with a warning.

    Parameters
    ----------
    X_train:
        Feature matrix, shape ``(m, d)``.
    y_train:
        Binary labels, shape ``(m,)``.
    k_neighbors:
        Number of nearest neighbours used by SMOTE.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    classes, counts = np.unique(y_train, return_counts=True)
    dist_before = dict(zip(classes.tolist(), counts.tolist()))
    logger.info("Class distribution before SMOTE: %s", dist_before)

    if len(classes) < 2:
        logger.warning(
            "Only one class present (%s); skipping SMOTE.", classes.tolist()
        )
        return X_train.copy(), y_train.copy()

    n_minority = int(counts.min())
    effective_k = min(k_neighbors, n_minority - 1)
    if effective_k < 1:
        logger.warning(
            "Minority class has only %d sample(s); skipping SMOTE.", n_minority
        )
        return X_train.copy(), y_train.copy()

    if effective_k < k_neighbors:
        logger.warning(
            "Reduced k_neighbors from %d to %d (minority class size = %d).",
            k_neighbors,
            effective_k,
            n_minority,
        )

    smote = SMOTE(k_neighbors=effective_k, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    classes_after, counts_after = np.unique(y_res, return_counts=True)
    dist_after = dict(zip(classes_after.tolist(), counts_after.tolist()))
    logger.info("Class distribution after SMOTE:  %s", dist_after)

    return X_res, y_res
