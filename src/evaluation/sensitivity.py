"""Cost-ratio sensitivity analysis for the MRL risk index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.mrl_index import compute_censored_risk
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def cost_ratio_sensitivity(
    detectors_results: dict[str, dict],
    cF_grid: list[float],
    cD_grid: list[float],
    epsilon: float,
    Tmax: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute detector rankings over a cost-ratio grid.

    For every ``(cF, cD)`` combination, computes the censored risk R-tilde
    for each detector and assigns integer ranks (1 = best = lowest R-tilde,
    ties share the same rank).

    Parameters
    ----------
    detectors_results:
        ``{detector_name: {'FP': int, 'MRL': float}}``
    cF_grid:
        List of false-positive cost values.
    cD_grid:
        List of detection-delay cost values.
    epsilon:
        Lower censoring bound for MRL.
    Tmax:
        Upper censoring bound for MRL.

    Returns
    -------
    rankings_df : pd.DataFrame
        Multi-index ``(cF, cD)``, columns = detector names, values = int ranks.
    raw_df : pd.DataFrame
        Same index/columns structure, values = raw R-tilde floats.
    """
    detector_names = list(detectors_results.keys())
    index_tuples = [(cF, cD) for cF in cF_grid for cD in cD_grid]
    midx = pd.MultiIndex.from_tuples(index_tuples, names=["cF", "cD"])

    raw_records: list[dict] = []
    rank_records: list[dict] = []

    for cF, cD in index_tuples:
        r_tilde_row = {}
        for name, res in detectors_results.items():
            r_tilde_row[name] = compute_censored_risk(
                FP=res["FP"],
                MRL=res["MRL"],
                cF=cF,
                cD=cD,
                epsilon=epsilon,
                Tmax=Tmax,
            )
        raw_records.append(r_tilde_row)

        # Rank: 1 = lowest R-tilde (best); use dense ranking
        values = np.array([r_tilde_row[n] for n in detector_names], dtype=float)
        # Replace inf with a large finite number for ranking purposes
        finite_max = np.nanmax(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 1.0
        rank_values = np.where(np.isfinite(values), values, finite_max * 1e6)
        # scipy.stats.rankdata would give average ties; use simple argsort-based dense rank
        order = np.argsort(rank_values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        # Collapse ties: give tied values the same (minimum) rank
        for i in range(len(detector_names)):
            tied = np.where(rank_values == rank_values[i])[0]
            ranks[i] = int(ranks[tied].min())
        rank_records.append(dict(zip(detector_names, ranks.tolist())))

    raw_df = pd.DataFrame(raw_records, index=midx, columns=detector_names)
    rankings_df = pd.DataFrame(rank_records, index=midx, columns=detector_names)

    logger.info(
        "Sensitivity analysis — %d detectors, %d (cF, cD) combinations",
        len(detector_names),
        len(index_tuples),
    )
    return rankings_df, raw_df
