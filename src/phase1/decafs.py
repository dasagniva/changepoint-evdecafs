"""EV-DeCAFS dynamic programming recursion (Algorithm 1).

Implements the penalised cost changepoint detection algorithm with an
AR(1) observation model and a time-varying adaptive penalty.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def ev_decafs(
    y: np.ndarray,
    alpha_t: np.ndarray,
    lambda_param: float,
    gamma: float,
    phi: float,
    n_grid: int = 1000,
) -> dict:
    """Run the EV-DeCAFS penalised changepoint detection algorithm.

    Implements Algorithm 1 from the paper.  The cost-function recursion is::

        Q_t(mu) = min_u {
            Q_{t-1}(u)
            + min(lambda * (mu - u)^2, alpha_t[t])
            + gamma * ((y[t] - mu) - phi * (y[t-1] - u))^2
        }

    The minimisation over the continuous level ``mu`` is approximated on a
    uniform grid of ``n_grid`` points spanning
    ``[min(y) - 2*std(y), max(y) + 2*std(y)]``.

    Changepoints are recovered by backtracking: a changepoint at ``t`` is
    declared when ``(mu_hat[t] - mu_hat[t-1])^2 > alpha_t[t] / lambda``.

    Parameters
    ----------
    y:
        Univariate time series, shape ``(n,)``.
    alpha_t:
        Time-varying penalty schedule, shape ``(n,)``.
    lambda_param:
        Precision of the level process: ``lambda = 1 / sigma_eta^2``.
    gamma:
        Precision of the observation noise: ``gamma = 1 / sigma_v^2``.
    phi:
        AR(1) autocorrelation coefficient.
    n_grid:
        Number of grid points for the mu discretisation.

    Returns
    -------
    dict with keys:
        - ``'changepoints'``: np.ndarray of int — detected changepoint indices.
        - ``'means'``: np.ndarray of float, shape ``(n,)`` — estimated mu_t.
        - ``'cost'``: float — minimum total cost F_n.
    """
    y = np.asarray(y, dtype=float)
    alpha_t = np.asarray(alpha_t, dtype=float)
    n = len(y)

    if n < 2:
        return {"changepoints": np.array([], dtype=int), "means": y.copy(), "cost": 0.0}

    # --- Build mu grid ---
    std_y = np.std(y)
    if std_y < 1e-10:
        std_y = 1.0
    mu_min = np.min(y) - 2.0 * std_y
    mu_max = np.max(y) + 2.0 * std_y
    mu_grid = np.linspace(mu_min, mu_max, n_grid)

    # Broadcasting shapes: mu is current level (rows), u is previous level (cols)
    mu_col = mu_grid[:, np.newaxis]  # (n_grid, 1)
    u_row = mu_grid[np.newaxis, :]  # (1, n_grid)

    # Pre-compute the time-invariant quadratic level-change penalty
    quadratic_penalty = lambda_param * (mu_col - u_row) ** 2  # (n_grid, n_grid)

    # --- Initialise Q_0(mu) = gamma * (y[0] - mu)^2 ---
    Q = gamma * (y[0] - mu_grid) ** 2  # (n_grid,)

    # --- Forward pass ---
    # backtrack_ptr[t, i] = grid index j that was optimal at step t for current grid i
    # int16 is sufficient because n_grid <= 1000 << 32767
    backtrack_ptr = np.empty((n, n_grid), dtype=np.int16)

    t0 = time.perf_counter()

    for t in tqdm(range(1, n), desc="EV-DeCAFS", unit="step", leave=True):
        # AR(1) observation cost: gamma * ((y[t] - mu_i) - phi*(y[t-1] - u_j))^2
        ar1_cost = gamma * (
            (y[t] - mu_col) - phi * (y[t - 1] - u_row)
        ) ** 2  # (n_grid, n_grid)

        # Adaptive penalty: flat alpha_t wins over quadratic when mu changes a lot
        penalty = np.minimum(quadratic_penalty, alpha_t[t])  # (n_grid, n_grid)

        # Total cost: Q_{t-1}(u_j) + penalty(mu_i, u_j) + ar1_cost(mu_i, u_j)
        total = Q[np.newaxis, :] + penalty + ar1_cost  # (n_grid, n_grid)

        # For each current grid point mu_i, find the best previous grid point u_j
        best_j = np.argmin(total, axis=1)  # (n_grid,)
        Q = total[np.arange(n_grid), best_j]  # (n_grid,)
        backtrack_ptr[t] = best_j.astype(np.int16)

    elapsed = time.perf_counter() - t0

    # --- Backtrack to recover optimal mu sequence ---
    best_final = int(np.argmin(Q))
    min_cost = float(Q[best_final])

    mu_hat_idx = np.empty(n, dtype=int)
    mu_hat_idx[n - 1] = best_final
    for t in range(n - 2, -1, -1):
        mu_hat_idx[t] = int(backtrack_ptr[t + 1, mu_hat_idx[t + 1]])

    mu_hat = mu_grid[mu_hat_idx]

    # --- Detect changepoints ---
    # A changepoint at t is declared when the level jump exceeds the threshold
    # implied by the adaptive penalty: (delta_mu)^2 > alpha_t[t] / lambda_param
    changepoints = []
    for t in range(1, n):
        jump_sq = (mu_hat[t] - mu_hat[t - 1]) ** 2
        threshold = alpha_t[t] / (lambda_param + 1e-12)
        if jump_sq > threshold:
            changepoints.append(t)

    changepoints = np.array(changepoints, dtype=int)

    logger.info(
        "EV-DeCAFS — %d changepoints detected, cost=%.4f, elapsed=%.1fs",
        len(changepoints),
        min_cost,
        elapsed,
    )
    return {"changepoints": changepoints, "means": mu_hat, "cost": min_cost}
