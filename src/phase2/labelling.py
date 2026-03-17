"""Changepoint labelling (Algorithm 3).

Labels each detected changepoint as 'sustained' (1) or 'recoiled' (0)
based on magnitude and persistence thresholds.

Also provides:
  - relabel_with_hypersensitive(): 4-class relabelling using X-algorithm flags
  - self_supervised_oilwell_labels(): pseudo-labels for oilwell (no ground truth)
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


# ---------------------------------------------------------------------------
# X-Algorithm relabelling (Change 3)
# ---------------------------------------------------------------------------

def relabel_with_hypersensitive(
    cp_indices: np.ndarray,
    x_flags: np.ndarray,
    true_cp_indices: np.ndarray | None,
    existing_labels: np.ndarray,
    tolerance: int = 10,
) -> np.ndarray:
    """Relabel detected CPs using X-algorithm (BOCPD/CUSUM) flags.

    Applies a 4-class rule per detected changepoint at index i:

    +------+----------+--------------------+
    | X hit| True hit | Label              |
    +======+==========+====================+
    | True | True     | Sustained          |
    | True | False    | Abrupt             |
    | False| True     | Abrupt-Preceded    |
    | False| False    | keep existing      |
    +------+----------+--------------------+

    For datasets with no ground-truth (``true_cp_indices`` is None or empty)
    only the first two rules apply (x_hit drives the label; true_hit is
    always False).  A warning is logged in this case.

    Parameters
    ----------
    cp_indices:
        Array of detected changepoint indices in the **same coordinate system**
        as ``x_flags`` and ``true_cp_indices``.
    x_flags:
        Boolean array of length n (series length) — True where the X detector
        (BOCPD or CUSUM) flagged a CP.
    true_cp_indices:
        Ground-truth CP indices.  Pass ``None`` or an empty array to operate
        in no-label mode.
    existing_labels:
        Current labels (int 0/1 or string) — used as fallback when neither
        x_hit nor true_hit is True.
    tolerance:
        Window half-width (obs) within which a flag/true CP is considered
        'near' a detected CP.  Default 10.

    Returns
    -------
    new_labels : np.ndarray of str, shape (m,)
        Updated string labels: "Sustained", "Recoiled", "Abrupt", or
        "Abrupt-Preceded".
    """
    cp_indices = np.asarray(cp_indices, dtype=int)
    x_flags = np.asarray(x_flags, dtype=bool)
    m = len(cp_indices)

    no_label_mode = (true_cp_indices is None or len(true_cp_indices) == 0)
    if no_label_mode:
        logger.warning(
            "relabel_with_hypersensitive: no ground-truth CPs provided — "
            "operating in no-label mode (x_hit drives labels; true_hit always False)."
        )
        true_cps = np.array([], dtype=int)
    else:
        true_cps = np.asarray(true_cp_indices, dtype=int)

    # Pre-compute x_flags index set: indices where any flag is True within tolerance
    # For efficiency, gather all flagged positions
    flag_positions = np.where(x_flags)[0]

    def _x_hit(cp_idx: int) -> bool:
        if len(flag_positions) == 0:
            return False
        return bool(np.any(np.abs(flag_positions - cp_idx) <= tolerance))

    def _true_hit(cp_idx: int) -> bool:
        if len(true_cps) == 0:
            return False
        return bool(np.any(np.abs(true_cps - cp_idx) <= tolerance))

    # Convert existing labels to canonical strings
    def _existing_str(lbl) -> str:
        if isinstance(lbl, str):
            return lbl
        return "Sustained" if int(lbl) == 1 else "Recoiled"

    new_labels = np.empty(m, dtype=object)
    counts = {"Sustained": 0, "Abrupt": 0, "Abrupt-Preceded": 0, "kept": 0}

    for i, cp in enumerate(cp_indices):
        xh = _x_hit(int(cp))
        th = _true_hit(int(cp))

        if xh and th:
            new_labels[i] = "Sustained"
            counts["Sustained"] += 1
        elif xh and not th:
            new_labels[i] = "Abrupt"
            counts["Abrupt"] += 1
        elif (not xh) and th:
            new_labels[i] = "Abrupt-Preceded"
            counts["Abrupt-Preceded"] += 1
        else:
            new_labels[i] = _existing_str(existing_labels[i])
            counts["kept"] += 1

    logger.info(
        "Relabelling — Sustained=%d, Abrupt=%d, Abrupt-Preceded=%d, kept=%d",
        counts["Sustained"], counts["Abrupt"],
        counts["Abrupt-Preceded"], counts["kept"],
    )
    return new_labels


# ---------------------------------------------------------------------------
# Self-supervised oilwell labelling (Change 4)
# ---------------------------------------------------------------------------

def self_supervised_oilwell_labels(
    y: np.ndarray,
    cp_indices: np.ndarray,
    ar1_params: dict,
    config: dict,
) -> np.ndarray:
    """Generate pseudo-labels for oilwell data using three heuristics.

    Applies overrides in order of priority (later overrides take precedence):

    4a. Dual-sensitivity DeCAFS consensus:
        - CP at BOTH conservative + aggressive C → "Sustained"
        - CP ONLY at aggressive C → "Recoiled"

    4b. EVI field override:
        If xi_local > 0.1 at CP i → "Recoiled" (Fréchet-class tail)

    4c. Segment duration override:
        If segment ending at CP i has duration < min_segment_length → "Recoiled"

    Parameters
    ----------
    y:
        Training series.
    cp_indices:
        Detected changepoint indices (from conservative C run).
    ar1_params:
        Dict with keys 'phi', 'sigma_v_sq', 'sigma_eta_sq', and optionally
        'xi_local' (array of local EVI values per CP).
    config:
        Dict with keys:
          - 'conservative_C': float — the BIC-selected penalty multiplier
          - 'min_segment_length': int — min segment duration (default 50)
          - 'window_halfwidth_w': int — for EVI field (default 50)
          - 'gpd_percentile_q0': float — for EVI field (default 0.90)
          - 'lambda_param': float — 1/sigma_eta_sq
          - 'gamma': float — 1/sigma_v_sq

    Returns
    -------
    labels : np.ndarray of str, shape (m,)
        Pseudo-labels: "Sustained" or "Recoiled".
    """
    from src.phase1.decafs import ev_decafs
    from src.phase1.ar1_model import compute_bic_penalty
    from src.phase1.evt_penalty import compute_evi_field

    y = np.asarray(y, dtype=float)
    cp_indices = np.asarray(cp_indices, dtype=int)
    m = len(cp_indices)
    n = len(y)

    conservative_C = float(config.get("conservative_C", 8.0))
    min_seg_len = int(config.get("min_segment_length", 50))
    w = int(config.get("window_halfwidth_w", 50))
    q0 = float(config.get("gpd_percentile_q0", 0.90))
    lambda_param = float(config.get("lambda_param", 1.0 / (ar1_params["sigma_eta_sq"] + 1e-12)))
    gamma = float(config.get("gamma", 1.0 / (ar1_params["sigma_v_sq"] + 1e-12)))
    phi = float(ar1_params["phi"])

    # --- 4a: Dual-sensitivity consensus ---
    alpha_0_cons = compute_bic_penalty(n, conservative_C)
    alpha_0_aggr = compute_bic_penalty(n, 1.0)  # aggressive C=1.0

    res_cons = ev_decafs(y, np.full(n, alpha_0_cons), lambda_param, gamma, phi)
    res_aggr = ev_decafs(y, np.full(n, alpha_0_aggr), lambda_param, gamma, phi)

    cp_set_cons = set(res_cons["changepoints"].tolist())
    cp_set_aggr = set(res_aggr["changepoints"].tolist())

    labels = np.empty(m, dtype=object)
    for i, cp in enumerate(cp_indices):
        # Check presence in each sensitivity run (with ±5 obs tolerance)
        in_cons = any(abs(cp - c) <= 5 for c in cp_set_cons)
        in_aggr = any(abs(cp - c) <= 5 for c in cp_set_aggr)
        if in_cons and in_aggr:
            labels[i] = "Sustained"
        else:
            labels[i] = "Recoiled"

    # --- 4b: EVI field override ---
    xi_local = ar1_params.get("xi_local", None)
    if xi_local is None:
        # Compute EVI field and extract local values
        xi_field = compute_evi_field(y, w=w, q0=q0)
        xi_local = np.array([float(xi_field[min(int(cp), n - 1)]) for cp in cp_indices])

    for i, (cp, xi_v) in enumerate(zip(cp_indices, xi_local)):
        if float(xi_v) > 0.1:
            labels[i] = "Recoiled"

    # --- 4c: Segment duration override ---
    boundaries = [0] + list(cp_indices) + [n]
    for i, cp in enumerate(cp_indices):
        seg_len = int(cp) - int(boundaries[i])
        if seg_len < min_seg_len:
            labels[i] = "Recoiled"

    n_sustained = int(np.sum(labels == "Sustained"))
    n_recoiled = int(np.sum(labels == "Recoiled"))
    logger.info(
        "Self-supervised oilwell labels — %d Sustained, %d Recoiled "
        "(cons_C=%.1f, min_seg=%d)",
        n_sustained, n_recoiled, conservative_C, min_seg_len,
    )
    return labels
