"""BOCPD-based labelling oracle for Phase II training.

Uses Bayesian Online Change Point Detection (Adams & MacKay 2007) as a
deliberately hypersensitive detector to generate training labels for the
FPNN. BOCPD flags every possible mean deviation; cross-referencing its
output with ground truth (where available) produces sustained/recoiled
labels for DeCAFS-detected changepoints.

At TEST TIME, BOCPD is NOT used — only DeCAFS + FPNN.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def run_bocpd(
    y: np.ndarray,
    hazard_rate: float = 1 / 200,
    mu_prior: float = 0.0,
    kappa_prior: float = 1.0,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run Bayesian Online Change Point Detection.

    Implements Adams & MacKay (2007) with a constant hazard rate and
    Normal-Inverse-Gamma conjugate prior.

    Args:
        y: time series (n,)
        hazard_rate: prior probability of changepoint at each step
                     (lower = more sensitive, default 1/200)
        mu_prior: prior mean
        kappa_prior: prior precision scaling
        alpha_prior: prior shape for inverse-gamma
        beta_prior: prior rate for inverse-gamma
        threshold: posterior probability threshold for declaring a CP
                   (lower = more sensitive, default 0.5)

    Returns:
        changepoints: array of detected CP indices
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Run-length posterior: R[t, r] = P(run_length = r | y_{1:t})
    # For memory efficiency, keep only current and previous step
    max_run = min(n, 500)  # cap run length for memory

    # Sufficient statistics for each run length
    # Normal-Inverse-Gamma: track mu, kappa, alpha, beta
    mu = np.full(max_run + 1, mu_prior)
    kappa = np.full(max_run + 1, kappa_prior)
    alpha = np.full(max_run + 1, alpha_prior)
    beta = np.full(max_run + 1, beta_prior)

    # Log run-length probabilities
    log_R = np.full(max_run + 1, -np.inf)
    log_R[0] = 0.0  # start with run length 0

    changepoints = []
    log_H = np.log(hazard_rate)
    log_1mH = np.log(1.0 - hazard_rate)

    for t in range(n):
        x = y[t]

        # Predictive probability under each run length
        # Student-t predictive: p(x | run_length = r)
        df = 2.0 * alpha[:max_run + 1]
        pred_var = beta[:max_run + 1] * (kappa[:max_run + 1] + 1.0) / (
            kappa[:max_run + 1] * alpha[:max_run + 1]
        )
        pred_var = np.maximum(pred_var, 1e-10)

        # Log Student-t PDF
        z = (x - mu[:max_run + 1]) ** 2 / pred_var
        log_pred = (
            gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(np.pi * df * pred_var)
            - ((df + 1) / 2) * np.log(1 + z / df)
        )

        # Growth probabilities (existing run continues)
        log_growth = log_R[:max_run + 1] + log_pred + log_1mH

        # Changepoint probability (run length resets to 0)
        log_cp = np.logaddexp.reduce(log_R[:max_run + 1] + log_pred + log_H)

        # Update run-length distribution
        new_log_R = np.full(max_run + 1, -np.inf)
        new_log_R[0] = log_cp
        new_log_R[1:max_run + 1] = log_growth[:max_run]

        # Normalize
        log_evidence = np.logaddexp.reduce(new_log_R[:max_run + 1])
        new_log_R[:max_run + 1] -= log_evidence

        # Check if changepoint probability exceeds threshold
        cp_prob = np.exp(new_log_R[0])
        if cp_prob > threshold and t > 0:
            changepoints.append(t)

        # Update sufficient statistics
        new_mu = np.full(max_run + 1, mu_prior)
        new_kappa = np.full(max_run + 1, kappa_prior)
        new_alpha = np.full(max_run + 1, alpha_prior)
        new_beta = np.full(max_run + 1, beta_prior)

        # For run lengths > 0, update from previous
        if max_run > 0:
            old_kappa = kappa[:max_run]
            new_kappa[1:max_run + 1] = old_kappa + 1
            new_mu[1:max_run + 1] = (old_kappa * mu[:max_run] + x) / new_kappa[1:max_run + 1]
            new_alpha[1:max_run + 1] = alpha[:max_run] + 0.5
            new_beta[1:max_run + 1] = (
                beta[:max_run]
                + 0.5 * old_kappa * (x - mu[:max_run]) ** 2 / new_kappa[1:max_run + 1]
            )

        log_R = new_log_R
        mu = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta = new_beta

    changepoints = np.array(changepoints, dtype=int)
    logger.info(
        "BOCPD: %d changepoints detected (hazard=%.4f, threshold=%.2f)",
        len(changepoints), hazard_rate, threshold,
    )
    return changepoints


def label_with_bocpd(
    decafs_cps: np.ndarray,
    bocpd_cps: np.ndarray,
    true_cps: np.ndarray,
    tolerance: int,
    has_ground_truth: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Generate sustained/recoiled labels for DeCAFS-detected changepoints.

    Cross-references DeCAFS detections with BOCPD output and ground truth.

    For each DeCAFS CP tau:
      - If tau is near a TRUE CP (within tolerance) → Sustained (1)
      - If tau is near a BOCPD CP but NOT near a true CP → Recoiled (0)
      - If tau is not near any BOCPD or true CP → Recoiled (0)

    When no ground truth is available (has_ground_truth=False):
      - If tau is near a BOCPD CP → placeholder -1 (caller refines)
      - Otherwise → Recoiled (0)

    Args:
        decafs_cps: array of DeCAFS-detected CP indices
        bocpd_cps: array of BOCPD-detected CP indices (hypersensitive)
        true_cps: array of ground-truth CP indices (empty if no ground truth)
        tolerance: matching window (e.g., 2% of series length)
        has_ground_truth: whether true_cps are expert-annotated

    Returns:
        labels: array of {0, 1} (or -1 pending) same length as decafs_cps
        label_reasons: list of strings explaining each label assignment
    """
    decafs_cps = np.asarray(decafs_cps, dtype=int)
    bocpd_cps = np.asarray(bocpd_cps, dtype=int)
    true_cps = np.asarray(true_cps, dtype=int)

    labels = np.zeros(len(decafs_cps), dtype=int)
    reasons = []

    for i, tau in enumerate(decafs_cps):
        # Check proximity to true CPs
        near_true = False
        if has_ground_truth and len(true_cps) > 0:
            if np.min(np.abs(true_cps - tau)) <= tolerance:
                near_true = True

        # Check proximity to BOCPD CPs
        near_bocpd = False
        if len(bocpd_cps) > 0:
            if np.min(np.abs(bocpd_cps - tau)) <= tolerance:
                near_bocpd = True

        if has_ground_truth:
            if near_true:
                labels[i] = 1  # Sustained — confirmed by ground truth
                reasons.append("sustained: near true CP")
            elif near_bocpd:
                labels[i] = 0  # Recoiled — BOCPD saw something but not a true CP
                reasons.append("recoiled: BOCPD-only detection")
            else:
                labels[i] = 0  # Recoiled — neither BOCPD nor ground truth
                reasons.append("recoiled: unconfirmed by BOCPD or ground truth")
        else:
            # No ground truth — use BOCPD confirmation + heuristic
            if near_bocpd:
                labels[i] = -1  # placeholder — caller decides based on features
                reasons.append("bocpd-confirmed: pending feature check")
            else:
                labels[i] = 0
                reasons.append("recoiled: unconfirmed by BOCPD")

    n_sustained = int(np.sum(labels == 1))
    n_recoiled = int(np.sum(labels == 0))
    n_pending = int(np.sum(labels == -1))
    logger.info(
        "BOCPD labelling: %d sustained, %d recoiled, %d pending "
        "(from %d DeCAFS CPs)",
        n_sustained, n_recoiled, n_pending, len(decafs_cps),
    )

    return labels, reasons


def refine_pending_labels(
    labels: np.ndarray,
    features: np.ndarray,
    kappa_mu: float,
    kappa_S: float,
) -> np.ndarray:
    """For labels marked as -1 (BOCPD-confirmed, no ground truth),
    apply the heuristic rule from Algorithm 3 to decide sustained/recoiled.

    Args:
        labels: array with some entries == -1
        features: feature matrix (m, 5) with columns [delta_mu, S, phi, V, xi]
        kappa_mu: magnitude threshold
        kappa_S: persistence threshold

    Returns:
        refined_labels: array with all entries in {0, 1}
    """
    refined = labels.copy()
    for i in range(len(refined)):
        if refined[i] == -1:
            delta_mu = abs(float(features[i, 0]))
            S = float(features[i, 1])
            if delta_mu > kappa_mu and S > kappa_S:
                refined[i] = 1  # Sustained
            else:
                refined[i] = 0  # Recoiled
    return refined
