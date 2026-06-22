"""Full end-to-end EV-DeCAFS pipeline.

Run with::

    python scripts/run_pipeline.py --dataset both
    python scripts/run_pipeline.py --dataset bitcoin --skip-baselines
    python scripts/run_pipeline.py --dataset welllog --output-dir my_results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow imports from repo root when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_tcpd_for_pipeline, load_welllog_data, load_oilwell_data, load_brent_crude, load_us_ip_growth
from src.evaluation.tail_diagnostics import run_tail_diagnostics
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hausdorff import symmetric_hausdorff
from src.evaluation.monte_carlo import (
    assign_nearest_labels,
    generate_synthetic_series,
    run_monte_carlo,
)
from src.evaluation.mrl_index import (
    compute_censored_risk,
    compute_mrl,
    convert_to_relative,
)
from src.evaluation.sensitivity import cost_ratio_sensitivity
from src.phase1.ar1_model import compute_bic_penalty, estimate_ar1_params
from src.phase1.baseline_detectors import run_all_baseline_detectors
from src.phase1.decafs import ev_decafs
from src.phase1.evt_penalty import (
    compute_adaptive_penalty,
    compute_evi_field,
    compute_exceedance_count_penalty,
)
from src.phase1.feature_extract import extract_features
from src.phase2.baselines import get_baselines, train_and_evaluate_all
from src.phase2.fpnn import FourierPNN
from src.phase2.bocpd_labeller import label_with_bocpd, refine_pending_labels
from src.phase1.hypersensitive_cpd import run_bocpd, run_cusum
from src.phase2.labelling import (
    compute_kappa_mu,
    label_changepoints,
    relabel_with_hypersensitive,
    self_supervised_oilwell_labels,
)
from src.phase2.smote_balance import balance_training_data
from src.utils.logging_config import setup_logger
from src.visualization.run_charts import plot_changepoint_comparison, plot_run_chart
from src.visualization.style import apply_style
from src.visualization.phase2_visualization import plot_phase2_train_test, plot_us_ip_annotated
from src.visualization.mc_comparison import plot_mc_classifier_comparison
from src.visualization.phase1_comparison import plot_phase1_multimetric
from src.visualization.tail_plots import plot_tail_diagnostics

logger = setup_logger("pipeline")


# =============================================================================
# Argument parsing & config
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full EV-DeCAFS two-phase pipeline."
    )
    parser.add_argument("--config", type=str, default="config/params.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["welllog", "oilwell", "brent_crude", "us_ip_growth", "all"],
        default="all",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Phase I helpers
# =============================================================================

def run_phase1(
    y_train: np.ndarray,
    params: dict,
    alpha_0: float | None = None,
) -> dict:
    """Estimate AR(1) params, build adaptive penalty, run EV-DeCAFS.

    Parameters
    ----------
    alpha_0:
        Override base penalty.  If None, uses BIC or fixed value from params.

    Returns a dict with keys: ar1, xi_field, alpha_gpd, alpha_ec,
    decafs_result, elapsed_ar1, elapsed_evi, elapsed_decafs, alpha_0_used.
    """
    p1 = params["phase1"]

    if alpha_0 is None:
        if p1.get("alpha_0_mode", "fixed") == "bic":
            alpha_0 = compute_bic_penalty(len(y_train), p1.get("bic_multiplier", 2.0))
        else:
            alpha_0 = float(p1["alpha_0"])
    logger.info("Phase I alpha_0=%.4f (n_train=%d)", alpha_0, len(y_train))

    # --- AR(1) ---
    t0 = time.perf_counter()
    ar1 = estimate_ar1_params(y_train)
    elapsed_ar1 = time.perf_counter() - t0
    logger.info("AR(1): phi=%.4f  sigma_v^2=%.4e  sigma_eta^2=%.4e  (%.2fs)",
                ar1["phi"], ar1["sigma_v_sq"], ar1["sigma_eta_sq"], elapsed_ar1)

    lambda_param = 1.0 / (ar1["sigma_eta_sq"] + 1e-12)
    gamma = 1.0 / (ar1["sigma_v_sq"] + 1e-12)

    # --- EVI field + GPD penalty ---
    t0 = time.perf_counter()
    xi_field = compute_evi_field(
        y_train,
        w=p1["window_halfwidth_w"],
        q0=p1["gpd_percentile_q0"],
    )
    alpha_gpd = compute_adaptive_penalty(
        xi_field, alpha_0=alpha_0, lambda_ev=p1["evt_sensitivity_lambda_ev"]
    )
    elapsed_evi = time.perf_counter() - t0
    logger.info("EVI field computed (%.2fs)", elapsed_evi)

    # --- Exceedance-count penalty (robustness check) ---
    # Use a flat mean estimate as mu_est proxy (simple running mean)
    mu_est_proxy = np.full_like(y_train, np.mean(y_train))
    alpha_ec = compute_exceedance_count_penalty(
        y_train,
        mu_est=mu_est_proxy,
        sigma_v=float(np.sqrt(ar1["sigma_v_sq"])),
        w=p1["window_halfwidth_w"],
        c=p1["exceedance_multiplier_c"],
        alpha_0=alpha_0,
    )

    # --- DeCAFS (flat/fixed penalty — EVT moves to Phase II features) ---
    # FIXED penalty — no EVT modulation in Phase I (EVT moves to Phase II features)
    alpha_flat = np.full(len(y_train), alpha_0)
    t0 = time.perf_counter()
    decafs_result = ev_decafs(
        y_train,
        alpha_t=alpha_flat,
        lambda_param=lambda_param,
        gamma=gamma,
        phi=ar1["phi"],
    )
    elapsed_decafs = time.perf_counter() - t0
    logger.info(
        "DeCAFS (vanilla/fixed penalty): %d changepoints, cost=%.4f (%.2fs)",
        len(decafs_result["changepoints"]),
        decafs_result["cost"],
        elapsed_decafs,
    )

    return dict(
        ar1=ar1,
        lambda_param=lambda_param,
        gamma=gamma,
        xi_field=xi_field,
        alpha_gpd=alpha_gpd,
        alpha_ec=alpha_ec,
        decafs_result=decafs_result,
        elapsed_ar1=elapsed_ar1,
        elapsed_evi=elapsed_evi,
        elapsed_decafs=elapsed_decafs,
        alpha_0_used=alpha_0,
    )


# =============================================================================
# BIC sweep
# =============================================================================

def run_bic_sweep(
    y_train: np.ndarray,
    ar1: dict,
    xi_field: np.ndarray,
    params: dict,
) -> tuple[float, float, pd.DataFrame]:
    """Sweep BIC multiplier values and return the best alpha_0.

    Reuses pre-computed AR(1) and EVI field so only EV-DeCAFS is re-run
    for each candidate C value (fast: ~5s per C).

    Returns
    -------
    chosen_alpha_0, chosen_C, sweep_df
    """
    p1 = params["phase1"]
    sweep_values = p1.get("bic_sweep_values", [1.5, 2.0, 3.0, 4.0, 5.0])
    expected = params["evaluation"].get("expected_n_changepoints", 12)
    lambda_param = 1.0 / (ar1["sigma_eta_sq"] + 1e-12)
    gamma = 1.0 / (ar1["sigma_v_sq"] + 1e-12)

    logger.info("=== BIC Multiplier Sweep (expected ~%d CPs) ===", expected)
    rows = []
    for C in sweep_values:
        a0 = compute_bic_penalty(len(y_train), C)
        at = np.full(len(y_train), a0)  # flat penalty for BIC sweep
        res = ev_decafs(y_train, at, lambda_param, gamma, ar1["phi"])
        n_det = len(res["changepoints"])
        rows.append({"C": C, "alpha_0": round(a0, 4),
                     "alpha_t_flat": round(float(at.mean()), 4),
                     "n_detected": n_det, "cost": round(res["cost"], 4)})
        logger.info("  C=%.1f: alpha_0=%.2f (flat), n_detected=%d",
                    C, a0, n_det)

    sweep_df = pd.DataFrame(rows)

    # New criterion: largest C where n_detected >= expected
    # This avoids underfitting by preferring conservative penalties that still meet the target.
    qualifying = sweep_df[sweep_df["n_detected"] >= expected]
    if len(qualifying) > 0:
        best_idx = int(qualifying["C"].idxmax())
    else:
        # Fallback: closest to target (original behaviour)
        best_idx = int((sweep_df["n_detected"] - expected).abs().argmin())
        logger.warning(
            "BIC sweep fallback: no C gave n_detected >= %d; "
            "selecting closest (C=%.2f, n_detected=%d).",
            expected,
            float(sweep_df.loc[best_idx, "C"]),
            int(sweep_df.loc[best_idx, "n_detected"]),
        )

    chosen_C = float(sweep_df.loc[best_idx, "C"])
    chosen_alpha_0 = float(sweep_df.loc[best_idx, "alpha_0"])
    logger.info("Auto-selected C=%.1f (n_detected=%d, target=%d)",
                chosen_C, sweep_df.loc[best_idx, "n_detected"], expected)
    return chosen_alpha_0, chosen_C, sweep_df


# =============================================================================
# Phase II helpers
# =============================================================================

def run_phase2_train(
    y_train: np.ndarray,
    phase1: dict,
    params: dict,
    **extra_kwargs,
) -> dict:
    """Extract features, label, SMOTE-balance, and fit FPNN.

    Returns a dict with keys: X_train, labels_train, X_balanced,
    y_balanced, kappa_mu, fpnn, elapsed_features, elapsed_smote,
    elapsed_fpnn, labels_train_bocpd, labels_train_cusum.

    extra_kwargs
    ------------
    true_cps_train : array-like or None
        Ground-truth CP indices in training coordinates (for relabelling).
    """
    lp = params["labelling"]
    sp = params["smote"]
    fp = params["fpnn"]

    cps = phase1["decafs_result"]["changepoints"]
    means = phase1["decafs_result"]["means"]

    if len(cps) == 0:
        logger.warning("No training changepoints detected — skipping Phase II.")
        return {}

    # --- Feature extraction (5 features when xi_field available) ---
    t0 = time.perf_counter()
    X_train, feature_names = extract_features(
        y_train, cps, means, L=lp["window_L"], xi_field=phase1.get("xi_field")
    )
    logger.info("Training features: %d features — %s", X_train.shape[1], feature_names)
    elapsed_features = time.perf_counter() - t0

    # --- Labelling ---
    kappa_mu = compute_kappa_mu(X_train, percentile=lp["kappa_mu_percentile"])
    labels_train = label_changepoints(X_train, kappa_mu, kappa_S=lp["kappa_S"])
    logger.info(
        "Labels: %d sustained, %d recoiled (kappa_mu=%.4f)",
        labels_train.sum(), (labels_train == 0).sum(), kappa_mu,
    )

    # --- X-Algorithm relabelling (BOCPD primary, CUSUM ablation) ---
    hcpd_cfg = params.get("hypersensitive_cpd", {})
    bocpd_threshold = hcpd_cfg.get("bocpd_threshold", 0.5)
    cusum_h = hcpd_cfg.get("cusum_h_multiplier", 1.0)
    tol_window = hcpd_cfg.get("relabel_tolerance_window", 10)

    ar1 = phase1["ar1"]
    phi_est = ar1["phi"]
    sigma_v_est = float(np.sqrt(ar1["sigma_v_sq"]))

    bocpd_flags = run_bocpd(y_train, phi=phi_est, sigma_v=sigma_v_est,
                            threshold=bocpd_threshold)
    cusum_flags = run_cusum(y_train, phi=phi_est, sigma_v=sigma_v_est,
                            h_multiplier=cusum_h)

    # true_cps_train: passed down from run_dataset via extra_kwargs
    true_cps_train = extra_kwargs.get("true_cps_train", None)

    labels_train_relabelled_bocpd = relabel_with_hypersensitive(
        cp_indices=cps,
        x_flags=bocpd_flags,
        true_cp_indices=true_cps_train,
        existing_labels=labels_train,
        tolerance=tol_window,
    )
    labels_train_relabelled_cusum = relabel_with_hypersensitive(
        cp_indices=cps,
        x_flags=cusum_flags,
        true_cp_indices=true_cps_train,
        existing_labels=labels_train,
        tolerance=tol_window,
    )
    logger.info("BOCPD relabelling: %s", dict(zip(*np.unique(labels_train_relabelled_bocpd, return_counts=True))))
    logger.info("CUSUM relabelling:  %s", dict(zip(*np.unique(labels_train_relabelled_cusum, return_counts=True))))

    # --- BOCPD-based primary labelling (Change 3) ---
    # Convert boolean flags to CP indices for label_with_bocpd
    bocpd_cps = np.where(bocpd_flags)[0]
    has_gt = (true_cps_train is not None and len(true_cps_train) > 0)
    bocpd_cfg = params.get("bocpd", {})
    tol_bocpd = int(bocpd_cfg.get("tolerance_fraction", 0.02) * len(y_train))

    labels_bocpd, label_reasons = label_with_bocpd(
        decafs_cps=cps,
        bocpd_cps=bocpd_cps,
        true_cps=np.asarray(true_cps_train, dtype=int) if has_gt else np.array([], dtype=int),
        tolerance=tol_bocpd,
        has_ground_truth=has_gt,
    )

    # Resolve pending labels (-1) for datasets without ground truth
    if not has_gt:
        labels_bocpd = refine_pending_labels(labels_bocpd, X_train, kappa_mu,
                                              kappa_S=lp["kappa_S"])

    # One-class fallback: if BOCPD gives single class, fall back to feature labels
    n_sus_bocpd = int(np.sum(labels_bocpd == 1))
    n_rec_bocpd = int(np.sum(labels_bocpd == 0))
    if n_sus_bocpd == 0 or n_rec_bocpd == 0:
        logger.warning(
            "BOCPD labelling produced single class (%d sustained, %d recoiled) — "
            "falling back to feature-based labels",
            n_sus_bocpd, n_rec_bocpd,
        )
        labels_primary = labels_train  # feature-based fallback
    else:
        labels_primary = labels_bocpd

    # Log per-CP decisions at DEBUG level
    for idx, (cp, lbl) in enumerate(zip(cps, labels_primary)):
        reason = label_reasons[idx] if idx < len(label_reasons) else ""
        logger.debug("  CP[%d] at t=%d: label=%d (%s)", idx, cp, lbl, reason)

    # BOCPD summary statistics (Change 5)
    bocpd_summary = {
        "n_bocpd_detections": len(bocpd_cps),
        "n_decafs_detections": len(cps),
        "n_true_cps": len(true_cps_train) if has_gt else 0,
        "has_ground_truth": has_gt,
        "label_distribution": f"{n_sus_bocpd} sustained, {n_rec_bocpd} recoiled",
    }
    logger.info("BOCPD summary: %s", bocpd_summary)

    # --- SMOTE ---
    t0 = time.perf_counter()
    X_balanced, y_balanced = balance_training_data(
        X_train, labels_primary,
        k_neighbors=sp["k_neighbors"],
        random_state=sp["random_state"],
    )
    elapsed_smote = time.perf_counter() - t0

    # --- FPNN ---
    t0 = time.perf_counter()
    fpnn = FourierPNN(
        J=fp["J_harmonics"],
        scaling_range=tuple(fp["scaling_range"]),
    )
    fpnn.fit(X_balanced, y_balanced)
    elapsed_fpnn = time.perf_counter() - t0
    logger.info("FPNN fitted (J=%d, %.2fs)", fp["J_harmonics"], elapsed_fpnn)

    return dict(
        X_train=X_train,
        labels_train=labels_train,           # feature-based (Algorithm 3)
        labels_primary=labels_primary,       # BOCPD-based primary (used for FPNN)
        labels_bocpd_raw=labels_bocpd,       # BOCPD labels before fallback
        labels_train_bocpd=labels_train_relabelled_bocpd,
        labels_train_cusum=labels_train_relabelled_cusum,
        bocpd_summary=bocpd_summary,
        kappa_mu=kappa_mu,
        X_balanced=X_balanced,
        y_balanced=y_balanced,
        fpnn=fpnn,
        elapsed_features=elapsed_features,
        elapsed_smote=elapsed_smote,
        elapsed_fpnn=elapsed_fpnn,
    )


def run_phase2_test(
    y_test: np.ndarray,
    phase1_train: dict,
    phase2_train: dict,
    params: dict,
    skip_baselines: bool,
) -> dict:
    """Apply EV-DeCAFS to test data, extract features, classify, evaluate."""
    lp = params["labelling"]
    p1 = params["phase1"]

    # Use the same alpha_0 that was chosen during training (BIC-tuned or fixed)
    alpha_0_test = phase1_train.get("alpha_0_used", float(p1.get("alpha_0", 10.0)))

    # --- DeCAFS on test (flat penalty) ---
    t0 = time.perf_counter()
    xi_field_test = compute_evi_field(
        y_test,
        w=p1["window_halfwidth_w"],
        q0=p1["gpd_percentile_q0"],
    )
    # Flat penalty — EVT contributes via Phase II features, not Phase I penalty
    alpha_test = np.full(len(y_test), alpha_0_test)
    res_test = ev_decafs(
        y_test,
        alpha_t=alpha_test,
        lambda_param=phase1_train["lambda_param"],
        gamma=phase1_train["gamma"],
        phi=phase1_train["ar1"]["phi"],
    )
    elapsed_test_decafs = time.perf_counter() - t0
    logger.info(
        "Test DeCAFS (flat penalty): %d changepoints (%.2fs)",
        len(res_test["changepoints"]),
        elapsed_test_decafs,
    )

    cps_test = res_test["changepoints"]
    means_test = res_test["means"]

    if len(cps_test) == 0:
        logger.warning("No test changepoints detected — skipping test classification.")
        return {"decafs_result": res_test}

    # --- Feature extraction on test (include xi_field as 5th feature) ---
    X_test, _ = extract_features(y_test, cps_test, means_test, L=lp["window_L"],
                                 xi_field=xi_field_test)

    # Auto-label test changepoints using training thresholds
    labels_test = label_changepoints(
        X_test,
        kappa_mu=phase2_train["kappa_mu"],
        kappa_S=lp["kappa_S"],
    )

    # --- Classification evaluation (always includes baselines) ---
    fpnn = phase2_train["fpnn"]
    t0 = time.perf_counter()
    logger.info("Training baseline classifiers...")
    baselines = get_baselines(params["baselines"], input_dim=X_test.shape[1])
    clf_df = train_and_evaluate_all(
        phase2_train["X_balanced"],
        phase2_train["y_balanced"],
        X_test,
        labels_test,
        baselines,
        fpnn,
    )
    logger.info("Baseline comparison:\n%s", clf_df.to_string())
    roc_results = {
        "FPNN": {
            "y_true": labels_test,
            "y_proba": fpnn.predict_proba(X_test)[:, 1],
        }
    }
    elapsed_clf = time.perf_counter() - t0

    return dict(
        decafs_result=res_test,
        X_test=X_test,
        labels_test=labels_test,
        clf_df=clf_df,
        roc_results=roc_results,
        elapsed_test_decafs=elapsed_test_decafs,
        elapsed_clf=elapsed_clf,
    )


# =============================================================================
# MRL + sensitivity analysis
# =============================================================================

def run_mrl_analysis(
    phase1_train: dict,
    phase2_test: dict,
    true_cps_test_relative: np.ndarray,
    params: dict,
) -> dict | None:
    """Compute MRL, Hausdorff, and cost-ratio sensitivity for the well-log.

    Parameters
    ----------
    true_cps_test_relative:
        True changepoint indices already converted to **test-relative**
        coordinates (i.e., absolute_index - len(y_train)).  Use
        :func:`~src.evaluation.mrl_index.convert_to_relative` to prepare this.
    """
    if true_cps_test_relative is None or len(true_cps_test_relative) == 0:
        logger.info("No ground-truth changepoints in test set — skipping MRL analysis.")
        return None

    ep = params["evaluation"]
    n_test = len(phase2_test.get("decafs_result", {}).get("means", [1]))
    Tmax = ep["censoring_Tmax_fraction"] * n_test
    epsilon = ep["censoring_epsilon"]
    tolerance = int(ep["hausdorff_tolerance_fraction"] * n_test)

    # detected CPs are already in test-relative coordinates
    detected = np.asarray(phase2_test["decafs_result"]["changepoints"])

    logger.info(
        "MRL evaluation: %d detected CPs vs %d true CPs (test-relative), "
        "tolerance=%d",
        len(detected),
        len(true_cps_test_relative),
        tolerance,
    )

    boundary_window = int(ep.get("boundary_exclusion_window", 0))
    mrl_result = compute_mrl(
        detected, true_cps_test_relative,
        tolerance=tolerance,
        boundary_exclusion_window=boundary_window,
    )
    logger.info(
        "MRL result: FP=%d, MRL=%.2f, n_missed=%d/%d",
        mrl_result["FP"],
        mrl_result["MRL"],
        mrl_result["n_missed"],
        mrl_result["n_true_cps"],
    )

    detectors_summary = {
        "EV-DeCAFS (GPD)": {"FP": mrl_result["FP"], "MRL": mrl_result["MRL"]},
    }

    # Hausdorff distance (use test-relative for both)
    h = symmetric_hausdorff(detected, true_cps_test_relative)
    logger.info("Hausdorff distance (EV-DeCAFS vs true, test-relative): %.2f", h)

    rankings, raw_rtilde = cost_ratio_sensitivity(
        detectors_summary,
        cF_grid=ep["cost_cF_grid"],
        cD_grid=ep["cost_cD_grid"],
        epsilon=epsilon,
        Tmax=Tmax,
    )

    return dict(
        mrl_result=mrl_result,
        detectors_summary=detectors_summary,
        hausdorff=h,
        rankings=rankings,
        raw_rtilde=raw_rtilde,
    )


# =============================================================================
# Per-dataset orchestration
# =============================================================================

def run_dataset(
    name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    true_cps: np.ndarray | None,
    true_outliers: np.ndarray | None,
    params: dict,
    output_dir: Path,
    skip_baselines: bool,
    is_financial: bool = False,
    data_dict: dict | None = None,
) -> dict:
    """Run the full pipeline for a single dataset. Returns timing dict."""
    logger.info("=" * 60)
    logger.info("Dataset: %s  (train=%d, test=%d)", name, len(y_train), len(y_test))
    logger.info("=" * 60)

    fig_dir = output_dir / "figures"
    tbl_dir = output_dir / "tables"
    fig_fmt = params["visualization"].get("figure_format", "pdf")

    timings: dict[str, float] = {}

    # ---- BIC sweep (optional, before main Phase I) ----
    chosen_alpha_0 = None
    if params["phase1"].get("tune_bic", False) and \
            params["phase1"].get("alpha_0_mode", "fixed") == "bic":
        # Pre-compute AR(1) and EVI for the sweep (reused by run_phase1 below)
        _ar1_sweep = estimate_ar1_params(y_train)
        _xi_sweep = compute_evi_field(
            y_train,
            w=params["phase1"]["window_halfwidth_w"],
            q0=params["phase1"]["gpd_percentile_q0"],
        )
        chosen_alpha_0, chosen_C, sweep_df = run_bic_sweep(
            y_train, _ar1_sweep, _xi_sweep, params
        )
        sweep_df.to_csv(tbl_dir / f"{name}_bic_sweep.csv", index=False)
        logger.info("BIC sweep saved: %s", tbl_dir / f"{name}_bic_sweep.csv")

    # ---- Phase I (training) ----
    t_phase1_start = time.perf_counter()
    phase1 = run_phase1(y_train, params, alpha_0=chosen_alpha_0)
    timings["phase1_total"] = time.perf_counter() - t_phase1_start

    # ---- Phase II training ----
    # Compute training-relative true CPs early so relabelling can use them
    if true_cps is not None and len(true_cps) > 0:
        _true_cps_train_early = np.asarray(
            [cp for cp in true_cps if cp < len(y_train)]
        )
    else:
        _true_cps_train_early = np.array([], dtype=int)

    t_phase2_start = time.perf_counter()
    phase2_train = run_phase2_train(
        y_train, phase1, params, true_cps_train=_true_cps_train_early
    )
    timings["phase2_train"] = time.perf_counter() - t_phase2_start

    if not phase2_train:
        logger.warning("[%s] Aborting: no training changepoints.", name)
        return timings

    # ---- Phase II test ----
    phase2_test = run_phase2_test(
        y_test, phase1, phase2_train, params, skip_baselines
    )
    timings["phase2_test"] = phase2_test.get("elapsed_clf", 0.0)

    # ---- MRL analysis (skipped for financial datasets) ----
    # Convert absolute true CP indices to test-relative coordinates.
    # Detected CPs from ev_decafs on y_test are already test-relative.
    if is_financial:
        logger.info("Financial dataset — skipping MRL/sensitivity analysis.")
        true_cps_test_rel = np.array([])
        true_cps_train_only = np.array([])
        mrl_analysis = None
    elif true_cps is not None and len(true_cps) > 0:
        true_cps_test_rel, n_excluded = convert_to_relative(
            np.asarray(true_cps), split_index=len(y_train)
        )
        logger.info(
            "True CPs in test set: %d (excluded %d that fall in training set)",
            len(true_cps_test_rel),
            n_excluded,
        )
        logger.info("Test-relative true CP positions: %s", true_cps_test_rel)
        # Training-relative true CPs (absolute = relative since train starts at 0)
        true_cps_train_only = np.asarray(
            [cp for cp in true_cps if cp < len(y_train)]
        )
        mrl_analysis = run_mrl_analysis(phase1, phase2_test, true_cps_test_rel, params)
    else:
        true_cps_test_rel = np.array([])
        true_cps_train_only = np.array([])
        mrl_analysis = run_mrl_analysis(phase1, phase2_test, true_cps_test_rel, params)

    # ---- Phase I baseline detector comparison ----
    alpha_0_used = phase1.get("alpha_0_used", params["phase1"]["alpha_0"])
    if "decafs_result" in phase2_test:
        _run_phase1_detector_comparison(
            y_test=y_test,
            evdecafs_cps=np.asarray(phase2_test["decafs_result"]["changepoints"]),
            true_cps_test_rel=true_cps_test_rel,
            alpha_0=alpha_0_used,
            phase1=phase1,
            params=params,
            tbl_dir=tbl_dir,
            fig_dir=fig_dir,
            name=name,
            fig_fmt=fig_fmt,
        )

    # ---- Save tables ----
    if "clf_df" in phase2_test:
        clf_path = tbl_dir / f"{name}_classification_results.csv"
        phase2_test["clf_df"].to_csv(clf_path)
        logger.info("Classification table saved: %s", clf_path)

    if mrl_analysis is not None:
        mrl_path = tbl_dir / f"{name}_mrl_summary.csv"
        pd.DataFrame(mrl_analysis["detectors_summary"]).T.to_csv(mrl_path)
        mrl_analysis["rankings"].to_csv(tbl_dir / f"{name}_sensitivity_rankings.csv")
        mrl_analysis["raw_rtilde"].to_csv(tbl_dir / f"{name}_sensitivity_rtilde.csv")
        logger.info("MRL / sensitivity tables saved.")

    # ---- Tail diagnostics ----
    _run_tail_diagnostics_section(
        name=name,
        y_train=y_train,
        y_test=y_test,
        params=params,
        tbl_dir=tbl_dir,
        fig_dir=fig_dir,
    )

    # ---- Monte Carlo coverage simulation ----
    if "fpnn" in phase2_train:
        _run_monte_carlo_section(
            name=name,
            phase1_train=phase1,
            phase2_train=phase2_train,
            params=params,
            tbl_dir=tbl_dir,
            fig_dir=fig_dir,
            fig_fmt=fig_fmt,
        )

    # ---- Save runtime ----
    timings["phase1_ar1"] = phase1.get("elapsed_ar1", 0.0)
    timings["phase1_evi"] = phase1.get("elapsed_evi", 0.0)
    timings["phase1_decafs"] = phase1.get("elapsed_decafs", 0.0)
    timings["phase2_features"] = phase2_train.get("elapsed_features", 0.0)
    timings["phase2_smote"] = phase2_train.get("elapsed_smote", 0.0)
    timings["phase2_fpnn"] = phase2_train.get("elapsed_fpnn", 0.0)
    pd.Series(timings, name=name).to_csv(
        tbl_dir / f"{name}_runtime.csv", header=True
    )

    # ---- Figures ----
    _make_figures(
        name=name,
        y_train=y_train,
        y_test=y_test,
        phase1=phase1,
        phase2_test=phase2_test,
        mrl_analysis=mrl_analysis,
        true_cps_train=true_cps_train_only,
        true_cps_test_rel=true_cps_test_rel,
        fig_dir=fig_dir,
        fig_fmt=fig_fmt,
        is_financial=is_financial,
        data_dict=data_dict,
        phase2_train=phase2_train,
    )

    return timings


def _run_tail_diagnostics_section(
    name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    tbl_dir: Path,
    fig_dir: Path | None = None,
) -> None:
    """Run GPD tail diagnostics on train and test sets, save CSV tables and figures."""
    p1 = params["phase1"]
    w = p1["window_halfwidth_w"]
    q0 = p1["gpd_percentile_q0"]
    fig_fmt = params["visualization"].get("figure_format", "pdf")

    for split_name, y_split in [("train", y_train), ("test", y_test)]:
        if len(y_split) < 2 * w + 5:
            logger.warning("[%s] %s set too short for tail diagnostics (n=%d).",
                           name, split_name, len(y_split))
            continue
        if split_name == "test" and len(y_split) <= 200:
            logger.info("[%s] Test set too short for tail diagnostics figures (n=%d).",
                        name, len(y_split))
            continue
        try:
            summary, per_window = run_tail_diagnostics(
                y_split, w=w, q0=q0, n_windows=100, n_bootstrap=200, seed=42,
            )
            per_window.to_csv(
                tbl_dir / f"{name}_tail_diagnostics_{split_name}.csv", index=False
            )
            pd.DataFrame([summary]).to_csv(
                tbl_dir / f"{name}_tail_summary_{split_name}.csv", index=False
            )
            logger.info("[%s] Tail diagnostics (%s): %s", name, split_name,
                        summary.get("overall_classification", "?"))
            # Generate tail diagnostics figure
            if fig_dir is not None:
                try:
                    plot_tail_diagnostics(
                        per_window, summary, name, split_name.capitalize(),
                        save_path=str(fig_dir / f"{name}_tail_diagnostics_{split_name}.{fig_fmt}"),
                    )
                except Exception as fig_exc:
                    logger.warning("[%s] Tail diagnostics figure (%s) failed: %s",
                                   name, split_name, fig_exc)
        except Exception as exc:
            logger.warning("[%s] Tail diagnostics (%s) failed: %s", name, split_name, exc)


def _run_phase1_detector_comparison(
    y_test: np.ndarray,
    evdecafs_cps: np.ndarray,
    true_cps_test_rel: np.ndarray,
    alpha_0: float,
    phase1: dict,
    params: dict,
    tbl_dir: Path,
    fig_dir: Path,
    name: str,
    fig_fmt: str,
) -> None:
    """Run all Phase I baseline detectors on y_test and compare via MRL."""
    ep = params["evaluation"]
    n_test = len(y_test)
    Tmax = ep["censoring_Tmax_fraction"] * n_test
    epsilon = ep["censoring_epsilon"]
    tolerance = int(ep["hausdorff_tolerance_fraction"] * n_test)

    logger.info("Running Phase I baseline detectors on test set...")
    baseline_cps = run_all_baseline_detectors(
        y_test,
        pen=alpha_0,
        decafs_params={
            "alpha_0": alpha_0,
            "lambda_param": phase1["lambda_param"],
            "gamma": phase1["gamma"],
            "phi": phase1["ar1"]["phi"],
        },
    )

    all_detectors = {"EV-DeCAFS (proposed)": evdecafs_cps}
    all_detectors.update(baseline_cps)

    rows = []
    for det_name, cps in all_detectors.items():
        cps_arr = np.asarray(cps, dtype=int)
        h = symmetric_hausdorff(cps_arr, true_cps_test_rel) if len(true_cps_test_rel) > 0 else np.nan
        if len(true_cps_test_rel) > 0:
            mrl_r = compute_mrl(cps_arr, true_cps_test_rel, tolerance=tolerance,
                                boundary_exclusion_window=int(ep.get("boundary_exclusion_window", 0)))
            fp = mrl_r["FP"]
            mrl_val = mrl_r["MRL"]
            n_missed = mrl_r["n_missed"]
        else:
            fp, mrl_val, n_missed = len(cps_arr), np.nan, 0
        r_tilde = compute_censored_risk(fp, mrl_val, cF=1.0, cD=1.0,
                                        epsilon=epsilon, Tmax=Tmax)
        rows.append({
            "Detector": det_name,
            "n_detected": len(cps_arr),
            "FP": fp,
            "MRL": round(mrl_val, 2) if np.isfinite(mrl_val) else np.inf,
            "n_missed": n_missed,
            "Hausdorff": round(h, 2) if np.isfinite(h) else np.nan,
            "R_tilde(cF=1,cD=1)": round(r_tilde, 4) if np.isfinite(r_tilde) else np.inf,
        })
        logger.info("  %s: n=%d, FP=%d, MRL=%.1f, Hausdorff=%.1f",
                    det_name, len(cps_arr), fp,
                    mrl_val if np.isfinite(mrl_val) else float("inf"), h)

    comp_df = pd.DataFrame(rows).set_index("Detector")
    comp_path = tbl_dir / f"{name}_phase1_detector_comparison.csv"
    comp_df.to_csv(comp_path)
    logger.info("Phase I detector comparison saved: %s", comp_path)

    # Multi-detector sensitivity analysis (tables only — heatmaps removed)
    if len(true_cps_test_rel) > 0:
        det_summary = {
            r["Detector"]: {"FP": r["FP"], "MRL": r["MRL"]}
            for r in rows if np.isfinite(r["MRL"])
        }
        if det_summary:
            rankings, raw_rtilde = cost_ratio_sensitivity(
                det_summary,
                cF_grid=ep["cost_cF_grid"],
                cD_grid=ep["cost_cD_grid"],
                epsilon=epsilon,
                Tmax=Tmax,
            )
            rankings.to_csv(tbl_dir / f"{name}_multidet_sensitivity_rankings.csv")
            raw_rtilde.to_csv(tbl_dir / f"{name}_multidet_sensitivity_rtilde.csv")
            logger.info("Multi-detector sensitivity tables saved.")

    # Phase I multimetric bar chart (always generated, even with 0 true CPs)
    try:
        detector_results_chart = {}
        for r in rows:
            detector_results_chart[r["Detector"]] = {
                'n_detected': r["n_detected"],
                'FP': r["FP"],
                'n_missed': r["n_missed"],
                'hausdorff': r["Hausdorff"] if np.isfinite(r["Hausdorff"]) else 0,
                'MRL': r["MRL"] if np.isfinite(r["MRL"]) else 0,
            }
        plot_phase1_multimetric(
            detector_results_chart,
            true_n_cps=len(true_cps_test_rel),
            save_path=str(fig_dir / f"{name}_phase1_multimetric.{fig_fmt}"),
        )
        logger.info("Phase I multimetric bar chart saved.")
    except Exception as exc:
        logger.warning("Phase I multimetric chart failed: %s", exc)

    # Multi-detector run chart on test data
    if len(all_detectors) > 1:
        try:
            plot_changepoint_comparison(
                y_test,
                detectors_dict={k: np.asarray(v) for k, v in all_detectors.items()},
                true_cps=true_cps_test_rel if len(true_cps_test_rel) > 0 else np.array([]),
                save_path=fig_dir / f"{name}_phase1_detector_comparison.{fig_fmt}",
            )
            logger.info("Multi-detector comparison figure saved.")
        except Exception as exc:
            logger.warning("Multi-detector figure failed: %s", exc)


def _run_monte_carlo_section(
    name: str,
    phase1_train: dict,
    phase2_train: dict,
    params: dict,
    tbl_dir: Path,
    fig_dir: Path,
    fig_fmt: str,
) -> None:
    """Run Monte Carlo coverage simulation and save results to CSV."""
    mc_params = params.get("monte_carlo", {})
    B = mc_params.get("B", 500)
    logger.info("=" * 60)
    logger.info("Starting Monte Carlo coverage simulation (B=%d)", B)

    p1 = params["phase1"]
    lp = params["labelling"]
    sp = params["smote"]
    fp_params = params["fpnn"]

    def pipeline_single_run(
        y_train_mc, y_test_mc,
        true_cps_train_mc, true_labels_train_mc,
        true_cps_test_mc, true_labels_test_mc,
    ):
        """Single pipeline run for Monte Carlo — returns metric dict for all classifiers."""
        from src.phase1.ar1_model import estimate_ar1_params as _est

        ar1_mc = _est(y_train_mc)
        lp_mc = 1.0 / (ar1_mc["sigma_eta_sq"] + 1e-12)
        gm_mc = 1.0 / (ar1_mc["sigma_v_sq"] + 1e-12)

        # Use BIC penalty if configured
        if p1.get("alpha_0_mode", "fixed") == "bic":
            a0_mc = compute_bic_penalty(len(y_train_mc), p1.get("bic_multiplier", 2.0))
        else:
            a0_mc = float(p1["alpha_0"])

        xi_mc = compute_evi_field(
            y_train_mc, w=p1["window_halfwidth_w"], q0=p1["gpd_percentile_q0"],
        )
        alpha_mc = np.full(len(y_train_mc), a0_mc)  # flat penalty — EVT via features
        res_mc = ev_decafs(y_train_mc, alpha_mc, lp_mc, gm_mc, ar1_mc["phi"])
        cps_mc = res_mc["changepoints"]
        means_mc = res_mc["means"]

        if len(cps_mc) < 2:
            return None  # not enough training CPs to label/SMOTE

        X_tr_mc, _ = extract_features(y_train_mc, cps_mc, means_mc, L=lp["window_L"],
                                      xi_field=xi_mc)
        kmu_mc = compute_kappa_mu(X_tr_mc, percentile=lp["kappa_mu_percentile"])
        lab_feat_mc = label_changepoints(X_tr_mc, kmu_mc, kappa_S=lp["kappa_S"])

        # BOCPD-based labelling for MC training data (Change 4)
        bocpd_flags_mc = run_bocpd(
            y_train_mc, phi=ar1_mc["phi"],
            sigma_v=float(np.sqrt(ar1_mc["sigma_v_sq"])),
            threshold=params.get("bocpd", {}).get("threshold", 0.3),
        )
        bocpd_cps_mc = np.where(bocpd_flags_mc)[0]
        tol_mc_train = int(0.02 * len(y_train_mc))
        has_gt_mc = len(true_cps_train_mc) > 0
        lab_tr_mc, _ = label_with_bocpd(
            decafs_cps=cps_mc,
            bocpd_cps=bocpd_cps_mc,
            true_cps=true_cps_train_mc,
            tolerance=tol_mc_train,
            has_ground_truth=has_gt_mc,
        )
        if not has_gt_mc:
            lab_tr_mc = refine_pending_labels(lab_tr_mc, X_tr_mc, kmu_mc, lp["kappa_S"])

        # Fall back to feature labels if BOCPD gives single class
        if len(np.unique(lab_tr_mc)) < 2:
            lab_tr_mc = lab_feat_mc

        if len(np.unique(lab_tr_mc)) < 2:
            return None  # only one class — can't train classifiers

        X_bal_mc, lab_bal_mc = balance_training_data(
            X_tr_mc, lab_tr_mc,
            k_neighbors=min(sp["k_neighbors"], int(np.bincount(lab_tr_mc).min()) - 1),
            random_state=sp["random_state"],
        )

        # Train FPNN
        fpnn_mc = FourierPNN(J=fp_params["J_harmonics"],
                             scaling_range=tuple(fp_params["scaling_range"]))
        fpnn_mc.fit(X_bal_mc, lab_bal_mc)

        # Train fast baselines (no CV for speed); GRU included via get_baselines
        baselines_mc = get_baselines(params["baselines"], input_dim=X_bal_mc.shape[1])

        # Test set (flat penalty — EVT via Phase II features)
        xi_te_mc = compute_evi_field(
            y_test_mc, w=p1["window_halfwidth_w"], q0=p1["gpd_percentile_q0"],
        )
        alpha_te_mc = np.full(len(y_test_mc), a0_mc)
        res_te_mc = ev_decafs(y_test_mc, alpha_te_mc, lp_mc, gm_mc, ar1_mc["phi"])
        cps_te_mc = res_te_mc["changepoints"]
        means_te_mc = res_te_mc["means"]

        if len(cps_te_mc) == 0:
            return None

        X_te_mc, _ = extract_features(y_test_mc, cps_te_mc, means_te_mc, L=lp["window_L"],
                                      xi_field=xi_te_mc)
        tol_mc = int(0.02 * len(y_test_mc))
        test_labels_mc = assign_nearest_labels(
            cps_te_mc, true_cps_test_mc, true_labels_test_mc, tolerance=tol_mc
        )

        from sklearn.metrics import brier_score_loss, cohen_kappa_score

        out: dict = {}

        # FPNN
        y_pred_fpnn = fpnn_mc.predict(X_te_mc)
        y_proba_fpnn = fpnn_mc.predict_proba(X_te_mc)
        m = compute_classification_metrics(test_labels_mc, y_pred_fpnn, y_proba_fpnn)
        for k in ("balanced_accuracy", "mcc", "auc_roc"):
            out[f"fpnn_{k}"] = m[k]
        try:
            out["fpnn_brier_score"] = brier_score_loss(test_labels_mc, y_proba_fpnn[:, 1])
            out["fpnn_cohen_kappa"] = cohen_kappa_score(test_labels_mc, y_pred_fpnn)
        except Exception:
            out["fpnn_brier_score"] = float("nan")
            out["fpnn_cohen_kappa"] = float("nan")

        # Baselines
        for bname, clf in baselines_mc.items():
            bkey = bname.lower().replace(" ", "_").replace("-", "_")
            try:
                if bname in ("Isolation Forest", "One-Class SVM"):
                    clf.fit(X_bal_mc)
                    raw_pred = clf.predict(X_te_mc)
                    y_pred_b = (raw_pred == 1).astype(int)
                    scores = clf.decision_function(X_te_mc)
                    s_min, s_max = scores.min(), scores.max()
                    y_proba_b = (scores - s_min) / (s_max - s_min + 1e-12)
                    y_proba_b = np.column_stack([1 - y_proba_b, y_proba_b])
                else:
                    clf.fit(X_bal_mc, lab_bal_mc)
                    y_pred_b = clf.predict(X_te_mc)
                    if hasattr(clf, "predict_proba"):
                        y_proba_b = clf.predict_proba(X_te_mc)
                    else:
                        y_proba_b = None
                mb = compute_classification_metrics(test_labels_mc, y_pred_b, y_proba_b)
                for k in ("balanced_accuracy", "mcc", "auc_roc"):
                    out[f"{bkey}_{k}"] = mb[k]
                try:
                    if y_proba_b is not None:
                        out[f"{bkey}_brier_score"] = brier_score_loss(test_labels_mc, y_proba_b[:, 1])
                    else:
                        out[f"{bkey}_brier_score"] = float("nan")
                    out[f"{bkey}_cohen_kappa"] = cohen_kappa_score(test_labels_mc, y_pred_b)
                except Exception:
                    out[f"{bkey}_brier_score"] = float("nan")
                    out[f"{bkey}_cohen_kappa"] = float("nan")
            except Exception:
                for k in ("balanced_accuracy", "mcc", "auc_roc", "brier_score", "cohen_kappa"):
                    out[f"{bkey}_{k}"] = float("nan")

        return out

    series_p = {
        "n": mc_params.get("series_n", 2000),
        "n_changepoints": mc_params.get("n_changepoints", 8),
        "n_outliers": mc_params.get("n_outliers", 15),
        "phi": mc_params.get("phi", 0.5),
        "sigma_v": mc_params.get("sigma_v", 2000.0),
    }

    mc_results = run_monte_carlo(
        pipeline_func=pipeline_single_run,
        B=B,
        train_fraction=params["splitting"]["welllog_train_fraction"],
        series_params=series_p,
        seed=mc_params.get("seed", 42),
    )

    # Save raw metric summary
    mc_rows = {
        mname: [r["mean"], r["std"], r["ci_lower"], r["ci_upper"], r["median"]]
        for mname, r in mc_results.items()
    }
    mc_df = pd.DataFrame(
        mc_rows, index=["mean", "std", "ci_lower", "ci_upper", "median"]
    ).T
    mc_path = tbl_dir / f"{name}_monte_carlo_coverage.csv"
    mc_df.to_csv(mc_path)
    logger.info("Monte Carlo coverage results saved to %s", mc_path)

    # Build per-classifier summary table with all 5 metrics
    clf_prefixes = {
        "FPNN": "fpnn",
        "Logistic Regression": "logistic_regression",
        "Isolation Forest": "isolation_forest",
        "One-Class SVM": "one_class_svm",
        "Feedforward NN": "feedforward_nn",
        "GRU (RNN)": "gru_(rnn)",
    }
    clf_rows = []
    for clf_name, prefix in clf_prefixes.items():
        row = {"Classifier": clf_name}
        for metric in ("balanced_accuracy", "mcc", "auc_roc", "brier_score", "cohen_kappa"):
            key = f"{prefix}_{metric}"
            if key in mc_results:
                r = mc_results[key]
                row[f"{metric}_mean"] = round(r["mean"], 4)
                row[f"{metric}_std"] = round(r["std"], 4)
                row[f"{metric}_ci_lower"] = round(r["ci_lower"], 4)
                row[f"{metric}_ci_upper"] = round(r["ci_upper"], 4)
            else:
                for sfx in ("mean", "std", "ci_lower", "ci_upper"):
                    row[f"{metric}_{sfx}"] = float("nan")
        clf_rows.append(row)

    clf_mc_df = pd.DataFrame(clf_rows).set_index("Classifier")
    clf_mc_path = tbl_dir / f"{name}_monte_carlo_all_classifiers.csv"
    clf_mc_df.to_csv(clf_mc_path)
    # Also write to legacy path for backward compat (first dataset wins)
    legacy_path = tbl_dir / "monte_carlo_all_classifiers.csv"
    if not legacy_path.exists():
        clf_mc_df.to_csv(legacy_path)
    logger.info("Per-classifier MC table saved to %s", clf_mc_path)
    logger.info("MC classifier summary:\n%s", clf_mc_df.to_string())

    # Violin plot figure (replaces box plot)
    try:
        violin_data = {}
        for clf_name, prefix in clf_prefixes.items():
            clf_data = {}
            for metric in ("balanced_accuracy", "mcc", "brier_score", "cohen_kappa"):
                key = f"{prefix}_{metric}"
                if key in mc_results:
                    clf_data[metric] = mc_results[key].get("values", np.array([]))
            if clf_data:
                violin_data[clf_name] = clf_data
        if violin_data:
            plot_mc_classifier_comparison(
                violin_data,
                save_path=str(fig_dir / f"{name}_fig_mc_classifier_comparison.{fig_fmt}"),
            )
    except Exception as exc:
        logger.warning("MC violin plot failed: %s", exc)
        # Fallback to box plot
        _make_mc_boxplot(mc_results, clf_prefixes, fig_dir, fig_fmt, name=name)


def _make_mc_boxplot(
    mc_results: dict,
    clf_prefixes: dict,
    fig_dir: Path,
    fig_fmt: str,
    name: str = "welllog",
) -> None:
    """Box plots of balanced accuracy across classifiers from MC simulation."""
    try:
        import matplotlib.pyplot as plt
        from src.visualization.style import apply_style
        apply_style()

        metric = "balanced_accuracy"
        data = []
        labels = []
        for clf_name, prefix in clf_prefixes.items():
            key = f"{prefix}_{metric}"
            if key in mc_results:
                vals = mc_results[key]["values"]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    data.append(vals)
                    labels.append(clf_name)

        if not data:
            logger.warning("No MC data available for box plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        colors = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4"]
        for patch, color in zip(bp["boxes"], colors[: len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random chance")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title("Monte Carlo Coverage — Balanced Accuracy by Classifier")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        save_path = fig_dir / f"{name}_fig_mc_classifier_comparison.{fig_fmt}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("MC box plot saved: %s", save_path)
    except Exception as exc:
        logger.warning("MC box plot failed: %s", exc)


def _make_figures(
    name, y_train, y_test, phase1, phase2_test,
    mrl_analysis, true_cps_train, true_cps_test_rel, fig_dir, fig_fmt,
    is_financial: bool = False,
    data_dict: dict | None = None,
    phase2_train: dict | None = None,
):
    ext = fig_fmt
    labels_train = phase2_train.get("labels_train", np.array([])) if phase2_train else np.array([])
    labels_test = phase2_test.get("labels_test", np.array([]))

    # Training run chart — coordinate-correct true CPs
    plot_run_chart(
        y_train,
        detected_cps=phase1["decafs_result"]["changepoints"],
        true_cps=true_cps_train,
        means=phase1["decafs_result"]["means"],
        title=f"{name} — Training (DeCAFS-FPNN Proposed)",
        save_path=fig_dir / f"{name}_train_run_chart.{ext}",
        labels=labels_train if len(labels_train) > 0 else None,
    )

    # Test run chart — coordinate-correct true CPs (test-relative)
    if "decafs_result" in phase2_test:
        plot_run_chart(
            y_test,
            detected_cps=phase2_test["decafs_result"]["changepoints"],
            true_cps=true_cps_test_rel,
            means=phase2_test["decafs_result"]["means"],
            title=f"{name} — Test (DeCAFS-FPNN Proposed)",
            save_path=fig_dir / f"{name}_test_run_chart.{ext}",
            labels=labels_test if len(labels_test) > 0 else None,
        )

    # Phase II classification figure (THE KEY FIGURE)
    if "decafs_result" in phase2_test and len(labels_train) > 0:
        try:
            plot_phase2_train_test(
                y_train=y_train,
                y_test=y_test,
                cps_train=phase1["decafs_result"]["changepoints"],
                cps_test=phase2_test["decafs_result"]["changepoints"],
                means_train=phase1["decafs_result"]["means"],
                means_test=phase2_test["decafs_result"]["means"],
                labels_train=labels_train,
                labels_test=labels_test if len(labels_test) > 0 else np.array([]),
                true_cps_train=true_cps_train if true_cps_train is not None else np.array([]),
                true_cps_test=true_cps_test_rel if true_cps_test_rel is not None else np.array([]),
                dataset_name=name,
                save_dir=str(fig_dir),
            )
        except Exception as exc:
            logger.warning("Phase II classification figure failed: %s", exc)

    # US IP annotated figure (special figure for this dataset)
    if name == "us_ip_growth" and data_dict is not None and "decafs_result" in phase2_test:
        try:
            all_dates = np.concatenate([data_dict['dates_train'], data_dict['dates_test']])
            all_index = np.concatenate([data_dict['index_train'], data_dict['index_test']])
            all_cps = list(phase1["decafs_result"]["changepoints"])
            all_labels = list(labels_train)
            if len(phase2_test["decafs_result"]["changepoints"]) > 0:
                for cp in phase2_test["decafs_result"]["changepoints"]:
                    all_cps.append(cp + len(y_train))
                all_labels.extend(list(labels_test))
            plot_us_ip_annotated(
                index_values=all_index,
                dates=all_dates,
                detected_cps=np.array(all_cps),
                labels=np.array(all_labels),
                nber_dates=data_dict.get('nber_dates', []),
                train_end_idx=len(y_train),
                save_path=str(fig_dir / f"us_ip_growth_annotated.{ext}"),
            )
        except Exception as exc:
            logger.warning("US IP annotated figure failed: %s", exc)


# =============================================================================
# Main
# =============================================================================

# Module-level param store so _make_figures can access it without threading
params_global: dict = {}


def main() -> None:
    """Run the full pipeline end-to-end."""
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    # Store globally so helper functions can access params
    global params_global
    params_global = load_config(str(config_path))
    params = params_global

    apply_style()

    # Create output directories
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Resolve dataset list
    ds_arg = args.dataset
    if ds_arg == "all":
        datasets_to_run = [d["name"] for d in params.get("datasets", [{"name": "welllog"}])]
    else:
        datasets_to_run = [ds_arg]

    # Build per-dataset config map (for expected_n_changepoints overrides)
    ds_config_map: dict[str, dict] = {
        d["name"]: d for d in params.get("datasets", [])
    }

    all_timings: dict[str, dict] = {}

    for ds_name in datasets_to_run:
        logger.info("Loading %s data...", ds_name)
        ds_cfg = ds_config_map.get(ds_name, {})

        # Override expected_n_changepoints per dataset if configured
        if "expected_n_changepoints" in ds_cfg:
            params["evaluation"]["expected_n_changepoints"] = ds_cfg["expected_n_changepoints"]
            logger.info("expected_n_changepoints overridden to %d for %s",
                        ds_cfg["expected_n_changepoints"], ds_name)

        is_financial = False
        ds_data_dict = None
        if ds_name == "welllog":
            y_train, y_test, true_cps, true_outliers = load_welllog_data(
                train_fraction=params["splitting"]["welllog_train_fraction"],
                cache_path="data/raw/welllog.csv",
            )
        elif ds_name == "oilwell":
            oilwell_data = load_oilwell_data(
                train_fraction=ds_cfg.get("train_fraction", 0.75)
            )
            y_train = oilwell_data["y_train"]
            y_test = oilwell_data["y_test"]
            true_cps = np.concatenate([
                oilwell_data["true_cps_train"],
                oilwell_data["true_cps_test"] + oilwell_data["split_index"],
            ])
            true_outliers = None
        elif ds_name == "brent_crude":
            brent_data = load_brent_crude(
                train_end_date=params.get("brent_train_end", "2024-12-31")
            )
            y_train = brent_data["y_train"]
            y_test = brent_data["y_test"]
            true_cps = np.array([], dtype=int)
            true_outliers = None
            is_financial = True
            ds_data_dict = brent_data
        elif ds_name == "us_ip_growth" or ds_cfg.get("type") == "fred":
            usip_data = load_us_ip_growth(
                series_id=ds_cfg.get("fred_series_id", "INDPRO"),
                start_date=ds_cfg.get("start_date", "2000-01-01"),
                end_date=ds_cfg.get("end_date", "2026-01-01"),
                train_end_date=ds_cfg.get("train_end_date", "2023-12-01"),
            )
            y_train = usip_data["y_train"]
            y_test = usip_data["y_test"]
            # Combine train + test CPs into absolute indices
            true_cps = np.concatenate([
                usip_data["true_cps_train"],
                usip_data["true_cps_test"] + usip_data["split_index"],
            ])
            true_outliers = None
            is_financial = False
            ds_data_dict = usip_data
        elif ds_cfg.get("type") == "tcpd":
            # TCPD series
            train_frac = ds_cfg.get("train_fraction", 0.75)
            tcpd_data = load_tcpd_for_pipeline(ds_name, train_fraction=train_frac)
            y_train = tcpd_data["y_train"]
            y_test = tcpd_data["y_test"]
            true_cps = tcpd_data["metadata"]["consensus_cps"]
            true_outliers = None
        else:
            # Unknown dataset — attempt TCPD
            train_frac = ds_cfg.get("train_fraction", 0.75)
            tcpd_data = load_tcpd_for_pipeline(ds_name, train_fraction=train_frac)
            y_train = tcpd_data["y_train"]
            y_test = tcpd_data["y_test"]
            true_cps = tcpd_data["metadata"]["consensus_cps"]
            true_outliers = None

        # Fix A: verify well-log 75/25 split
        if ds_name == "welllog":
            expected_train = int(4050 * 0.75)
            assert len(y_train) == expected_train, (
                f"Well-log train size wrong: got {len(y_train)}, expected {expected_train}"
            )
            logger.info(
                "Well-log train size verified: %d (%.0f%% of 4050)",
                len(y_train), 100 * len(y_train) / 4050,
            )

        t = run_dataset(
            name=ds_name,
            y_train=y_train,
            y_test=y_test,
            true_cps=true_cps,
            true_outliers=true_outliers,
            params=params,
            output_dir=output_dir,
            skip_baselines=args.skip_baselines,
            is_financial=is_financial,
            data_dict=ds_data_dict,
        )
        all_timings[ds_name] = t

    # Consolidated runtime table
    runtime_df = pd.DataFrame(all_timings).T
    runtime_df.to_csv(output_dir / "tables" / "runtime.csv")
    logger.info("Runtime table saved to %s/tables/runtime.csv", output_dir)

    logger.info("Pipeline complete.  All results in %s/", output_dir)


if __name__ == "__main__":
    main()
