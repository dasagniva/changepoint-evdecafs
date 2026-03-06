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

from src.data.loader import load_bitcoin_data, load_welllog_data
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hausdorff import symmetric_hausdorff
from src.evaluation.mrl_index import compute_censored_risk, compute_mrl
from src.evaluation.sensitivity import cost_ratio_sensitivity
from src.phase1.ar1_model import estimate_ar1_params
from src.phase1.decafs import ev_decafs
from src.phase1.evt_penalty import (
    compute_adaptive_penalty,
    compute_evi_field,
    compute_exceedance_count_penalty,
)
from src.phase1.feature_extract import extract_features
from src.phase2.baselines import get_baselines, train_and_evaluate_all
from src.phase2.fpnn import FourierPNN
from src.phase2.labelling import compute_kappa_mu, label_changepoints
from src.phase2.smote_balance import balance_training_data
from src.utils.logging_config import setup_logger
from src.visualization.roc_curves import plot_roc_curves
from src.visualization.run_charts import plot_changepoint_comparison, plot_run_chart
from src.visualization.sensitivity_heatmap import plot_sensitivity_heatmap
from src.visualization.style import apply_style

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
        choices=["bitcoin", "welllog", "both"],
        default="both",
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
) -> dict:
    """Estimate AR(1) params, build adaptive penalty, run EV-DeCAFS.

    Returns a dict with keys: ar1, xi_field, alpha_gpd, alpha_ec,
    decafs_result, elapsed_ar1, elapsed_evi, elapsed_decafs.
    """
    p1 = params["phase1"]

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
        xi_field, alpha_0=p1["alpha_0"], lambda_ev=p1["evt_sensitivity_lambda_ev"]
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
        alpha_0=p1["alpha_0"],
    )

    # --- EV-DeCAFS ---
    t0 = time.perf_counter()
    decafs_result = ev_decafs(
        y_train,
        alpha_t=alpha_gpd,
        lambda_param=lambda_param,
        gamma=gamma,
        phi=ar1["phi"],
    )
    elapsed_decafs = time.perf_counter() - t0
    logger.info(
        "EV-DeCAFS: %d changepoints, cost=%.4f (%.2fs)",
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
    )


# =============================================================================
# Phase II helpers
# =============================================================================

def run_phase2_train(
    y_train: np.ndarray,
    phase1: dict,
    params: dict,
) -> dict:
    """Extract features, label, SMOTE-balance, and fit FPNN.

    Returns a dict with keys: X_train, labels_train, X_balanced,
    y_balanced, kappa_mu, fpnn, elapsed_features, elapsed_smote,
    elapsed_fpnn.
    """
    lp = params["labelling"]
    sp = params["smote"]
    fp = params["fpnn"]

    cps = phase1["decafs_result"]["changepoints"]
    means = phase1["decafs_result"]["means"]

    if len(cps) == 0:
        logger.warning("No training changepoints detected — skipping Phase II.")
        return {}

    # --- Feature extraction ---
    t0 = time.perf_counter()
    X_train, _ = extract_features(y_train, cps, means, L=lp["window_L"])
    elapsed_features = time.perf_counter() - t0

    # --- Labelling ---
    kappa_mu = compute_kappa_mu(X_train, percentile=lp["kappa_mu_percentile"])
    labels_train = label_changepoints(X_train, kappa_mu, kappa_S=lp["kappa_S"])
    logger.info(
        "Labels: %d sustained, %d recoiled (kappa_mu=%.4f)",
        labels_train.sum(), (labels_train == 0).sum(), kappa_mu,
    )

    # --- SMOTE ---
    t0 = time.perf_counter()
    X_balanced, y_balanced = balance_training_data(
        X_train, labels_train,
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
        labels_train=labels_train,
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

    # --- EV-DeCAFS on test ---
    t0 = time.perf_counter()
    xi_test = compute_evi_field(
        y_test,
        w=params["phase1"]["window_halfwidth_w"],
        q0=params["phase1"]["gpd_percentile_q0"],
    )
    alpha_test = compute_adaptive_penalty(
        xi_test,
        alpha_0=params["phase1"]["alpha_0"],
        lambda_ev=params["phase1"]["evt_sensitivity_lambda_ev"],
    )
    res_test = ev_decafs(
        y_test,
        alpha_t=alpha_test,
        lambda_param=phase1_train["lambda_param"],
        gamma=phase1_train["gamma"],
        phi=phase1_train["ar1"]["phi"],
    )
    elapsed_test_decafs = time.perf_counter() - t0
    logger.info(
        "Test EV-DeCAFS: %d changepoints (%.2fs)",
        len(res_test["changepoints"]),
        elapsed_test_decafs,
    )

    cps_test = res_test["changepoints"]
    means_test = res_test["means"]

    if len(cps_test) == 0:
        logger.warning("No test changepoints detected — skipping test classification.")
        return {"decafs_result": res_test}

    # --- Feature extraction on test ---
    X_test, _ = extract_features(y_test, cps_test, means_test, L=lp["window_L"])

    # Auto-label test changepoints using training thresholds
    labels_test = label_changepoints(
        X_test,
        kappa_mu=phase2_train["kappa_mu"],
        kappa_S=lp["kappa_S"],
    )

    # --- Classification evaluation ---
    fpnn = phase2_train["fpnn"]
    t0 = time.perf_counter()
    if skip_baselines:
        y_pred = fpnn.predict(X_test)
        y_proba = fpnn.predict_proba(X_test)
        m = compute_classification_metrics(labels_test, y_pred, y_proba)
        clf_df = pd.DataFrame(
            [{
                "Balanced Accuracy": m["balanced_accuracy"],
                "MCC": m["mcc"],
                "F1 (class 0)": m["f1_class0"],
                "F1 (class 1)": m["f1_class1"],
                "AUC-ROC": m["auc_roc"],
            }],
            index=["FPNN"],
        )
        roc_results = {
            "FPNN": {"y_true": labels_test, "y_proba": y_proba[:, 1]},
        }
    else:
        baselines = get_baselines(params["baselines"])
        clf_df = train_and_evaluate_all(
            phase2_train["X_balanced"],
            phase2_train["y_balanced"],
            X_test, labels_test,
            baselines, fpnn,
        )
        # Collect proba for ROC curves (re-fit baselines for proba)
        roc_results = {"FPNN": {
            "y_true": labels_test,
            "y_proba": fpnn.predict_proba(X_test)[:, 1],
        }}
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
    true_cps: np.ndarray,
    params: dict,
) -> dict | None:
    """Compute MRL, Hausdorff, and cost-ratio sensitivity for the well-log."""
    if true_cps is None or len(true_cps) == 0:
        logger.info("No ground-truth changepoints — skipping MRL analysis.")
        return None

    ep = params["evaluation"]
    n_test = len(phase2_test.get("decafs_result", {}).get("means", [1]))
    Tmax = ep["censoring_Tmax_fraction"] * n_test
    epsilon = ep["censoring_epsilon"]

    detected = phase2_test["decafs_result"]["changepoints"]

    # One MRL per true changepoint (aggregate: use the closest true CP)
    all_mrl = {}
    for tc in true_cps:
        mrl = compute_mrl(detected, tc)
        all_mrl[tc] = mrl

    # Aggregate: mean FP, mean MRL across true changepoints
    total_fp = sum(r["FP"] for r in all_mrl.values())
    finite_mrls = [r["MRL"] for r in all_mrl.values() if np.isfinite(r["MRL"])]
    mean_mrl = float(np.mean(finite_mrls)) if finite_mrls else np.inf

    detectors_summary = {
        "EV-DeCAFS (GPD)": {"FP": total_fp, "MRL": mean_mrl},
    }

    # Hausdorff distance
    h = symmetric_hausdorff(detected, true_cps)
    logger.info("Hausdorff distance (EV-DeCAFS vs true): %.2f", h)

    # Sensitivity
    rankings, raw_rtilde = cost_ratio_sensitivity(
        detectors_summary,
        cF_grid=ep["cost_cF_grid"],
        cD_grid=ep["cost_cD_grid"],
        epsilon=epsilon,
        Tmax=Tmax,
    )

    return dict(
        all_mrl=all_mrl,
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
) -> dict:
    """Run the full pipeline for a single dataset. Returns timing dict."""
    logger.info("=" * 60)
    logger.info("Dataset: %s  (train=%d, test=%d)", name, len(y_train), len(y_test))
    logger.info("=" * 60)

    fig_dir = output_dir / "figures"
    tbl_dir = output_dir / "tables"
    fig_fmt = params["visualization"].get("figure_format", "pdf")

    timings: dict[str, float] = {}

    # ---- Phase I (training) ----
    t_phase1_start = time.perf_counter()
    phase1 = run_phase1(y_train, params)
    timings["phase1_total"] = time.perf_counter() - t_phase1_start

    # ---- Phase II training ----
    t_phase2_start = time.perf_counter()
    phase2_train = run_phase2_train(y_train, phase1, params)
    timings["phase2_train"] = time.perf_counter() - t_phase2_start

    if not phase2_train:
        logger.warning("[%s] Aborting: no training changepoints.", name)
        return timings

    # ---- Phase II test ----
    phase2_test = run_phase2_test(
        y_test, phase1, phase2_train, params, skip_baselines
    )
    timings["phase2_test"] = phase2_test.get("elapsed_clf", 0.0)

    # ---- MRL analysis (well-log only) ----
    mrl_analysis = run_mrl_analysis(phase1, phase2_test, true_cps, params)

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
        true_cps=true_cps,
        true_outliers=true_outliers,
        fig_dir=fig_dir,
        fig_fmt=fig_fmt,
    )

    return timings


def _make_figures(
    name, y_train, y_test, phase1, phase2_test,
    mrl_analysis, true_cps, true_outliers, fig_dir, fig_fmt,
):
    ext = fig_fmt

    # Training run chart
    plot_run_chart(
        y_train,
        detected_cps=phase1["decafs_result"]["changepoints"],
        true_cps=true_cps,
        means=phase1["decafs_result"]["means"],
        title=f"{name} — Training (EV-DeCAFS GPD)",
        save_path=fig_dir / f"{name}_train_run_chart.{ext}",
        outlier_indices=true_outliers,
    )

    # Test run chart
    if "decafs_result" in phase2_test:
        plot_run_chart(
            y_test,
            detected_cps=phase2_test["decafs_result"]["changepoints"],
            true_cps=true_cps,
            means=phase2_test["decafs_result"]["means"],
            title=f"{name} — Test (EV-DeCAFS GPD)",
            save_path=fig_dir / f"{name}_test_run_chart.{ext}",
        )

    # Detector comparison: GPD vs flat-penalty
    flat_alpha = np.full(len(y_train), params_global["phase1"]["alpha_0"])
    flat_result = ev_decafs(
        y_train, flat_alpha,
        lambda_param=phase1["lambda_param"],
        gamma=phase1["gamma"],
        phi=phase1["ar1"]["phi"],
    )
    plot_changepoint_comparison(
        y_train,
        detectors_dict={
            "EV-DeCAFS (GPD)": phase1["decafs_result"]["changepoints"],
            "EV-DeCAFS (flat)": flat_result["changepoints"],
        },
        true_cps=true_cps if true_cps is not None else np.array([]),
        save_path=fig_dir / f"{name}_detector_comparison.{ext}",
    )

    # ROC curves
    if "roc_results" in phase2_test and len(phase2_test["roc_results"]) > 0:
        plot_roc_curves(
            phase2_test["roc_results"],
            save_path=fig_dir / f"{name}_roc.{ext}",
        )

    # Sensitivity heatmap
    if mrl_analysis is not None:
        plot_sensitivity_heatmap(
            mrl_analysis["rankings"],
            mrl_analysis["raw_rtilde"],
            save_path=fig_dir / f"{name}_sensitivity.{ext}",
        )


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

    datasets_to_run = (
        ["bitcoin", "welllog"] if args.dataset == "both" else [args.dataset]
    )

    all_timings: dict[str, dict] = {}

    for ds_name in datasets_to_run:
        logger.info("Loading %s data...", ds_name)
        if ds_name == "bitcoin":
            y_train, y_test, _, _ = load_bitcoin_data(
                train_end_date=params["splitting"]["bitcoin_train_end"],
                cache_path="data/raw/btc_usd.csv",
            )
            true_cps = None
            true_outliers = None
        else:
            y_train, y_test, true_cps, true_outliers = load_welllog_data(
                train_fraction=params["splitting"]["welllog_train_fraction"],
                cache_path="data/raw/welllog.csv",
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
        )
        all_timings[ds_name] = t

    # Consolidated runtime table
    runtime_df = pd.DataFrame(all_timings).T
    runtime_df.to_csv(output_dir / "tables" / "runtime.csv")
    logger.info("Runtime table saved to %s/tables/runtime.csv", output_dir)

    logger.info("Pipeline complete.  All results in %s/", output_dir)


if __name__ == "__main__":
    main()
