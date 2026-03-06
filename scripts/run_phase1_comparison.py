"""Phase I detector comparison script.

Compares three penalty variants of EV-DeCAFS on both datasets:

- **GPD penalty** (EV-DeCAFS, full method)
- **Flat penalty** (standard DeCAFS with constant alpha_0)
- **Exceedance-count penalty** (count-based EVT approximation)

Produces:
- MRL comparison table  (``results/tables/{dataset}_phase1_mrl.csv``)
- Hausdorff comparison  (``results/tables/{dataset}_phase1_hausdorff.csv``)
- Sensitivity heatmaps  (``results/figures/{dataset}_phase1_sensitivity.*``)
- Run-chart comparison  (``results/figures/{dataset}_phase1_comparison.*``)

Run with::

    python scripts/run_phase1_comparison.py --dataset both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_bitcoin_data, load_welllog_data
from src.evaluation.hausdorff import symmetric_hausdorff
from src.evaluation.mrl_index import compute_mrl
from src.evaluation.sensitivity import cost_ratio_sensitivity
from src.phase1.ar1_model import estimate_ar1_params
from src.phase1.decafs import ev_decafs
from src.phase1.evt_penalty import (
    compute_adaptive_penalty,
    compute_evi_field,
    compute_exceedance_count_penalty,
)
from src.utils.logging_config import setup_logger
from src.visualization.run_charts import plot_changepoint_comparison
from src.visualization.sensitivity_heatmap import plot_sensitivity_heatmap
from src.visualization.style import apply_style

logger = setup_logger("phase1_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase I changepoint detectors."
    )
    parser.add_argument("--config", type=str, default="config/params.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bitcoin", "welllog", "both"],
        default="both",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def _build_detectors(y: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    """Fit all three Phase I detector variants. Returns {name: changepoints}."""
    p1 = params["phase1"]

    ar1 = estimate_ar1_params(y)
    lambda_param = 1.0 / (ar1["sigma_eta_sq"] + 1e-12)
    gamma = 1.0 / (ar1["sigma_v_sq"] + 1e-12)
    phi = ar1["phi"]

    # GPD penalty
    t0 = time.perf_counter()
    xi = compute_evi_field(y, w=p1["window_halfwidth_w"], q0=p1["gpd_percentile_q0"])
    alpha_gpd = compute_adaptive_penalty(xi, p1["alpha_0"], p1["evt_sensitivity_lambda_ev"])
    res_gpd = ev_decafs(y, alpha_gpd, lambda_param, gamma, phi)
    logger.info("GPD:  %d changepoints (%.1fs)", len(res_gpd["changepoints"]),
                time.perf_counter() - t0)

    # Flat penalty
    t0 = time.perf_counter()
    alpha_flat = np.full(len(y), p1["alpha_0"])
    res_flat = ev_decafs(y, alpha_flat, lambda_param, gamma, phi)
    logger.info("Flat: %d changepoints (%.1fs)", len(res_flat["changepoints"]),
                time.perf_counter() - t0)

    # Exceedance-count penalty
    t0 = time.perf_counter()
    mu_proxy = np.full_like(y, np.mean(y))
    alpha_ec = compute_exceedance_count_penalty(
        y, mu_proxy,
        sigma_v=float(np.sqrt(ar1["sigma_v_sq"])),
        w=p1["window_halfwidth_w"],
        c=p1["exceedance_multiplier_c"],
        alpha_0=p1["alpha_0"],
    )
    res_ec = ev_decafs(y, alpha_ec, lambda_param, gamma, phi)
    logger.info("EC:   %d changepoints (%.1fs)", len(res_ec["changepoints"]),
                time.perf_counter() - t0)

    return {
        "EV-DeCAFS (GPD)": res_gpd["changepoints"],
        "DeCAFS (flat)":   res_flat["changepoints"],
        "EV-DeCAFS (EC)":  res_ec["changepoints"],
    }


def _evaluate(
    detectors: dict[str, np.ndarray],
    true_cps: np.ndarray | None,
    params: dict,
    n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MRL + Hausdorff for each detector. Returns (mrl_df, hausdorff_df)."""
    ep = params["evaluation"]
    Tmax = ep["censoring_Tmax_fraction"] * n
    epsilon = ep["censoring_epsilon"]

    mrl_rows = {}
    h_rows = {}

    for name, cps in detectors.items():
        if true_cps is not None and len(true_cps) > 0:
            fps, mrls = [], []
            for tc in true_cps:
                r = compute_mrl(cps, tc)
                fps.append(r["FP"])
                mrls.append(r["MRL"] if np.isfinite(r["MRL"]) else Tmax)
            mrl_rows[name] = {
                "FP (total)": sum(fps),
                "MRL (mean)": float(np.mean(mrls)),
                "Detection rate": sum(1 for m in mrls if m < Tmax) / len(mrls),
            }
            h_rows[name] = {"Hausdorff": symmetric_hausdorff(cps, true_cps)}
        else:
            mrl_rows[name] = {"FP (total)": len(cps), "MRL (mean)": float("nan"),
                              "Detection rate": float("nan")}
            h_rows[name] = {"Hausdorff": float("nan")}

    return pd.DataFrame(mrl_rows).T, pd.DataFrame(h_rows).T


def run_comparison(
    name: str,
    y_train: np.ndarray,
    true_cps: np.ndarray | None,
    params: dict,
    output_dir: Path,
    fig_fmt: str,
) -> None:
    logger.info("=== Phase I comparison — %s (n=%d) ===", name, len(y_train))

    detectors = _build_detectors(y_train, params)

    # Comparison run chart
    plot_changepoint_comparison(
        y_train, detectors,
        true_cps=true_cps if true_cps is not None else np.array([]),
        save_path=output_dir / "figures" / f"{name}_phase1_comparison.{fig_fmt}",
    )

    # MRL + Hausdorff tables
    mrl_df, h_df = _evaluate(detectors, true_cps, params, len(y_train))
    mrl_df.to_csv(output_dir / "tables" / f"{name}_phase1_mrl.csv")
    h_df.to_csv(output_dir / "tables" / f"{name}_phase1_hausdorff.csv")
    logger.info("\nMRL summary:\n%s", mrl_df.to_string())
    logger.info("\nHausdorff:\n%s", h_df.to_string())

    # Sensitivity heatmap (only when we have ground-truth)
    if true_cps is not None and len(true_cps) > 0:
        ep = params["evaluation"]
        n = len(y_train)
        Tmax = ep["censoring_Tmax_fraction"] * n
        eps = ep["censoring_epsilon"]
        det_summary = {
            nm: {
                "FP": int(mrl_df.loc[nm, "FP (total)"]),
                "MRL": float(mrl_df.loc[nm, "MRL (mean)"]),
            }
            for nm in detectors
        }
        rankings, raw = cost_ratio_sensitivity(
            det_summary,
            cF_grid=ep["cost_cF_grid"],
            cD_grid=ep["cost_cD_grid"],
            epsilon=eps,
            Tmax=Tmax,
        )
        plot_sensitivity_heatmap(
            rankings, raw,
            save_path=output_dir / "figures" / f"{name}_phase1_sensitivity.{fig_fmt}",
        )
        rankings.to_csv(output_dir / "tables" / f"{name}_phase1_sensitivity_ranks.csv")
        raw.to_csv(output_dir / "tables" / f"{name}_phase1_sensitivity_rtilde.csv")


def main() -> None:
    args = parse_args()
    params = yaml.safe_load(open(args.config))
    output_dir = Path(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    apply_style()

    fig_fmt = params["visualization"].get("figure_format", "pdf")
    datasets = ["bitcoin", "welllog"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        if ds == "bitcoin":
            y_train, _, _, _ = load_bitcoin_data(
                train_end_date=params["splitting"]["bitcoin_train_end"],
                cache_path="data/raw/btc_usd.csv",
            )
            true_cps = None
        else:
            y_train, _, true_cps, _ = load_welllog_data(
                train_fraction=params["splitting"]["welllog_train_fraction"],
                cache_path="data/raw/welllog.csv",
            )

        run_comparison(ds, y_train, true_cps, params, output_dir, fig_fmt)

    logger.info("Phase I comparison complete.")


if __name__ == "__main__":
    main()
