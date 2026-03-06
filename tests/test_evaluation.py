"""Tests for MRL index, Hausdorff distance, sensitivity analysis,
and visualization smoke tests."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.mrl_index import (
    compute_mrl,
    compute_risk,
    compute_censored_mrl,
    compute_censored_risk,
)
from src.evaluation.hausdorff import directed_hausdorff, symmetric_hausdorff
from src.evaluation.sensitivity import cost_ratio_sensitivity
from src.visualization.run_charts import plot_run_chart, plot_changepoint_comparison
from src.visualization.roc_curves import plot_roc_curves
from src.visualization.sensitivity_heatmap import plot_sensitivity_heatmap


# ---------------------------------------------------------------------------
# MRL index
# ---------------------------------------------------------------------------

class TestComputeMrl:
    def test_perfect_detection(self):
        r = compute_mrl(np.array([100]), true_cp=100)
        assert r["FP"] == 0
        assert r["MRL"] == pytest.approx(0.0)
        assert r["T_first"] == pytest.approx(100.0)

    def test_delayed_detection(self):
        r = compute_mrl(np.array([115]), true_cp=100)
        assert r["FP"] == 0
        assert r["MRL"] == pytest.approx(15.0)

    def test_false_positive_before_cp(self):
        r = compute_mrl(np.array([50, 80, 100]), true_cp=100)
        assert r["FP"] == 2
        assert r["MRL"] == pytest.approx(0.0)

    def test_missed_detection(self):
        r = compute_mrl(np.array([50, 80]), true_cp=100)
        assert r["FP"] == 2
        assert r["MRL"] == np.inf
        assert r["T_first"] == np.inf

    def test_empty_detections(self):
        r = compute_mrl(np.array([]), true_cp=100)
        assert r["FP"] == 0
        assert r["MRL"] == np.inf

    def test_tolerance_window(self):
        # Detection at 103 is within tolerance=5 of true_cp=100
        r = compute_mrl(np.array([103]), true_cp=100, tolerance=5)
        assert r["FP"] == 0
        assert r["MRL"] == pytest.approx(3.0)

    def test_tolerance_excludes_fp(self):
        # Detection at 90 is outside tolerance=5, so it's a FP
        r = compute_mrl(np.array([90, 105]), true_cp=100, tolerance=5)
        assert r["FP"] == 1  # 90 < 95 = 100-5
        assert r["MRL"] == pytest.approx(5.0)

    def test_fp_count_only_before_cp(self):
        # Detections after true_cp but far from it (no tolerance) → not FP
        r = compute_mrl(np.array([100, 150, 200]), true_cp=100)
        assert r["FP"] == 0
        assert r["MRL"] == pytest.approx(0.0)

    def test_returns_keys(self):
        r = compute_mrl(np.array([100]), true_cp=100)
        assert set(r.keys()) == {"FP", "MRL", "T_first"}


class TestComputeRisk:
    def test_basic(self):
        R = compute_risk(FP=2, MRL=10.0, cF=1.0, cD=1.0)
        assert R == pytest.approx(2.0 / 10.0)

    def test_zero_mrl_gives_inf(self):
        assert compute_risk(0, 0.0, 1, 1) == np.inf

    def test_inf_mrl_gives_inf(self):
        assert compute_risk(0, np.inf, 1, 1) == np.inf

    def test_zero_fp(self):
        R = compute_risk(FP=0, MRL=10.0, cF=1.0, cD=1.0)
        assert R == pytest.approx(0.0)

    def test_cost_scaling(self):
        R1 = compute_risk(1, 5.0, cF=2.0, cD=1.0)
        R2 = compute_risk(1, 5.0, cF=4.0, cD=1.0)
        assert R2 == pytest.approx(2 * R1)


class TestComputeCensoredMrl:
    def test_zero_maps_to_epsilon(self):
        assert compute_censored_mrl(0.0, epsilon=1.0, Tmax=100.0) == pytest.approx(1.0)

    def test_inf_maps_to_tmax(self):
        assert compute_censored_mrl(np.inf, epsilon=1.0, Tmax=100.0) == pytest.approx(100.0)

    def test_normal_value_unchanged(self):
        assert compute_censored_mrl(15.0, epsilon=1.0, Tmax=100.0) == pytest.approx(15.0)

    def test_nan_maps_to_tmax(self):
        assert compute_censored_mrl(float("nan"), epsilon=1.0, Tmax=100.0) == pytest.approx(100.0)


class TestComputeCensoredRisk:
    def test_basic(self):
        R = compute_censored_risk(FP=2, MRL=10.0, cF=1, cD=1, epsilon=1, Tmax=100)
        assert R == pytest.approx(2.0 / 10.0)

    def test_missed_detection_uses_tmax(self):
        R = compute_censored_risk(FP=0, MRL=np.inf, cF=1, cD=1, epsilon=1, Tmax=50)
        assert R == pytest.approx(0.0 / 50.0)

    def test_zero_mrl_uses_epsilon(self):
        R = compute_censored_risk(FP=1, MRL=0.0, cF=1, cD=1, epsilon=2.0, Tmax=100)
        assert R == pytest.approx(1.0 / 2.0)


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------

class TestDirectedHausdorff:
    def test_identical_sets(self):
        A = np.array([10, 50, 100])
        assert directed_hausdorff(A, A) == pytest.approx(0.0)

    def test_empty_set_returns_zero(self):
        assert directed_hausdorff(np.array([]), np.array([10, 20])) == 0.0
        assert directed_hausdorff(np.array([10, 20]), np.array([])) == 0.0

    def test_single_element(self):
        assert directed_hausdorff(np.array([10.0]), np.array([15.0])) == pytest.approx(5.0)

    def test_asymmetry(self):
        A = np.array([0.0, 10.0])
        B = np.array([0.0, 5.0, 10.0])
        # d(A→B): max(min(|0-0|,|0-5|,|0-10|), min(|10-0|,|10-5|,|10-10|)) = max(0,0) = 0
        assert directed_hausdorff(A, B) == pytest.approx(0.0)
        # d(B→A): max(min(|0-0|,|0-10|), min(|5-0|,|5-10|), min(|10-0|,|10-10|)) = max(0,5,0) = 5
        assert directed_hausdorff(B, A) == pytest.approx(5.0)

    def test_known_value(self):
        A = np.array([0.0, 20.0])
        B = np.array([5.0, 15.0])
        # d(A→B): max(min(5,15), min(15,5)) = max(5,5) = 5
        assert directed_hausdorff(A, B) == pytest.approx(5.0)


class TestSymmetricHausdorff:
    def test_symmetric(self):
        A = np.array([0.0, 10.0])
        B = np.array([3.0, 13.0])
        h = symmetric_hausdorff(A, B)
        assert h == pytest.approx(max(directed_hausdorff(A, B), directed_hausdorff(B, A)))

    def test_nonneg(self):
        assert symmetric_hausdorff(np.array([1.0]), np.array([4.0])) >= 0.0

    def test_empty_returns_zero(self):
        assert symmetric_hausdorff(np.array([]), np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class TestCostRatioSensitivity:
    def _run(self):
        detectors = {
            "Good": {"FP": 0, "MRL": 2.0},
            "Bad":  {"FP": 3, "MRL": 10.0},
        }
        return cost_ratio_sensitivity(
            detectors,
            cF_grid=[1, 2],
            cD_grid=[1, 5],
            epsilon=0.5,
            Tmax=100.0,
        )

    def test_returns_two_dataframes(self):
        ranks, raw = self._run()
        assert isinstance(ranks, pd.DataFrame)
        assert isinstance(raw, pd.DataFrame)

    def test_index_is_multiindex(self):
        ranks, _ = self._run()
        assert isinstance(ranks.index, pd.MultiIndex)

    def test_row_count(self):
        ranks, _ = self._run()
        assert len(ranks) == 2 * 2  # |cF_grid| * |cD_grid|

    def test_columns_are_detector_names(self):
        ranks, _ = self._run()
        assert set(ranks.columns) == {"Good", "Bad"}

    def test_best_rank_is_one(self):
        ranks, _ = self._run()
        assert (ranks.min(axis=1) == 1).all()

    def test_better_detector_ranks_first(self):
        """'Good' (low FP, low MRL) should consistently rank 1."""
        ranks, _ = self._run()
        assert (ranks["Good"] == 1).all(), "Expected 'Good' to always rank 1"

    def test_raw_values_nonneg(self):
        _, raw = self._run()
        assert (raw >= 0).all().all()

    def test_single_detector(self):
        ranks, raw = cost_ratio_sensitivity(
            {"Only": {"FP": 1, "MRL": 5.0}},
            cF_grid=[1], cD_grid=[1],
            epsilon=1, Tmax=100,
        )
        assert ranks.loc[(1, 1), "Only"] == 1


# ---------------------------------------------------------------------------
# Visualization smoke tests (Agg backend — no display needed)
# ---------------------------------------------------------------------------

class TestVisualizationSmoke:
    def _data(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        means = np.zeros(200)
        means[100:] = 2.0
        return y, means

    def test_plot_run_chart_no_error(self, tmp_path):
        y, means = self._data()
        plot_run_chart(
            y, detected_cps=np.array([100]),
            true_cps=np.array([100]), means=means,
            title="test", save_path=tmp_path / "run.png",
            outlier_indices=np.array([50, 150]),
        )
        assert (tmp_path / "run.png").exists()

    def test_plot_run_chart_none_optionals(self, tmp_path):
        y, _ = self._data()
        plot_run_chart(y, np.array([]), true_cps=None, means=None,
                       save_path=tmp_path / "run2.png")
        assert (tmp_path / "run2.png").exists()

    def test_plot_changepoint_comparison(self, tmp_path):
        y, _ = self._data()
        plot_changepoint_comparison(
            y,
            {"Det A": np.array([100, 120]), "Det B": np.array([95])},
            true_cps=np.array([100]),
            save_path=tmp_path / "cmp.png",
        )
        assert (tmp_path / "cmp.png").exists()

    def test_plot_roc_curves(self, tmp_path):
        rng = np.random.default_rng(1)
        y_true = np.array([0] * 50 + [1] * 50)
        results = {
            "Clf A": {"y_true": y_true, "y_proba": rng.uniform(size=100)},
            "Clf B": {"y_true": y_true, "y_proba": rng.uniform(size=100)},
        }
        plot_roc_curves(results, save_path=tmp_path / "roc.png")
        assert (tmp_path / "roc.png").exists()

    def test_plot_sensitivity_heatmap(self, tmp_path):
        detectors = {"A": {"FP": 0, "MRL": 2.0}, "B": {"FP": 2, "MRL": 8.0}}
        ranks, raw = cost_ratio_sensitivity(
            detectors, [1, 2], [1, 3], epsilon=1, Tmax=50
        )
        plot_sensitivity_heatmap(ranks, raw, save_path=tmp_path / "heatmap.png")
        assert (tmp_path / "heatmap_ranks.png").exists()
        assert (tmp_path / "heatmap_values.png").exists()
