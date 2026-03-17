"""Smoke tests — verify that all modules import without errors and that
public APIs have the expected signatures."""

import importlib
import inspect


MODULES_TO_CHECK = [
    "src.data.loader",
    "src.phase1.ar1_model",
    "src.phase1.evt_penalty",
    "src.phase1.decafs",
    "src.phase1.feature_extract",
    "src.phase1.hypersensitive_cpd",
    "src.phase2.labelling",
    "src.phase2.fpnn",
    "src.phase2.smote_balance",
    "src.phase2.baselines",
    "src.evaluation.classification_metrics",
    "src.evaluation.mrl_index",
    "src.evaluation.hausdorff",
    "src.evaluation.sensitivity",
    "src.visualization.run_charts",
    "src.visualization.roc_curves",
    "src.visualization.sensitivity_heatmap",
    "src.visualization.style",
    "src.utils.logging_config",
]


def test_all_modules_importable():
    """All project modules should import without raising exceptions."""
    for module_path in MODULES_TO_CHECK:
        mod = importlib.import_module(module_path)
        assert mod is not None, f"Failed to import {module_path}"


def test_logging_config_signature():
    from src.utils.logging_config import setup_logger
    sig = inspect.signature(setup_logger)
    assert "name" in sig.parameters
    assert "log_dir" in sig.parameters


def test_fpnn_class_exists():
    from src.phase2.fpnn import FourierPNN
    fpnn = FourierPNN(J=5)
    assert fpnn.J == 5
    assert hasattr(fpnn, "fit")
    assert hasattr(fpnn, "predict")
    assert hasattr(fpnn, "predict_proba")
    assert hasattr(fpnn, "get_coefficients")


def test_decafs_signature():
    from src.phase1.decafs import ev_decafs
    sig = inspect.signature(ev_decafs)
    for param in ["y", "alpha_t", "lambda_param", "gamma", "phi"]:
        assert param in sig.parameters, f"Missing param: {param}"


def test_mrl_index_signatures():
    from src.evaluation.mrl_index import (
        compute_mrl,
        compute_risk,
        compute_censored_mrl,
        compute_censored_risk,
    )
    for fn in [compute_mrl, compute_risk, compute_censored_mrl, compute_censored_risk]:
        assert callable(fn)


def test_style_module_applies_without_error():
    """apply_style() should not raise even without a display."""
    import matplotlib
    matplotlib.use("Agg")
    from src.visualization.style import apply_style
    apply_style()  # should not raise


def test_hypersensitive_relabelling_pipeline():
    """Smoke test: BOCPD + CUSUM flags feed into relabelling without error."""
    import numpy as np
    from src.phase1.hypersensitive_cpd import run_bocpd, run_cusum
    from src.phase2.labelling import relabel_with_hypersensitive

    rng = np.random.default_rng(42)
    n = 200
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    phi, sigma_v = 0.3, 1.0

    bocpd_flags = run_bocpd(y, phi=phi, sigma_v=sigma_v, threshold=0.5)
    cusum_flags = run_cusum(y, phi=phi, sigma_v=sigma_v, h_multiplier=1.0)

    assert bocpd_flags.shape == (n,)
    assert cusum_flags.shape == (n,)
    assert bocpd_flags.dtype == bool
    assert cusum_flags.dtype == bool

    cp_indices = np.array([100])
    existing = np.array([1])
    new_labels = relabel_with_hypersensitive(
        cp_indices=cp_indices,
        x_flags=bocpd_flags,
        true_cp_indices=np.array([100]),
        existing_labels=existing,
        tolerance=10,
    )
    assert len(new_labels) == 1
    assert new_labels[0] in {"Sustained", "Abrupt", "Abrupt-Preceded", "Recoiled"}
