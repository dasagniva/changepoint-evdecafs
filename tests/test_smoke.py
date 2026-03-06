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
