"""Time-series run charts with changepoint overlays."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging_config import setup_logger
from src.visualization.style import PALETTE

logger = setup_logger(__name__)


def plot_run_chart(
    y: np.ndarray,
    detected_cps: np.ndarray,
    true_cps: np.ndarray | None,
    means: np.ndarray | None,
    title: str = "",
    save_path: str | Path | None = None,
    labels: np.ndarray | None = None,
    dpi: int = 300,
) -> None:
    """Plot a time series with detected and true changepoints.

    Parameters
    ----------
    y:
        Observed time series.
    detected_cps:
        Indices of detected changepoints in this series' coordinates.
    true_cps:
        Indices of true changepoints — MUST be in this series' coordinates.
        Only CPs within [0, n) are plotted.  Pass ``None`` to omit.
    means:
        Piecewise-constant mean estimate — drawn as a red step overlay.
        Pass ``None`` to omit.
    title:
        Figure title.
    save_path:
        If provided, the figure is saved at ``dpi`` DPI (format inferred from
        the file extension).
    labels:
        Optional array same length as detected_cps.
        Supports both binary int labels and 4-class string labels:
        - 1 / "Sustained"        → green solid
        - 0 / "Recoiled"         → orange dashed
        - "Abrupt"               → red solid
        - "Abrupt-Preceded"      → purple dashed
        When provided, detected CPs are coloured accordingly.
    dpi:
        Output resolution (default 300).
    """
    n = len(y)
    fig, ax = plt.subplots(figsize=(14, 5))

    # Data
    ax.plot(range(n), y, color='#888888', linewidth=0.4, alpha=0.8,
            label='Observed')

    # Mean estimate
    if means is not None and len(means) == n:
        ax.plot(range(n), means, color='#D32F2F', linewidth=1.5,
                label='Mean estimate')

    # True CPs — ONLY those within [0, n)
    if true_cps is not None and len(true_cps) > 0:
        true_cps_valid = [cp for cp in true_cps if 0 <= cp < n]
        for i, cp in enumerate(true_cps_valid):
            ax.axvline(cp, color='#1976D2', linewidth=1.0, alpha=0.6,
                       label='True CP' if i == 0 else None)

    # Detected CPs — with 4-class label colouring
    if labels is not None and len(labels) == len(detected_cps):
        _seen_lbls: set = set()

        def _cp_style(raw_lbl):
            """Map raw label (int or str) → (color, linestyle, legend_label)."""
            if isinstance(raw_lbl, str):
                lbl_str = raw_lbl
            else:
                lbl_str = "Sustained" if int(raw_lbl) == 1 else "Recoiled"
            mapping = {
                "Sustained":      ('#2E7D32', '-',  'Sustained'),
                "Recoiled":       ('#E65100', '--', 'Recoiled'),
                "Abrupt":         ('#C62828', '-',  'Abrupt'),
                "Abrupt-Preceded":('#6A1B9A', '--', 'Abrupt-Preceded'),
            }
            return mapping.get(lbl_str, ('#555555', ':', lbl_str))

        for i, cp in enumerate(detected_cps):
            if 0 <= cp < n:
                color, style, leg_lbl = _cp_style(labels[i])
                show_leg = leg_lbl not in _seen_lbls
                if show_leg:
                    _seen_lbls.add(leg_lbl)
                ax.axvline(cp, color=color, linestyle=style, linewidth=1.2,
                           alpha=0.8, label=leg_lbl if show_leg else None)
    else:
        for i, cp in enumerate(detected_cps):
            if 0 <= cp < n:
                ax.axvline(cp, color='black', linestyle='--', linewidth=0.8,
                           alpha=0.7, label='Detected CP' if i == 0 else None)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, n)

    # Deduplicate legend
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc='upper right')

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info("Figure saved: %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_changepoint_comparison(
    y: np.ndarray,
    detectors_dict: dict[str, np.ndarray],
    true_cps: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Grid of subplots comparing multiple detectors on the same series.

    Each subplot shows the full series with true changepoints (solid blue)
    and one detector's detected changepoints (dashed black).

    Parameters
    ----------
    y:
        Observed time series.
    detectors_dict:
        ``{detector_name: detected_changepoints_array}``
    true_cps:
        Ground-truth changepoint indices.
    save_path:
        If provided, the figure is saved at 300 DPI.
    """
    n_det = len(detectors_dict)
    n_cols = min(n_det, 2)
    n_rows = math.ceil(n_det / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    t = np.arange(len(y))

    for ax_idx, (name, cps) in enumerate(detectors_dict.items()):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]

        ax.plot(t, y, color=PALETTE["series"], lw=0.7, alpha=0.85)

        n = len(y)
        true_cps_valid = [cp for cp in true_cps if 0 <= cp < n]
        for i, cp in enumerate(true_cps_valid):
            ax.axvline(cp, color=PALETTE["true_cp"], lw=1.0, ls="-", alpha=0.7,
                       label="true CP" if i == 0 else None)
        for i, cp in enumerate(np.asarray(cps)):
            if 0 <= cp < n:
                ax.axvline(cp, color=PALETTE["detected_cp"], lw=0.9, ls="--", alpha=0.6,
                           label="detected" if i == 0 else None)

        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
        ax.set_xlabel("Time")

    # Hide any unused subplots
    for ax_idx in range(n_det, n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Changepoint detector comparison", fontsize=12)
    _save_or_show(fig, save_path)


def plot_financial_analysis(
    prices: np.ndarray,
    dates,
    detected_cps_train: np.ndarray,
    detected_cps_test: np.ndarray,
    train_end_idx: int,
    labels_train: np.ndarray | None,
    labels_test: np.ndarray | None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """Two-panel financial analysis figure.

    Top panel: price series with blue shading for training period and
    vertical lines for detected changepoints coloured by label
    (green = sustained, red = recoiled, grey = unlabelled).

    Bottom panel: log-returns bar chart with a zero baseline.

    Parameters
    ----------
    prices:
        Full price series (train + test concatenated).
    dates:
        DatetimeIndex or array of dates corresponding to ``prices``.
    detected_cps_train:
        Detected changepoint indices (train-relative).
    detected_cps_test:
        Detected changepoint indices (test-relative).
    train_end_idx:
        Index in ``prices`` where the training period ends.
    labels_train:
        Binary labels for train changepoints (1=sustained, 0=recoiled).
        Pass ``None`` to draw all lines grey.
    labels_test:
        Binary labels for test changepoints.
    save_path:
        If provided, saved at ``dpi`` DPI (format from extension).
    dpi:
        Output DPI (default 300).
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    log_returns = np.concatenate([[0.0], np.diff(np.log(prices))])

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax_price, ax_ret = axes

    # ---- Price panel ----
    ax_price.plot(dates, prices, color=PALETTE["series"], lw=0.8, label="Price")
    ax_price.axvspan(dates[0], dates[min(train_end_idx, n - 1)],
                     alpha=0.08, color="steelblue", label="Training period")

    def _cp_color(label):
        if label is None:
            return "grey"
        return PALETTE.get("true_cp", "green") if label == 1 else PALETTE.get("detected_cp", "red")

    # Train CPs
    for j, cp in enumerate(np.asarray(detected_cps_train)):
        abs_idx = int(cp)
        if 0 <= abs_idx < n:
            lbl = int(labels_train[j]) if labels_train is not None else None
            color = _cp_color(lbl)
            ax_price.axvline(
                dates[abs_idx], color=color, lw=0.9, ls="--", alpha=0.7,
                label=("CP (train)" if j == 0 else None),
            )

    # Test CPs
    for j, cp in enumerate(np.asarray(detected_cps_test)):
        abs_idx = int(train_end_idx + cp)
        if 0 <= abs_idx < n:
            lbl = int(labels_test[j]) if labels_test is not None else None
            color = _cp_color(lbl)
            ax_price.axvline(
                dates[abs_idx], color=color, lw=0.9, ls="-.", alpha=0.7,
                label=("CP (test)" if j == 0 else None),
            )

    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize=8, framealpha=0.7)

    # ---- Log-returns panel ----
    ax_ret.bar(dates, log_returns, color=PALETTE["series"], alpha=0.5, width=1)
    ax_ret.axhline(0, color="black", lw=0.6)
    ax_ret.set_ylabel("Log-return")
    ax_ret.set_xlabel("Date")

    fig.suptitle("Financial series — changepoint analysis", fontsize=12)
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
