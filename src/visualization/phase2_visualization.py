"""Phase II classification visualization: sustained vs recoiled changepoints."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_phase2_classification(y, detected_cps, means, labels, true_cps,
                                title, save_path, dpi=300):
    """The key figure: changepoints colored by sustained/recoiled classification.

    Args:
        y: observed time series (n,)
        detected_cps: array of detected CP indices
        means: piecewise-constant mean estimate (n,)
        labels: array of FPNN classifications (1=sustained, 0=recoiled)
                same length as detected_cps
        true_cps: array of ground-truth CP indices (for reference)
        title: figure title
        save_path: output path
    """
    n = len(y)
    fig, ax = plt.subplots(figsize=(15, 5.5))

    # Observed data
    ax.plot(range(n), y, color='#9E9E9E', linewidth=0.35, alpha=0.8,
            zorder=1)

    # Piecewise-constant mean
    if means is not None and len(means) == n:
        ax.plot(range(n), means, color='#D32F2F', linewidth=1.8,
                label='Estimated mean', zorder=3)

    # True CPs (background reference — thin, light blue)
    true_cps_valid = [cp for cp in true_cps if 0 <= cp < n]
    for i, cp in enumerate(true_cps_valid):
        ax.axvline(cp, color='#90CAF9', linewidth=0.8, alpha=0.5,
                   zorder=2, label='True CP' if i == 0 else None)

    # Detected CPs colored by classification
    n_sustained = 0
    n_recoiled = 0
    for i, cp in enumerate(detected_cps):
        if not (0 <= cp < n):
            continue
        if i < len(labels):
            if labels[i] == 1:
                ax.axvline(cp, color='#1B5E20', linestyle='-', linewidth=2.0,
                           alpha=0.85, zorder=4,
                           label='Sustained shift' if n_sustained == 0 else None)
                # Add a small marker at the top
                ax.plot(cp, ax.get_ylim()[1] * 0.98, marker='v', color='#1B5E20',
                        markersize=8, zorder=5, clip_on=False)
                n_sustained += 1
            else:
                ax.axvline(cp, color='#E65100', linestyle='--', linewidth=2.0,
                           alpha=0.85, zorder=4,
                           label='Recoiled (outlier)' if n_recoiled == 0 else None)
                ax.plot(cp, ax.get_ylim()[1] * 0.98, marker='x', color='#E65100',
                        markersize=8, zorder=5, clip_on=False)
                n_recoiled += 1

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, n)

    # Legend — clean, outside or inside upper area
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9,
              loc='lower right', framealpha=0.9,
              edgecolor='gray')

    # Add summary text
    summary = f'Detected: {len(detected_cps)} CPs ({n_sustained} sustained, {n_recoiled} recoiled)'
    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Phase II classification figure saved: {save_path}")


def plot_phase2_train_test(y_train, y_test, cps_train, cps_test,
                            means_train, means_test,
                            labels_train, labels_test,
                            true_cps_train, true_cps_test,
                            dataset_name, save_dir, dpi=300):
    """Two-panel figure: training (top) and test (bottom), both with
    Phase II classification coloring."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                     gridspec_kw={'height_ratios': [2, 1]})

    # --- Training panel ---
    n_tr = len(y_train)
    ax1.plot(range(n_tr), y_train, color='#9E9E9E', linewidth=0.35, alpha=0.8)
    if means_train is not None:
        ax1.plot(range(n_tr), means_train, color='#D32F2F', linewidth=1.5,
                 label='Mean estimate')

    true_in_train = [cp for cp in true_cps_train if 0 <= cp < n_tr]
    for i, cp in enumerate(true_in_train):
        ax1.axvline(cp, color='#90CAF9', linewidth=0.8, alpha=0.5,
                    label='True CP' if i == 0 else None)

    s_count, r_count = 0, 0
    for i, cp in enumerate(cps_train):
        if not (0 <= cp < n_tr) or i >= len(labels_train):
            continue
        if labels_train[i] == 1:
            ax1.axvline(cp, color='#1B5E20', linewidth=1.8, alpha=0.8,
                       label='Sustained' if s_count == 0 else None)
            s_count += 1
        else:
            ax1.axvline(cp, color='#E65100', linestyle='--', linewidth=1.8,
                       alpha=0.8, label='Recoiled' if r_count == 0 else None)
            r_count += 1

    ax1.set_title(f'{dataset_name} — Training (Phase I + II)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_xlim(0, n_tr)
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(),
               fontsize=8, loc='upper right')

    # --- Test panel ---
    n_te = len(y_test)
    ax2.plot(range(n_te), y_test, color='#9E9E9E', linewidth=0.35, alpha=0.8)
    if means_test is not None:
        ax2.plot(range(n_te), means_test, color='#D32F2F', linewidth=1.5,
                 label='Mean estimate')

    true_in_test = [cp for cp in true_cps_test if 0 <= cp < n_te]
    for i, cp in enumerate(true_in_test):
        ax2.axvline(cp, color='#90CAF9', linewidth=0.8, alpha=0.5,
                    label='True CP' if i == 0 else None)

    s_count, r_count = 0, 0
    for i, cp in enumerate(cps_test):
        if not (0 <= cp < n_te) or i >= len(labels_test):
            continue
        if labels_test[i] == 1:
            ax2.axvline(cp, color='#1B5E20', linewidth=1.8, alpha=0.8,
                       label='Sustained' if s_count == 0 else None)
            s_count += 1
        else:
            ax2.axvline(cp, color='#E65100', linestyle='--', linewidth=1.8,
                       alpha=0.8, label='Recoiled' if r_count == 0 else None)
            r_count += 1

    ax2.set_title(f'{dataset_name} — Test (Phase I + II)', fontsize=13)
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_xlim(0, n_te)
    h, l = ax2.get_legend_handles_labels()
    ax2.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(),
               fontsize=8, loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name}_phase2_classification.pdf')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Phase II train+test figure saved: {save_path}")


def plot_us_ip_annotated(index_values, dates, detected_cps, labels,
                          nber_dates, train_end_idx, save_path, dpi=300):
    """US Industrial Production index with recession shading and CP overlay.

    Shows the raw IP index (not growth rate) with:
    - Gray recession bands (NBER dates)
    - Green lines: sustained CPs (regime shifts detected by DeCAFS-FPNN)
    - Orange dashed: recoiled CPs (transient outliers)
    - Vertical dotted line: train/test boundary
    """
    import matplotlib.dates as mdates
    import pandas as pd

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Plot IP index
    ax.plot(dates, index_values, color='#37474F', linewidth=0.8, alpha=0.9)

    # NBER recession bands (pairs: start, end)
    recession_pairs = [
        ('2001-03-01', '2001-11-01'),
        ('2007-12-01', '2009-06-01'),
        ('2020-02-01', '2020-04-01'),
    ]
    for i, (start, end) in enumerate(recession_pairs):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color='#E53935',
                   label='NBER recession' if i == 0 else None)

    # Train/test boundary
    if train_end_idx < len(dates):
        ax.axvline(dates[train_end_idx], color='gray', linestyle=':', linewidth=1.5,
                   alpha=0.7, label='Train/test split')

    # Detected CPs
    n_sust, n_rec = 0, 0
    for i, cp_idx in enumerate(detected_cps):
        if cp_idx >= len(dates):
            continue
        date = dates[cp_idx]
        if i < len(labels):
            if labels[i] == 1:
                ax.axvline(date, color='#1B5E20', linewidth=1.8, alpha=0.8,
                           label='Sustained shift' if n_sust == 0 else None)
                n_sust += 1
            else:
                ax.axvline(date, color='#E65100', linestyle='--',
                           linewidth=1.8, alpha=0.8,
                           label='Recoiled (outlier)' if n_rec == 0 else None)
                n_rec += 1

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Industrial Production Index (2017=100)', fontsize=11)
    ax.set_title('US Industrial Production: Regime Detection via DeCAFS-FPNN',
                 fontsize=13, fontweight='bold')

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9,
              loc='lower right', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"US IP annotated figure saved: {save_path}")
