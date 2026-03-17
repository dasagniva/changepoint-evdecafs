"""Phase I detector comparison: grouped bar chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_phase1_multimetric(detector_results, true_n_cps, save_path, dpi=300):
    """Grouped bar chart comparing Phase I detectors across multiple metrics.

    Args:
        detector_results: dict of {name: {'n_detected': int, 'FP': int,
                          'n_missed': int, 'hausdorff': float, 'MRL': float}}
        true_n_cps: number of true CPs in the evaluation set
        save_path: output path
    """
    detectors = list(detector_results.keys())
    # Short names
    short_names = [d.replace(' (proposed)', '*').replace('Vanilla ', 'V-')
                    .replace(' (w=100)', '') for d in detectors]

    metrics = ['n_detected', 'FP', 'n_missed', 'hausdorff']
    labels = ['Detected CPs', 'False Positives', 'Missed CPs', 'Hausdorff Dist.']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    bar_colors = ['#1565C0' if 'proposed' in d.lower() or 'DeCAFS-FPNN' in d
                  else '#78909C' for d in detectors]

    x = np.arange(len(detectors))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        values = [detector_results[d].get(metric, 0) for d in detectors]

        # Handle inf values
        values = [v if np.isfinite(v) else 0 for v in values]

        bars = ax.bar(x, values, color=bar_colors, edgecolor='white', width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02*max(values+[1]),
                        f'{val:.1f}' if isinstance(val, float) and val != int(val) else f'{int(val)}',
                        ha='center', va='bottom', fontsize=8)

        # Reference line for detected CPs
        if metric == 'n_detected':
            ax.axhline(true_n_cps, color='#E53935', linestyle='--', linewidth=1,
                      label=f'True ({true_n_cps})')
            ax.legend(fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8, rotation=35, ha='right')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)

    fig.suptitle('Phase I Detector Comparison', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Phase I multi-metric chart saved: {save_path}")
