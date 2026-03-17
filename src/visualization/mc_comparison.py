"""Monte Carlo classifier comparison: violin plot panel."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mc_classifier_comparison(mc_results, save_path, dpi=300):
    """2x2 violin plot panel showing MC metric distributions per classifier.

    Args:
        mc_results: dict of {classifier_name: {metric_name: array_of_values}}
                    e.g. {'FPNN': {'balanced_accuracy': [0.8, 0.7, ...], ...}, ...}
        save_path: output path
    """
    metrics = [
        ('balanced_accuracy', 'Balanced Accuracy', True),   # higher=better
        ('mcc', "Matthews Corr. Coeff.", True),
        ('brier_score', 'Brier Score', False),               # lower=better
        ('cohen_kappa', "Cohen's Kappa", True),
    ]

    # Fallback: if brier/kappa not available, use what we have
    available_metrics = []
    for key, label, higher in metrics:
        for clf_data in mc_results.values():
            if key in clf_data:
                available_metrics.append((key, label, higher))
                break

    if len(available_metrics) < 2:
        # Fallback to available metrics
        available_metrics = []
        for clf_data in mc_results.values():
            for key in clf_data:
                if key not in [m[0] for m in available_metrics]:
                    label = key.replace('_', ' ').title()
                    available_metrics.append((key, label, True))
                if len(available_metrics) >= 4:
                    break
            if len(available_metrics) >= 4:
                break

    n_metrics = min(len(available_metrics), 4)
    if n_metrics == 0:
        print("No MC data available for violin plot.")
        return

    if n_metrics <= 2:
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
    else:
        rows = 2
        cols = (n_metrics + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows))
        axes = axes.flatten()

    classifiers = list(mc_results.keys())
    # Color: FPNN in blue, others in gray shades
    colors = []
    for clf in classifiers:
        if 'fpnn' in clf.lower() or 'proposed' in clf.lower():
            colors.append('#1565C0')
        else:
            colors.append('#BDBDBD')

    for idx, (metric_key, metric_label, higher_better) in enumerate(available_metrics[:n_metrics]):
        ax = axes[idx]

        data = []
        labels = []
        for clf in classifiers:
            if metric_key in mc_results[clf]:
                vals = mc_results[clf][metric_key]
                vals = np.array(vals)
                vals = vals[np.isfinite(vals)]
                data.append(vals)
                # Short name for x-axis
                short = clf.replace('Logistic Regression', 'LR') \
                           .replace('Isolation Forest', 'IF') \
                           .replace('One-Class SVM', 'OC-SVM') \
                           .replace('Feedforward NN', 'FNN') \
                           .replace('GRU (RNN)', 'GRU')
                labels.append(short)

        if not data:
            continue

        parts = ax.violinplot(data, positions=range(len(data)),
                               showmeans=True, showmedians=True)

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i] if i < len(colors) else '#BDBDBD')
            pc.set_alpha(0.7)

        # Style the lines
        for partname in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            if partname in parts:
                parts[partname].set_color('black')
                parts[partname].set_linewidth(0.8)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
        ax.set_ylabel(metric_label, fontsize=10)

        # Add mean value annotations
        ylim = ax.get_ylim()
        for i, d in enumerate(data):
            if len(d) > 0:
                mean_val = np.mean(d)
                ax.text(i, ylim[1] * 0.95, f'{mean_val:.3f}',
                        ha='center', va='top', fontsize=7, fontweight='bold',
                        color=colors[i] if i < len(colors) else 'black')

        direction = '(higher = better)' if higher_better else '(lower = better)'
        ax.set_title(f'{metric_label} {direction}', fontsize=10)

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Monte Carlo Classification Performance (B = 500)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"MC comparison figure saved: {save_path}")
