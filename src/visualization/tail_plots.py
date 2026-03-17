"""Tail diagnostics visualization: xi histogram, KS p-values, tail pie chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tail_diagnostics(per_window_df, summary, dataset_name, split_name,
                           save_path, dpi=300):
    """Three-panel tail diagnostics figure.

    Panel 1: Histogram of xi estimates with Weibull/Gumbel/Frechet regions
    Panel 2: KS test p-values across window centers
    Panel 3: Pie chart of tail classification proportions

    Args:
        per_window_df: DataFrame with columns 'center', 'xi_hat', 'ks_p_value',
                       'tail_type', 'n_exceedances'
        summary: dict with aggregate statistics
        dataset_name: for title
        split_name: 'train' or 'test'
        save_path: output path
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    xi_col = 'xi_hat' if 'xi_hat' in per_window_df.columns else 'xi'
    xi_values = per_window_df[xi_col].values if xi_col in per_window_df.columns else np.array([])
    xi_values = xi_values[np.isfinite(xi_values)]

    # --- Panel 1: Histogram of xi ---
    if len(xi_values) > 0:
        ax1.hist(xi_values, bins=30, color='#546E7A', edgecolor='white',
                 alpha=0.8, density=True)

        xlim = ax1.get_xlim()
        # Shade regions
        ax1.axvspan(xlim[0], -0.05, alpha=0.1, color='#1565C0',
                    label='Weibull (ξ < −0.05)')
        ax1.axvspan(-0.05, 0.05, alpha=0.1, color='#4CAF50',
                    label='Gumbel (|ξ| ≤ 0.05)')
        ax1.axvspan(0.05, xlim[1], alpha=0.1, color='#E53935',
                    label='Fréchet (ξ > 0.05)')
        ax1.axvline(0, color='black', linewidth=1, linestyle='-')

        # Mean and median
        mean_xi = np.mean(xi_values)
        median_xi = np.median(xi_values)
        ax1.axvline(mean_xi, color='#D32F2F', linewidth=1.5, linestyle='--',
                    label=f'Mean = {mean_xi:.3f}')
        ax1.axvline(median_xi, color='#1B5E20', linewidth=1.5, linestyle=':',
                    label=f'Median = {median_xi:.3f}')
    else:
        ax1.text(0.5, 0.5, 'No xi data available', transform=ax1.transAxes,
                 ha='center', va='center', fontsize=12)

    ax1.set_xlabel('Shape parameter ξ', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution of Local EVI', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')

    # --- Panel 2: KS p-values across window positions ---
    center_col = 'center' if 'center' in per_window_df.columns else 'centre'
    ks_col = 'ks_p_value' if 'ks_p_value' in per_window_df.columns else 'ks_pvalue'
    if ks_col in per_window_df.columns and center_col in per_window_df.columns:
        centers = per_window_df[center_col].values
        p_values = per_window_df[ks_col].values
        valid = np.isfinite(p_values)

        ax2.scatter(centers[valid], p_values[valid], s=12, alpha=0.6,
                   color='#37474F', edgecolor='none')
        ax2.axhline(0.05, color='#E53935', linewidth=1.5, linestyle='--',
                    label='α = 0.05 (reject threshold)')

        n_not_rejected = np.sum(p_values[valid] >= 0.05)
        pct_not_rejected = n_not_rejected / np.sum(valid) * 100 if np.sum(valid) > 0 else 0

        ax2.set_xlabel('Window center (time index)', fontsize=11)
        ax2.set_ylabel('KS test p-value', fontsize=11)
        ax2.set_title(f'GPD Goodness-of-Fit (not rejected: {pct_not_rejected:.0f}%)',
                      fontsize=11, fontweight='bold')
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'KS data not available', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12)

    # --- Panel 3: Tail type classification pie chart ---
    if 'tail_type' in per_window_df.columns:
        tail_counts = per_window_df['tail_type'].value_counts()

        # Standardize names
        pie_data = {}
        for key, count in tail_counts.items():
            if 'weibull' in str(key).lower() or 'bounded' in str(key).lower():
                pie_data['Weibull\n(bounded)'] = pie_data.get('Weibull\n(bounded)', 0) + count
            elif 'frechet' in str(key).lower() or 'heavy' in str(key).lower():
                pie_data['Fréchet\n(heavy-tailed)'] = pie_data.get('Fréchet\n(heavy-tailed)', 0) + count
            else:
                pie_data['Gumbel\n(exponential)'] = pie_data.get('Gumbel\n(exponential)', 0) + count

        pie_colors = {'Weibull\n(bounded)': '#1565C0',
                      'Gumbel\n(exponential)': '#4CAF50',
                      'Fréchet\n(heavy-tailed)': '#E53935'}

        labels_pie = list(pie_data.keys())
        sizes = list(pie_data.values())
        colors_pie = [pie_colors.get(l, '#9E9E9E') for l in labels_pie]

        if sizes:
            wedges, texts, autotexts = ax3.pie(
                sizes, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 9},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
        else:
            ax3.text(0.5, 0.5, 'No tail type data', transform=ax3.transAxes,
                     ha='center', va='center')

        ax3.set_title('Tail Classification', fontsize=11, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Tail type data not available',
                transform=ax3.transAxes, ha='center', va='center')

    fig.suptitle(f'Tail Diagnostics — {dataset_name} ({split_name})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Tail diagnostics figure saved: {save_path}")
