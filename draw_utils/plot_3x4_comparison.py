#!/usr/bin/env python3
"""
3x4 grid plot: privacy metrics (row 1), quality metrics (row 2), cost metrics (row 3).
Compares Baseline, TAME, DP (epsilon), and DP-SGD (C).
Style follows plot_epsilon_comparison.py.
Outputs PDF.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Academic style configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

COLORS = {
    'baseline': '#999999',
    'dp': '#E41A1C',
    'dpsgd': '#377EB8',
    'tame': '#4DAF4A',
}

MARKERS = {
    'dp': 'o',
    'dpsgd': 's',
    'tame': '^',
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Baseline values (constant horizontal lines)
BASELINE = {
    'mem': 0.1560556907272236,
    'exact_matches': 135,
    'attack_auc': 0.5980419533838682,
    'js_probs': 0.018822934288622226,
    'mle': 0.6407144479173184,
    'shape': 0.9715488670611676,
    'alpha_precision': 0.9159491174481247,
    'beta_recall': 0.5493316511970203,
}

# TAME values (from ngradshoppers_-5_600000_1.txt, step 0 mem)
TAME = {
    'mem': 0.13813192754798598,
    'exact_matches': 121,
    'attack_auc': 0.5319802253104229,
    'js_probs': 0.0068667284002212805,
    'mle': 0.5863399834693987,
    'shape': 0.8956975358705557,
    'alpha_precision': 0.8632843711513623,
    'beta_recall': 0.33653539995794646,
    'wall_time': 0.0031,
    'cpu_time': 0.0156,
    'cpu_util': 497.8334,
    'gpu_mem': 599.7358,
}

# DP data keyed by epsilon
DP_DATA = {
    1:  {'mem': 0.06882941335496084, 'exact_matches': 100, 'attack_auc': 0.5010901246672227,
         'js_probs': 0.0003587021320007606, 'mle': 0.08517336680445026, 'shape': 0.30535780441160276,
         'alpha_precision': 0.04439638340692498, 'beta_recall': 0.05707248205220061,
         'wall_time': 0.0037, 'cpu_time': 0.0156, 'cpu_util': 417.1227, 'gpu_mem': 1033.9370},
    4:  {'mem': 0.12436694602144728, 'exact_matches': 121, 'attack_auc': 0.506225771608768,
         'js_probs': 0.0034013400761544626, 'mle': 0.2565421236115228, 'shape': 0.7574169194877495,
         'alpha_precision': 0.38267399597488816, 'beta_recall': 0.05880267940284167,
         'wall_time': 0.0037, 'cpu_time': 0.0156, 'cpu_util': 417.1227, 'gpu_mem': 1033.9370},
    8:  {'mem': 0.14652158240966032, 'exact_matches': 124, 'attack_auc': 0.5263824273349112,
         'js_probs': 0.004857722014234114, 'mle': 0.6375847749129126, 'shape': 0.8719223413735445,
         'alpha_precision': 0.7931933555615631, 'beta_recall': 0.1805052416569043,
         'wall_time': 0.0037, 'cpu_time': 0.0156, 'cpu_util': 417.1227, 'gpu_mem': 1033.9370},
    16: {'mem': 0.15841849148418496, 'exact_matches': 152, 'attack_auc': 0.5527967394459853,
         'js_probs': 0.01470292540679965, 'mle': 0.492975960751817, 'shape': 0.9101909424969712,
         'alpha_precision': 0.9028866660659037, 'beta_recall': 0.39346970652728963,
         'wall_time': 0.0037, 'cpu_time': 0.0156, 'cpu_util': 417.1227, 'gpu_mem': 1033.9370},
}

# DP-SGD data keyed by C (dp_mode_num)
DPSGD_DATA = {
    1:  {'mem': 0.03596737857078493, 'exact_matches': 10, 'attack_auc': 0.5013240394754491,
         'js_probs': 0.008558093237737695, 'mle': 0.14517336680445026, 'shape': 0.4749982477746738,
         'alpha_precision': 0.08680424138656107, 'beta_recall': 0.00010813733441472628,
         'wall_time': 1.7819, 'cpu_time': 1.8700, 'cpu_util': 104.9444, 'gpu_mem': 5552.3164},
    4:  {'mem': 0.042398846535099614, 'exact_matches': 102, 'attack_auc': 0.520195682419314,
         'js_probs': 0.002288435456301115, 'mle': 0.22719020455218713, 'shape': 0.5673004715989306,
         'alpha_precision': 0.04528551260100322, 'beta_recall': 0.001429815866149986,
         'wall_time': 1.7819, 'cpu_time': 1.8700, 'cpu_util': 104.9444, 'gpu_mem': 5552.3164},
    8:  {'mem': 0.05025682616923497, 'exact_matches': 110, 'attack_auc': 0.5502848478363806,
         'js_probs': 0.0010984263249110007, 'mle': 0.2606895787191475, 'shape': 0.30813633314309175,
         'alpha_precision': 0.03406, 'beta_recall': 0.000,
         'wall_time': 1.7819, 'cpu_time': 1.8700, 'cpu_util': 104.9444, 'gpu_mem': 5552.3164},
    16: {'mem': 0.09328647382175367, 'exact_matches': 130, 'attack_auc': 0.5693530025848669,
         'js_probs': 0.0108695705839652, 'mle': 0.3514342828015873, 'shape': 0.582920308792166,
         'alpha_precision': 0.07698176684389157, 'beta_recall': 0.002186777207052959,
         'wall_time': 1.7819, 'cpu_time': 1.8700, 'cpu_util': 104.9444, 'gpu_mem': 5552.3164},
}

# Row definitions: (metric_key, label, has_baseline)
ROW1_METRICS = [
    ('mem', 'MEM', True),
    ('exact_matches', 'Exact Matches', True),
    ('attack_auc', 'Avg Attack AUC', True),
    ('js_probs', 'Avg JS on Probs', True),
]

ROW2_METRICS = [
    ('mle', 'MLE', True),
    ('shape', 'Shape', True),
    ('alpha_precision', 'Alpha-Precision', True),
    ('beta_recall', 'Beta-Recall', True),
]

ROW3_METRICS = [
    ('wall_time', 'Wall Time', False),
    ('cpu_time', 'CPU Time', False),
    ('cpu_util', 'CPU Util', False),
    ('gpu_mem', 'GPU Mem', False),
]

ALL_ROWS = [ROW1_METRICS, ROW2_METRICS, ROW3_METRICS]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR = Path('/home/lxler/work/TAME/draw_utils')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_subplot_label(ax, label, x=-0.18, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='right',
            fontfamily='serif')


def plot_subplot(ax, metric_key, ylabel, has_baseline, dp_eps, dpsgd_nums, show_legend):
    """Plot a single dual-x-axis subplot."""
    dp_vals = [DP_DATA[e][metric_key] for e in dp_eps]
    dpsgd_vals = [DPSGD_DATA[d][metric_key] for d in dpsgd_nums]

    # Baseline horizontal line
    if has_baseline:
        ax.axhline(y=BASELINE[metric_key], color=COLORS['baseline'], linestyle='-',
                   linewidth=2.0, label='Baseline', zorder=1)

    # DP line on primary x-axis (draw first so TAME covers it)
    ax.plot(dp_eps, dp_vals, marker=MARKERS['dp'], color=COLORS['dp'],
            linestyle='-', linewidth=2.2, markersize=7, label='DP', zorder=2)

    # TAME horizontal line (draw on top of DP)
    tame_val = TAME[metric_key]
    ax.axhline(y=tame_val, color=COLORS['tame'], linestyle='--',
               linewidth=2.0, label='TAME', zorder=3)

    # DP-SGD line on secondary x-axis
    ax2 = ax.twiny()
    ax2.plot(dpsgd_nums, dpsgd_vals, marker=MARKERS['dpsgd'], color=COLORS['dpsgd'],
             linestyle='-', linewidth=2.2, markersize=7, label='DP-SGD', zorder=3)

    # Axis labels
    ax.set_xlabel(r'$\varepsilon$ (DP)', fontsize=12, color=COLORS['dp'])
    ax2.set_xlabel(r'$C$ (DP-SGD)', fontsize=12, color=COLORS['dpsgd'])
    ax.set_ylabel(ylabel, fontsize=13)

    ax.tick_params(axis='x', colors=COLORS['dp'])
    ax2.tick_params(axis='x', colors=COLORS['dpsgd'])
    ax.set_xticks(dp_eps)
    ax.set_xticklabels([str(e) for e in dp_eps])
    ax2.set_xticks(dpsgd_nums)
    ax2.set_xticklabels([str(d) for d in dpsgd_nums])

    # Unified grid: major only for consistent appearance across all subplots
    ax.grid(True, which='major', linestyle='--', alpha=0.5)

    # Legend only on the first subplot (0, 0)
    if show_legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                  framealpha=0.9, edgecolor='gray', fontsize=9)

    return ax2


def plot_3x4_comparison():
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    dp_eps = sorted(DP_DATA.keys())
    dpsgd_nums = sorted(DPSGD_DATA.keys())

    # Plot all rows
    for row_idx, row_metrics in enumerate(ALL_ROWS):
        for col_idx, (metric_key, ylabel, has_baseline) in enumerate(row_metrics):
            ax = axes[row_idx, col_idx]
            show_legend = (row_idx == 0 and col_idx == 0)
            plot_subplot(ax, metric_key, ylabel, has_baseline, dp_eps, dpsgd_nums, show_legend)

            # Subplot label (a)-(l)
            subplot_idx = row_idx * 4 + col_idx
            add_subplot_label(ax, f'({chr(97 + subplot_idx)})')

    # Align y-axis labels for the first column
    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.20, 0.5)

    fig.tight_layout(pad=1.5)
    output_path = OUT_DIR / 'shoppers_3x4_comparison.pdf'
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    plot_3x4_comparison()
