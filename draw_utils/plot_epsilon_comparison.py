#!/usr/bin/env python3
"""
Plot epsilon/dp_num vs. metrics (MEM, MLE, DQ) for DP and DP-SGD groups.
Dual-axis comparison: DP on primary x-axis (ε), DP-SGD on secondary x-axis (C).
Outputs PDF only.
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
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
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
# Data (parameterized for easy modification)
# ---------------------------------------------------------------------------
DP_DATA = {
    1:   {'MEM': 0.0688, 'MLE': 0.0851, 'DQ': 0.1930},
    4:   {'MEM': 0.1244, 'MLE': 0.2566, 'DQ': 0.4860},
    8:   {'MEM': 0.1405, 'MLE': 0.6449, 'DQ': 0.6730},
    16:  {'MEM': 0.1584, 'MLE': 0.4928, 'DQ': 0.7777},
}

TAME_DATA = {'MEM': 0.1381, 'MLE': 0.5869, 'DQ': 0.7430}

DPSGD_DATA = {
    1:   {'MEM': 0.036,  'MLE': 0.145,  'DQ': 0.249},
    4:   {'MEM': 0.0424, 'MLE': 0.227,  'DQ': 0.288},
    8:   {'MEM': 0.0503, 'MLE': 0.261,  'DQ': 0.163},
    16:  {'MEM': 0.0933, 'MLE': 0.351,  'DQ': 0.310},
    32:  {'MEM': 0.1433, 'MLE': 0.3893, 'DQ': 0.3637},
}

METRICS = ['MEM', 'MLE', 'DQ']
METRIC_LABELS = {'MEM': 'MEM', 'MLE': 'MLE', 'DQ': 'DQ'}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR = Path('/home/lxler/work/TAME/draw_utils')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_subplot_label(ax, label, x=-0.18, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='right',
            fontfamily='serif')


def plot_epsilon_comparison():
    """Dual-axis comparison: DP (primary x-axis) vs DP-SGD (secondary x-axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    dp_eps = sorted(DP_DATA.keys())
    dpsgd_nums = sorted(DPSGD_DATA.keys())

    for idx, (metric, ax) in enumerate(zip(METRICS, axes)):
        dp_vals = [DP_DATA[e][metric] for e in dp_eps]
        dpsgd_vals = [DPSGD_DATA[d][metric] for d in dpsgd_nums]

        # DP line on primary x-axis
        ax.plot(dp_eps, dp_vals, marker=MARKERS['dp'], color=COLORS['dp'],
                linestyle='-', linewidth=2.2, markersize=8, label='DP', zorder=3)

        # DP-SGD line on secondary x-axis
        ax2 = ax.twiny()
        ax2.plot(dpsgd_nums, dpsgd_vals, marker=MARKERS['dpsgd'], color=COLORS['dpsgd'],
                 linestyle='-', linewidth=2.2, markersize=8, label='DP-SGD', zorder=3)

        # TAME baseline
        ax.axhline(y=TAME_DATA[metric], color=COLORS['tame'], linestyle='--',
                   linewidth=2.0, label='TAME', zorder=2)

        ax.set_xlabel(r'$\varepsilon$ (DP)', fontsize=14, color=COLORS['dp'])
        ax2.set_xlabel(r'$C$ (DP-SGD)', fontsize=14, color=COLORS['dpsgd'])
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=15)
        ax.tick_params(axis='x', colors=COLORS['dp'])
        ax2.tick_params(axis='x', colors=COLORS['dpsgd'])
        ax.set_xticks(dp_eps)
        ax.set_xticklabels([str(e) for e in dp_eps])
        ax2.set_xticks(dpsgd_nums)
        ax2.set_xticklabels([str(d) for d in dpsgd_nums])
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best',
                  framealpha=0.9, edgecolor='gray')

        add_subplot_label(ax, f'({chr(97+idx)}) {metric}')

    fig.tight_layout(pad=1.5)
    output_path = OUT_DIR / 'epsilon_comparison.pdf'
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    plot_epsilon_comparison()
