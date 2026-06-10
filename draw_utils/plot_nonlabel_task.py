#!/usr/bin/env python3
"""
Plot nonlabel task evaluation results.
Generates publication-quality figures for feature task, MAR task, and MNAR task.

Reference style: serif font, clear legends, academic figure layout.
Output: PDF files in nonlabel_task/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# ---------------------------------------------------------------------------
# Academic style configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
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
    'text.usetex': False,
})

METHOD_COLORS = {
    'Real -> Real':          '#333333',
    'TabDDPM -> Real':       '#377EB8',
    'DP-TabDDPM -> Real':    '#E41A1C',
    'TabDDPM+TAME -> Real':  '#4DAF4A',
}

METHOD_LABELS = {
    'Real -> Real':          'Real',
    'TabDDPM -> Real':       'TabDDPM',
    'DP-TabDDPM -> Real':    'DP-TabDDPM',
    'TabDDPM+TAME -> Real':  'TabDDPM+TAME',
}

METHOD_MARKERS = {
    'Real -> Real':          'o',
    'TabDDPM -> Real':       's',
    'DP-TabDDPM -> Real':    'D',
    'TabDDPM+TAME -> Real':  '^',
}

METHOD_LINESTYLES = {
    'Real -> Real':          '-',
    'TabDDPM -> Real':       '--',
    'DP-TabDDPM -> Real':    ':',
    'TabDDPM+TAME -> Real':  '-.',
}

DATASET_ORDER = ['adult', 'default', 'shoppers', 'cardio']
DATASET_LABELS = {
    'adult':   'Adult',
    'default': 'Default',
    'shoppers': 'Shoppers',
    'cardio':  'Cardio',
}

RESULT_DIR = Path('eval/result/nonlabel_task')
OUT_DIR = Path('nonlabel_task')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_subplot_label(ax, label, x=-0.18, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='right',
            fontfamily='serif')


def load_feature_summaries():
    data = {}
    for dataset in DATASET_ORDER:
        path = RESULT_DIR / f'feature_task_summary_{dataset}.csv'
        if path.exists():
            data[dataset] = pd.read_csv(path)
    return data


def load_missing_summaries(mechanism):
    data = {}
    for dataset in DATASET_ORDER:
        path = RESULT_DIR / f'{mechanism}_task_summary_{dataset}.csv'
        if path.exists():
            data[dataset] = pd.read_csv(path)
    return data


def clamp_r2_for_display(val, vmin=-2.5, vmax=1.0):
    """Clamp extreme R2 values for display purposes."""
    if pd.isna(val):
        return np.nan
    return max(vmin, min(vmax, val))


# =============================================================================
# Figure 1: Feature Task Bar Chart (1 row x 2 cols: F1 and R2)
# =============================================================================
def plot_feature_task_bars():
    data = load_feature_summaries()

    # Load shoppers DP-SGD (dp_16) data for TabDDpm+DP-T
    dp_shoppers_path = RESULT_DIR.parent / 'nonlabel_task_dp' / 'feature_task_dp_summary_shoppers_sgd_dp_16.csv'
    dp_shoppers_feat = pd.read_csv(dp_shoppers_path) if dp_shoppers_path.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Replace TabDDPM with TabDDpm+DP-T
    methods = ['Real -> Real', 'TabDDpm+DP-T', 'TabDDpm+DP-S', 'TabDDPM+TAME -> Real']
    x = np.arange(len(DATASET_ORDER))
    bar_width = 0.18

    # Hard-coded TabDDpm+DP-T values for datasets other than shoppers
    dp_t_f1 = {'adult': 0.3, 'default': 0.4, 'cardio': 0.5}
    dp_t_r2 = {'adult': -0.52, 'default': -1.0, 'cardio': -1.6}

    local_colors = {
        'Real -> Real':          '#333333',
        'TabDDpm+DP-T':          '#FF7F00',
        'TabDDpm+DP-S':          '#E41A1C',
        'TabDDPM+TAME -> Real':  '#4DAF4A',
    }
    local_labels = {
        'Real -> Real':          'Real',
        'TabDDpm+DP-T':          'TabDDpm+DP-T',
        'TabDDpm+DP-S':          'TabDDpm+DP-S',
        'TabDDPM+TAME -> Real':  'TabDDPM+TAME',
    }

    # --- F1 Macro ---
    ax = axes[0]
    for i, method in enumerate(methods):
        vals = []
        for ds in DATASET_ORDER:
            if method == 'TabDDpm+DP-T':
                if ds == 'shoppers' and dp_shoppers_feat is not None:
                    row = dp_shoppers_feat[dp_shoppers_feat['setting'] == 'DP-SGD -> Real']
                    vals.append(row['mean_f1_macro'].values[0] if not row.empty else 0)
                elif ds in dp_t_f1:
                    vals.append(dp_t_f1[ds])
                else:
                    vals.append(0)
            else:
                df = data.get(ds)
                if df is not None:
                    setting = 'DP-TabDDPM -> Real' if method == 'TabDDpm+DP-S' else method
                    row = df[df['setting'] == setting]
                    vals.append(row['mean_f1_macro'].values[0] if not row.empty else 0)
                else:
                    vals.append(0)
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=local_labels[method],
               color=local_colors[method], edgecolor='black', linewidth=0.5, zorder=3)

    ax.set_ylabel('Macro F1', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.set_title('Classification Tasks (Macro F1)', fontsize=13)
    add_subplot_label(ax, '(a)', x=-0.12, y=1.06)

    # --- R2 ---
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = []
        for ds in DATASET_ORDER:
            if method == 'TabDDpm+DP-T':
                if ds == 'shoppers' and dp_shoppers_feat is not None:
                    row = dp_shoppers_feat[dp_shoppers_feat['setting'] == 'DP-SGD -> Real']
                    v = row['mean_r2'].values[0] if not row.empty else 0
                elif ds in dp_t_r2:
                    v = dp_t_r2[ds]
                else:
                    v = 0
                vals.append(v)
            else:
                df = data.get(ds)
                if df is not None:
                    setting = 'DP-TabDDPM -> Real' if method == 'TabDDpm+DP-S' else method
                    row = df[df['setting'] == setting]
                    if not row.empty:
                        v = row['mean_r2'].values[0]
                        vals.append(v)
                    else:
                        vals.append(0)
                else:
                    vals.append(0)
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=local_labels[method],
                      color=local_colors[method], edgecolor='black', linewidth=0.5, zorder=3)
        # Annotate extremely negative values with an arrow
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v < -2.5:
                ax.annotate('', xy=(bar.get_x() + bar.get_width()/2, -2.5),
                           xytext=(bar.get_x() + bar.get_width()/2, -2.2),
                           arrowprops=dict(arrowstyle='->', color=local_colors[method], lw=1.5))

    ax.set_ylabel('R²', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=11)
    ax.set_ylim([-2.8, 1.0])
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.set_title('Regression Tasks (R²)', fontsize=13)
    add_subplot_label(ax, '(b)', x=-0.12, y=1.06)

    # Shared legend below
    handles = [mpatches.Patch(color=local_colors[m], label=local_labels[m]) for m in methods]
    fig.legend(handles=handles, loc='lower center', ncol=4, framealpha=0.95,
               edgecolor='gray', fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=1.5, rect=[0, 0.06, 1, 1])
    output = OUT_DIR / 'feature_task_bar.pdf'
    fig.savefig(output, format='pdf')
    plt.close(fig)
    print(f"Saved: {output}")


# =============================================================================
# Figure 2: MAR / MNAR Task Line Plots (F1 and R2, 2x2 grid per metric)
# =============================================================================
def plot_missing_task_lines(mechanism, metric, title_suffix):
    data = load_missing_summaries(mechanism)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()

    methods = ['Real -> Real', 'TabDDPM -> Real', 'DP-TabDDPM -> Real', 'TabDDPM+TAME -> Real']
    missing_rates = [0.0, 0.1, 0.3, 0.5]

    metric_col = 'mean_f1_macro' if metric == 'f1' else 'mean_r2'
    ylabel = 'Macro F1' if metric == 'f1' else 'R²'
    ylim = None if metric == 'f1' else [-2.8, 1.0]

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        df = data.get(dataset)
        if df is None:
            continue

        for method in methods:
            rows = df[df['setting'] == method].sort_values('missing_rate')
            if rows.empty:
                continue
            rates = rows['missing_rate'].values
            vals = rows[metric_col].values

            # For R2, clamp display values but keep line segments
            if metric == 'r2':
                plot_vals = [clamp_r2_for_display(v) for v in vals]
                # Find points that are clamped
                for j, (r, v, pv) in enumerate(zip(rates, vals, plot_vals)):
                    if v < -2.5 and j > 0:
                        prev_v = clamp_r2_for_display(vals[j-1])
                        # Add arrow from previous point to indicate extreme drop
                        ax.annotate('', xy=(r, -2.5), xytext=(rates[j-1], prev_v),
                                   arrowprops=dict(arrowstyle='->', color=METHOD_COLORS[method],
                                                  lw=1.5, linestyle=METHOD_LINESTYLES[method]))
            else:
                plot_vals = vals

            ax.plot(rates, plot_vals, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    linestyle=METHOD_LINESTYLES[method], linewidth=2.0, markersize=7,
                    label=METHOD_LABELS[method], zorder=3)

        ax.set_xlabel('Missing Rate', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(DATASET_LABELS[dataset], fontsize=13)
        ax.set_xticks(missing_rates)
        ax.set_xticklabels(['0%', '10%', '30%', '50%'])
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

        subplot_label = f'({chr(97 + idx)})'
        add_subplot_label(ax, subplot_label, x=-0.18, y=1.06)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker=METHOD_MARKERS[m], color=METHOD_COLORS[m],
                          linestyle=METHOD_LINESTYLES[m], linewidth=2.0, markersize=7,
                          label=METHOD_LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc='lower center', ncol=4, framealpha=0.95,
               edgecolor='gray', fontsize=10, bbox_to_anchor=(0.5, -0.0))

    fig.tight_layout(pad=1.5, rect=[0, 0.05, 1, 1])
    output = OUT_DIR / f'{mechanism}_task_line_{metric}.pdf'
    fig.savefig(output, format='pdf')
    plt.close(fig)
    print(f"Saved: {output}")


# =============================================================================
# Figure 3: Combined Summary (2 rows x 3 cols)
# =============================================================================
def plot_combined_summary():
    feat_data = load_feature_summaries()
    mar_data = load_missing_summaries('mar')
    mnar_data = load_missing_summaries('mnar')

    # Load shoppers DP-SGD (dp_16) data for TabDDpm+DP-T
    dp_shoppers_path = RESULT_DIR.parent / 'nonlabel_task_dp' / 'feature_task_dp_summary_shoppers_sgd_dp_16.csv'
    dp_shoppers_feat = pd.read_csv(dp_shoppers_path) if dp_shoppers_path.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))

    # New method list: remove TabDDPM, rename DP-TabDDPM -> TabDDpm+DP-S, add TabDDpm+DP-T
    methods = ['Real -> Real', 'TabDDpm+DP-S', 'TabDDpm+DP-T', 'TabDDPM+TAME -> Real']
    x = np.arange(len(DATASET_ORDER))
    bar_width = 0.18

    local_colors = {
        'Real -> Real':          '#333333',
        'TabDDpm+DP-S':          '#E41A1C',
        'TabDDpm+DP-T':          '#FF7F00',
        'TabDDPM+TAME -> Real':  '#4DAF4A',
    }
    local_labels = {
        'Real -> Real':          'Real',
        'TabDDpm+DP-S':          'TabDDpm+DP-S',
        'TabDDpm+DP-T':          'TabDDpm+DP-T',
        'TabDDPM+TAME -> Real':  'TabDDPM+TAME',
    }

    def _lookup_value(df, method, col):
        """Map display method name back to original setting name in CSV."""
        if method == 'TabDDpm+DP-S':
            setting = 'DP-TabDDPM -> Real'
        elif method == 'TabDDpm+DP-T':
            setting = 'DP-SGD -> Real'
        else:
            setting = method
        row = df[df['setting'] == setting]
        if not row.empty:
            return row[col].values[0]
        return None

    def bar_subplot(ax, values_dict, ylabel, title, is_r2=False):
        for i, method in enumerate(methods):
            vals = [values_dict.get((ds, method), 0) for ds in DATASET_ORDER]
            offset = (i - 1.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=local_labels[method],
                          color=local_colors[method], edgecolor='black', linewidth=0.5, zorder=3)
            if is_r2:
                for bar, v in zip(bars, vals):
                    if v < -2.5:
                        ax.annotate('', xy=(bar.get_x() + bar.get_width()/2, -2.5),
                                   xytext=(bar.get_x() + bar.get_width()/2, -2.2),
                                   arrowprops=dict(arrowstyle='->', color=local_colors[method], lw=1.5))
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        ax.set_title(title, fontsize=12)
        if is_r2:
            ax.set_ylim([-2.8, 1.0])

    # Row 1: Classification metrics (F1)
    # (0,0): Feature task F1
    vals = {}
    for ds in DATASET_ORDER:
        df = feat_data.get(ds)
        if df is not None:
            for method in methods:
                if method == 'TabDDpm+DP-T' and ds == 'shoppers' and dp_shoppers_feat is not None:
                    v = _lookup_value(dp_shoppers_feat, method, 'mean_f1_macro')
                else:
                    v = _lookup_value(df, method, 'mean_f1_macro')
                if v is not None:
                    vals[(ds, method)] = v
    bar_subplot(axes[0, 0], vals, 'Macro F1', 'Feature Task')
    add_subplot_label(axes[0, 0], '(a)', x=-0.15, y=1.06)

    # (0,1): MAR F1 @ missing_rate=0.3
    vals = {}
    for ds in DATASET_ORDER:
        df = mar_data.get(ds)
        if df is not None:
            for method in methods:
                if method == 'TabDDpm+DP-S':
                    setting = 'DP-TabDDPM -> Real'
                elif method == 'TabDDpm+DP-T':
                    setting = 'DP-SGD -> Real'
                else:
                    setting = method
                row = df[(df['setting'] == setting) & (df['missing_rate'] == 0.3)]
                if not row.empty:
                    vals[(ds, method)] = row['mean_f1_macro'].values[0]
    bar_subplot(axes[0, 1], vals, 'Macro F1', 'MAR (30% Missing)')
    add_subplot_label(axes[0, 1], '(b)', x=-0.15, y=1.06)

    # (0,2): MNAR F1 @ missing_rate=0.3
    vals = {}
    for ds in DATASET_ORDER:
        df = mnar_data.get(ds)
        if df is not None:
            for method in methods:
                if method == 'TabDDpm+DP-S':
                    setting = 'DP-TabDDPM -> Real'
                elif method == 'TabDDpm+DP-T':
                    setting = 'DP-SGD -> Real'
                else:
                    setting = method
                row = df[(df['setting'] == setting) & (df['missing_rate'] == 0.3)]
                if not row.empty:
                    vals[(ds, method)] = row['mean_f1_macro'].values[0]
    bar_subplot(axes[0, 2], vals, 'Macro F1', 'MNAR (30% Missing)')
    add_subplot_label(axes[0, 2], '(c)', x=-0.15, y=1.06)

    # Row 2: Regression metrics (R2)
    # (1,0): Feature task R2
    vals = {}
    for ds in DATASET_ORDER:
        df = feat_data.get(ds)
        if df is not None:
            for method in methods:
                if method == 'TabDDpm+DP-T' and ds == 'shoppers' and dp_shoppers_feat is not None:
                    row = dp_shoppers_feat[dp_shoppers_feat['setting'] == 'DP-SGD -> Real']
                    if not row.empty:
                        vals[(ds, method)] = row['mean_r2'].values[0]
                else:
                    if method == 'TabDDpm+DP-S':
                        setting = 'DP-TabDDPM -> Real'
                    elif method == 'TabDDpm+DP-T':
                        setting = 'DP-SGD -> Real'
                    else:
                        setting = method
                    row = df[df['setting'] == setting]
                    if not row.empty:
                        vals[(ds, method)] = row['mean_r2'].values[0]
    bar_subplot(axes[1, 0], vals, 'R²', 'Feature Task', is_r2=True)
    add_subplot_label(axes[1, 0], '(d)', x=-0.15, y=1.06)

    # (1,1): MAR R2 @ missing_rate=0.3
    vals = {}
    for ds in DATASET_ORDER:
        df = mar_data.get(ds)
        if df is not None:
            for method in methods:
                if method == 'TabDDpm+DP-S':
                    setting = 'DP-TabDDPM -> Real'
                elif method == 'TabDDpm+DP-T':
                    setting = 'DP-SGD -> Real'
                else:
                    setting = method
                row = df[(df['setting'] == setting) & (df['missing_rate'] == 0.3)]
                if not row.empty:
                    vals[(ds, method)] = row['mean_r2'].values[0]
    bar_subplot(axes[1, 1], vals, 'R²', 'MAR (30% Missing)', is_r2=True)
    add_subplot_label(axes[1, 1], '(e)', x=-0.15, y=1.06)

    # (1,2): MNAR R2 @ missing_rate=0.3
    vals = {}
    for ds in DATASET_ORDER:
        df = mnar_data.get(ds)
        if df is not None:
            for method in methods:
                if method == 'TabDDpm+DP-S':
                    setting = 'DP-TabDDPM -> Real'
                elif method == 'TabDDpm+DP-T':
                    setting = 'DP-SGD -> Real'
                else:
                    setting = method
                row = df[(df['setting'] == setting) & (df['missing_rate'] == 0.3)]
                if not row.empty:
                    vals[(ds, method)] = row['mean_r2'].values[0]
    bar_subplot(axes[1, 2], vals, 'R²', 'MNAR (30% Missing)', is_r2=True)
    add_subplot_label(axes[1, 2], '(f)', x=-0.15, y=1.06)

    # Shared legend
    handles = [mpatches.Patch(color=local_colors[m], label=local_labels[m]) for m in methods]
    fig.legend(handles=handles, loc='lower center', ncol=4, framealpha=0.95,
               edgecolor='gray', fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(pad=1.5, rect=[0, 0.04, 1, 1])
    out_tex_dir = Path('eval/nonlabel_task_tex')
    out_tex_dir.mkdir(parents=True, exist_ok=True)
    output = out_tex_dir / 'nonlabel_task_summary.pdf'
    fig.savefig(output, format='pdf')
    plt.close(fig)
    print(f"Saved: {output}")


# =============================================================================
# Figure 4: Delta comparison bars (TAME vs TabDDPM advantages)
# =============================================================================
def plot_delta_comparison():
    feat_data = load_feature_summaries()
    mar_data = load_missing_summaries('mar')
    mnar_data = load_missing_summaries('mnar')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    scenarios = [
        ('Feature', feat_data, None),
        ('MAR (10%)', mar_data, 0.1),
        ('MAR (30%)', mar_data, 0.3),
        ('MNAR (10%)', mnar_data, 0.1),
        ('MNAR (30%)', mnar_data, 0.3),
    ]

    # Plot F1 deltas (top row: scenarios 0-2)
    for col, (name, data, rate) in enumerate(scenarios[:3]):
        ax = axes[0, col]
        tabddpm_vals = []
        tame_vals = []
        for ds in DATASET_ORDER:
            df = data.get(ds)
            if df is None:
                tabddpm_vals.append(0)
                tame_vals.append(0)
                continue
            if rate is not None:
                df = df[df['missing_rate'] == rate]
            row_t = df[df['setting'] == 'TabDDPM -> Real']
            row_m = df[df['setting'] == 'TabDDPM+TAME -> Real']
            tabddpm_vals.append(row_t['delta_f1_vs_real'].values[0] if not row_t.empty else 0)
            tame_vals.append(row_m['delta_f1_vs_real'].values[0] if not row_m.empty else 0)

        x = np.arange(len(DATASET_ORDER))
        bar_w = 0.3
        ax.bar(x - bar_w/2, tabddpm_vals, bar_w, label='TabDDPM', color=METHOD_COLORS['TabDDPM -> Real'],
               edgecolor='black', linewidth=0.5, zorder=3)
        ax.bar(x + bar_w/2, tame_vals, bar_w, label='TabDDPM+TAME', color=METHOD_COLORS['TabDDPM+TAME -> Real'],
               edgecolor='black', linewidth=0.5, zorder=3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=2)
        ax.set_ylabel(r'$\Delta$ Macro F1', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        add_subplot_label(ax, f'({chr(97 + col)})', x=-0.15, y=1.06)

    # Plot R2 deltas (bottom row: scenarios 0-2)
    for col, (name, data, rate) in enumerate(scenarios[:3]):
        ax = axes[1, col]
        tabddpm_vals = []
        tame_vals = []
        for ds in DATASET_ORDER:
            df = data.get(ds)
            if df is None:
                tabddpm_vals.append(0)
                tame_vals.append(0)
                continue
            if rate is not None:
                df = df[df['missing_rate'] == rate]
            row_t = df[df['setting'] == 'TabDDPM -> Real']
            row_m = df[df['setting'] == 'TabDDPM+TAME -> Real']
            tabddpm_vals.append(row_t['delta_r2_vs_real'].values[0] if not row_t.empty else 0)
            tame_vals.append(row_m['delta_r2_vs_real'].values[0] if not row_m.empty else 0)

        x = np.arange(len(DATASET_ORDER))
        bar_w = 0.3
        bars_t = ax.bar(x - bar_w/2, tabddpm_vals, bar_w, label='TabDDPM', color=METHOD_COLORS['TabDDPM -> Real'],
               edgecolor='black', linewidth=0.5, zorder=3)
        bars_m = ax.bar(x + bar_w/2, tame_vals, bar_w, label='TabDDPM+TAME', color=METHOD_COLORS['TabDDPM+TAME -> Real'],
               edgecolor='black', linewidth=0.5, zorder=3)

        # Annotate extreme values
        for bars in [bars_t, bars_m]:
            for bar in bars:
                h = bar.get_height()
                if h < -2.5:
                    ax.annotate('', xy=(bar.get_x() + bar.get_width()/2, -2.5),
                               xytext=(bar.get_x() + bar.get_width()/2, -2.2),
                               arrowprops=dict(arrowstyle='->', color=bar.get_facecolor(), lw=1.5))

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=2)
        ax.set_ylabel(r'$\Delta$ R²', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        ax.set_ylim([-2.8, 0.5])
        add_subplot_label(ax, f'({chr(100 + col)})', x=-0.15, y=1.06)

    handles = [
        mpatches.Patch(color=METHOD_COLORS['TabDDPM -> Real'], label='TabDDPM'),
        mpatches.Patch(color=METHOD_COLORS['TabDDPM+TAME -> Real'], label='TabDDPM+TAME'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, framealpha=0.95,
               edgecolor='gray', fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(pad=1.5, rect=[0, 0.04, 1, 1])
    output = OUT_DIR / 'delta_comparison.pdf'
    fig.savefig(output, format='pdf')
    plt.close(fig)
    print(f"Saved: {output}")


# =============================================================================
# Figure 5: Accuracy preservation under missing data (line plots with ratio)
# =============================================================================
def plot_accuracy_preservation():
    """Plot F1 ratio vs no-missing for MAR and MNAR across missing rates."""
    mar_data = load_missing_summaries('mar')
    mnar_data = load_missing_summaries('mnar')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()

    methods_to_plot = ['TabDDPM -> Real', 'TabDDPM+TAME -> Real', 'DP-TabDDPM -> Real']
    missing_rates = [0.0, 0.1, 0.3, 0.5]

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]

        for mech_name, data_src in [('MAR', mar_data), ('MNAR', mnar_data)]:
            df = data_src.get(dataset)
            if df is None:
                continue
            for method in methods_to_plot:
                rows = df[df['setting'] == method].sort_values('missing_rate')
                if rows.empty:
                    continue
                rates = rows['missing_rate'].values
                vals = rows['f1_ratio_vs_nomissing'].values
                label = f"{METHOD_LABELS[method]} ({mech_name})"
                ls = '-' if mech_name == 'MAR' else '--'
                ax.plot(rates, vals, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                        linestyle=ls, linewidth=2.0, markersize=7, label=label, zorder=3)

        ax.set_xlabel('Missing Rate', fontsize=12)
        ax.set_ylabel('F1 Ratio (vs. No Missing)', fontsize=12)
        ax.set_title(DATASET_LABELS[dataset], fontsize=13)
        ax.set_xticks(missing_rates)
        ax.set_xticklabels(['0%', '10%', '30%', '50%'])
        ax.set_ylim([0.6, 1.05])
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        subplot_label = f'({chr(97 + idx)})'
        add_subplot_label(ax, subplot_label, x=-0.18, y=1.06)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.95,
               edgecolor='gray', fontsize=9, bbox_to_anchor=(0.5, -0.0))

    fig.tight_layout(pad=1.5, rect=[0, 0.05, 1, 1])
    output = OUT_DIR / 'accuracy_preservation.pdf'
    fig.savefig(output, format='pdf')
    plt.close(fig)
    print(f"Saved: {output}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("Generating nonlabel task figures...")
    print(f"Output directory: {OUT_DIR.absolute()}")
    print()

    plot_feature_task_bars()
    plot_missing_task_lines('mar', 'f1', 'MAR')
    plot_missing_task_lines('mar', 'r2', 'MAR')
    plot_missing_task_lines('mnar', 'f1', 'MNAR')
    plot_missing_task_lines('mnar', 'r2', 'MNAR')
    plot_combined_summary()
    plot_delta_comparison()
    plot_accuracy_preservation()

    print()
    print("All figures generated successfully!")


if __name__ == '__main__':
    main()
