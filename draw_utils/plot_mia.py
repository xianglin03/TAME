import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 16,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.02,
    'text.usetex': False,
})

dp_s = {
    'exact_matches': 135,
    'attack_auc': 0.5983824273349112,
    'js_divergence': 0.018823464234234,
    'mean_prob_gap': 0.0438676464435146,
}

dp_t = {
    'exact_matches': 130,
    'attack_auc': 0.5693530025848669,
    'js_divergence': 0.0108695705839652,
    'mean_prob_gap': 0.05518033580711865,
}

tame = {
    'exact_matches': 121,
    'attack_auc': 0.5319802253104229,
    'js_divergence': 0.0068667284002212805,
    'mean_prob_gap': 0.028685483374797264,
}


def normalize_exact(v):
    return v / 200.0


def normalize_js(v):
    return v / 0.03


def normalize_gap(v):
    return v / 0.06


metrics = ['Exact Matches', 'Attack AUC', 'JS Divergence', 'Mean Prob Gap']
series = {
    'DP-S': [
        normalize_exact(dp_s['exact_matches']),
        dp_s['attack_auc'],
        normalize_js(dp_s['js_divergence']),
        normalize_gap(dp_s['mean_prob_gap']),
    ],
    'DP-T': [
        normalize_exact(dp_t['exact_matches']),
        dp_t['attack_auc'],
        normalize_js(dp_t['js_divergence']),
        normalize_gap(dp_t['mean_prob_gap']),
    ],
    'TAME (Ours)': [
        normalize_exact(tame['exact_matches']),
        tame['attack_auc'],
        normalize_js(tame['js_divergence']),
        normalize_gap(tame['mean_prob_gap']),
    ],
}

colors = {
    'DP-S': '#E3E3E3',
    'DP-T': '#EEA2AD',
    'TAME (Ours)': '#7CCD7C',
}

fig, ax = plt.subplots(figsize=(13.6, 6.0))

x = np.arange(len(metrics))
width = 0.22
offsets = [-width, 0, width]


def add_labels(bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.018, 0.98),
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=19,
            color='#333333',
        )


for offset, (label, values) in zip(offsets, series.items()):
    bars = ax.bar(
        x + offset,
        values,
        width,
        label=label,
        color=colors[label],
        edgecolor='black',
        linewidth=0.8,
        zorder=3,
    )
    add_labels(bars, values)

ax.set_ylabel('Normalized Privacy Score', fontsize=24, labelpad=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=22)
ax.set_ylim(0.0, 1.0)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))
ax.tick_params(axis='y', labelsize=22)
ax.grid(axis='y', linestyle='-', color='#D9D9D9', alpha=0.65, linewidth=1.0, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=colors[name], label=name) for name in series]
ax.legend(handles=handles, loc='upper left', ncol=3, frameon=True,
          framealpha=0.92, edgecolor='#CFCFCF', fontsize=20)

fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.965)

out_dir = Path('figures')
out_dir.mkdir(parents=True, exist_ok=True)
output = out_dir / 'MIA_style_preview.pdf'
with plt.rc_context({'savefig.bbox': None}):
    fig.savefig(output, format='pdf', bbox_inches=None)
plt.close(fig)

print(f'Saved: {output.resolve()}')
