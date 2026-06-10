import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ============================================================
# 数据来源：eval/attack/results/shoppers_attack_classifier_results.txt
# ============================================================

# DP-S: DP epsilon=8
dp_s = {
    'exact_matches': 124,
    'attack_auc': 0.5983824273349112,
    'js_divergence': 0.018823464234234,
    'mean_prob_gap': 0.0438676464435146,
}

# DP-T: DP-SGD dp_mode_num=16
dp_t = {
    'exact_matches': 130,
    'attack_auc': 0.5693530025848669,
    'js_divergence': 0.0108695705839652,
    'mean_prob_gap': 0.05518033580711865,
}

# TAME
tame = {
    'exact_matches': 121,
    'attack_auc': 0.5319802253104229,
    'js_divergence': 0.0068667284002212805,
    'mean_prob_gap': 0.028685483374797264,
}

# ============================================================
# 归一化方法（按论文描述）
# ============================================================

def normalize_exact(v):
    return v / 200.0

def normalize_js(v):
    return v / 0.03

def normalize_gap(v):
    return v / 0.06

metrics = ['Exact Matches', 'Attack AUC', 'JS Divergence', 'Mean Prob Gap']

# Normalize all values
dp_s_norm = [
    normalize_exact(dp_s['exact_matches']),
    dp_s['attack_auc'],
    normalize_js(dp_s['js_divergence']),
    normalize_gap(dp_s['mean_prob_gap']),
]

dp_t_norm = [
    normalize_exact(dp_t['exact_matches']),
    dp_t['attack_auc'],
    normalize_js(dp_t['js_divergence']),
    normalize_gap(dp_t['mean_prob_gap']),
]

tame_norm = [
    normalize_exact(tame['exact_matches']),
    tame['attack_auc'],
    normalize_js(tame['js_divergence']),
    normalize_gap(tame['mean_prob_gap']),
]

print("=" * 65)
print("归一化结果")
print("=" * 65)
print("{:<20} {:>12} {:>12} {:>12}".format('Metric', 'DP-S', 'DP-T', 'TAME'))
print("-" * 65)
for i, m in enumerate(metrics):
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format(m, dp_s_norm[i], dp_t_norm[i], tame_norm[i]))

# ============================================================
# 绘图（字号调大）
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(metrics))
width = 0.22

color_dp_s = '#dea6ad'   # 偏红粉色
color_dp_t = '#8B9DC3'   # steel blue
color_tame = '#95c985'   # 绿色

bars1 = ax.bar(x - width, dp_s_norm, width, label='DP-S', color=color_dp_s, edgecolor='gray', linewidth=0.5)
bars2 = ax.bar(x, dp_t_norm, width, label='DP-T', color=color_dp_t, edgecolor='gray', linewidth=0.5)
bars3 = ax.bar(x + width, tame_norm, width, label='TAME (Ours)', color=color_tame, edgecolor='gray', linewidth=0.5)

# 标签（字号调大）
def add_labels(bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate('{:.3f}'.format(val),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11)

add_labels(bars1, dp_s_norm)
add_labels(bars2, dp_t_norm)
add_labels(bars3, tame_norm)

ax.set_ylabel('Normalized Privacy Score', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='y', labelsize=13)
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.grid(axis='y', alpha=0.4, linestyle='-')

# 完整边框（参考图风格：上下左右都有黑色边框）
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('black')

fig.text(0.5, -0.02,
         'Fig. 8: Evaluation against membership inference attacks (shoppers dataset).',
         ha='center', fontsize=13, style='italic')

plt.tight_layout()
plt.savefig('/home/lxler/work/TAME/draw_utils/MIA.pdf', bbox_inches='tight')
print("\nSaved to MIA.pdf")
plt.show()
