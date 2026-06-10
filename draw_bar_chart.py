import re
import matplotlib.pyplot as plt
import numpy as np

# 提取各个 F1 值（binary_f1, weighted_f1, roc_auc, accuracy）
def extract_f1_score(tag, section):
    pattern = r'"' + tag + '"\s*:\s*([\d.]+)'
    return re.search(pattern, section).group(1)

methods = ["TabDDPM", "TabDDPM+G$_{\\text{cat}}$", "TabDDPM+G$_{\\text{mix}}$"]

path_list = [
    'eval/result/tabddpm_cardio_ori.txt',
    'eval/result/ngradcardio_-5_500_1.txt',
    'eval/result/tabddpm_cardio_05_005_05.txt',
]

datas = {}

for i, path in enumerate(path_list):
    with open(path, 'r') as f:
        txt_content = f.read()
        # 提取mem_all、cat_ori、num_ori
        mem_all = re.search(r'mem_all: ([\d.]+)', txt_content).group(1)
        cat_ori = re.search(r'cat_ori: ([\d.]+)', txt_content).group(1)
        num_ori = re.search(r'num_ori: ([\d.]+)', txt_content).group(1)

        # 从各个部分提取 F1 等值
        best_f1 = extract_f1_score("binary_f1", re.search(r'"best_f1_scores": {.*?}', txt_content, re.DOTALL).group())
        best_weighted_f1 = extract_f1_score("binary_f1", re.search(r'"best_weighted_scores": {.*?}', txt_content, re.DOTALL).group())
        best_auroc_f1 = extract_f1_score("binary_f1", re.search(r'"best_auroc_scores": {.*?}', txt_content, re.DOTALL).group())
        best_acc_f1 = extract_f1_score("binary_f1", re.search(r'"best_acc_scores": {.*?}', txt_content, re.DOTALL).group())
        best_avg_f1 = extract_f1_score("binary_f1", re.search(r'"best_avg_scores": {.*?}', txt_content, re.DOTALL).group())

        # 提取Density的Shape和Trend
        shape = re.search(r'Shape: ([\d.]+)', txt_content).group(1)
        trend = re.search(r'Trend: ([\d.]+)', txt_content).group(1)

        # 提取Alpha_Precision_all和Beta_Recall_all
        alpha_precision = re.search(r'Alpha_Precision_all = ([\d.]+)', txt_content).group(1)
        beta_recall = re.search(r'Beta_Recall_all = ([\d.]+)', txt_content).group(1)

        # 组织数据
        data = {
            "Mem": mem_all,
            "F1": best_f1,
            "WE": best_weighted_f1,
            "AUR": best_auroc_f1,
            "ACC": best_acc_f1,
            "AVG": best_avg_f1,
            "Shape": shape,
            "Trend": trend,
            "Alpha": alpha_precision,
            "Beta": beta_recall
        }
        datas[methods[i]] = data

# 数据处理
metrics = ['Mem', 'F1', 'WE', 'AUR', 'ACC', 'AVG', 'Shape', 'Trend', 'Alpha', 'Beta']
models = list(datas.keys())
values = {model: [float(datas[model][metric]) for metric in metrics] for model in models}

# Define the colors
colors = ['#7CCD7C', '#EEA2AD', '#20B2AA']

# Bar width
bar_width = 0.2

# Index for the categories
index = np.arange(len(metrics))

# Creating the bar plot
fig, ax = plt.subplots(figsize=(32, 18))
ax.set_facecolor('none')  # Background transparency
ax.grid(True, zorder=0, axis='y')  # Display grid on y-axis behind bars

# Adding labels with adjustments
ax.set_ylim(0, 1.11)
ax.set_ylabel('Memorization Ratio', fontsize=40)
ax.tick_params(axis='y', labelsize=40)
ax.tick_params(axis='x', labelsize=40)
ax.set_xticks(index + bar_width * (len(methods) - 1) / 2)
ax.set_xticklabels(metrics, fontsize=40)
ax.legend(loc='upper left', fontsize=40, frameon=True)

# Adding value labels to bars
def add_value_labels(bars, fontsize=28):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=fontsize,
                rotation=90)  # Rotate the text vertically

# Plotting each method's bars
for i, method in enumerate(methods):
    bars = ax.bar(index + i * bar_width, list(values[method]), bar_width, label=method, color=colors[i], edgecolor='black', zorder=3)
    add_value_labels(bars)

# Adding legend
ax.legend(methods, loc='upper left', fontsize=32, frameon=True)

# Adjust x-axis limits to fit the bars snugly
start = index[0] - 0.5 * bar_width - 0.2
end = index[-1] + (len(methods) - 0.5) * bar_width + 0.2
ax.set_xlim(start, end)

# Display the plot
plt.show()
plt.savefig('bar_chart.pdf', bbox_inches='tight')