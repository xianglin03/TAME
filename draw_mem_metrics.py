import re
import csv

with open('eval/result/shoppers_05_90w.txt', 'r') as file:
    txt_content = file.read()


# 提取mem_all、cat_ori、num_ori
mem_all = re.search(r'mem_all: ([\d.]+)', txt_content).group(1)
cat_ori = re.search(r'cat_ori: ([\d.]+)', txt_content).group(1)
num_ori = re.search(r'num_ori: ([\d.]+)', txt_content).group(1)

# 提取各个 F1 值（binary_f1, weighted_f1, roc_auc, accuracy）
def extract_f1_score(tag, section):
    pattern = r'"' + tag + '"\s*:\s*([\d.]+)'
    return re.search(pattern, section).group(1)

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
    "mem": cat_ori,
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

# 写入CSV
csv_file = 'mem-metrics.csv'
header = list(data.keys())
with open(csv_file, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writerow(data)


print(f"CSV文件已保存至 {csv_file}")
