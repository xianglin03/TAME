# MIA 评测汇总 (eval/attack/results/SUMMARY.md)

**评测协议**: 30 次 repeats,member/holdout 随机切分 50/50。
**真实数据**: `rebuttle/test_data/<ds>/train.csv`
**baseline 合成数据**: `tabddpm_<ds>_ori.csv` (原版 TabDDPM)
**ours 合成数据**: `tabddpm_<ds>_new.csv` (我们方法)

**攻击**: two-sample classifier MIA (HistGradientBoosting 判别 member vs gen,
再用 `P(member|x)` 在 member/holdout 上算 ROC-AUC)。详见 `../attack_classifier.md`。

读数:
- `AUC` 越接近 0.5 越安全;
- `JS` / `gap` 是 score 分布的 Jensen-Shannon 与均值差,越小越安全;
- `exact_matches` 是合成集中与真实表逐行字符串相同的行数,越小越安全。

---

## 主表 (AUC / JS / Exact matches)

| 数据集 | base AUC | ours AUC | Δ AUC | base JS | ours JS | base exact | ours exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| adult | 0.5475 | **0.5291** | −0.0184 | 0.004965 | **0.002164** | 0 | 0 |
| shoppers | 0.5980 | **0.5320** | −0.0660 | 0.018823 | **0.006867** | 135 | **121** |
| cardio | 0.5495 | **0.5193** | −0.0302 | 0.005089 | **0.001093** | 0 | 0 |
| default | 0.5362 | **0.5326** | −0.0036 | 0.003426 | **0.002545** | 23 | 25 |

四个数据集 classifier MIA 攻击 AUC 全部呈现 baseline > ours,降幅 0.4–6.6 pp;
JS 也同方向缩小。shoppers 上 exact_matches 由 135 → 121。

---

## 文件索引

| 数据集 | 结果文件 |
|---|---|
| adult | `adult_attack_classifier_results.txt` |
| shoppers | `shoppers_attack_classifier_results.txt` |
| cardio | `cardio_attack_classifier_results.txt` |
| default | `default_attack_classifier_results.txt` |
