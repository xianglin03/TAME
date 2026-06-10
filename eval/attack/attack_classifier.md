# `attack_classifier.py` — Two-Sample Classifier MIA 解释

## 一句话概括
训练一个判别器分 "真实 member" 与 "合成样本",再看它输出的 member 概率能否区分 "训练用过的真实样本 (member)" 与 "没用过的真实样本 (holdout)"。这是一个 **判别式 (discriminative) 的 Membership-Inference Attack (MIA)**。

## 攻击者的能力假设
- 拥有目标生成模型产出的合成表 `gen_df`;
- 拥有一批真实样本,且知道它们中 *哪些参与了生成模型的训练* (即 member 标签可在本地构造);
  - 在本脚本里通过 "随机 split + 多次 repeat" 模拟:每一次 repeat 把真实集随机切成 `member` / `holdout`,member 视作 "进了训练集" 的样本,holdout 视作 "没进训练集" 的样本。
- 不需要访问模型内部参数。

## 评测协议
对每个数据集 `dataset ∈ {adult, shoppers, cardio, default}`,各跑一遍 baseline TabDDPM 合成表 (`*_ori.csv`) 与我们方法合成表 (`*_new.csv`),进行下面流程 `repeats` 次:

1. **预处理 (`safe_prepare`)**: 取真实表与合成表的公共列,丢 NA,object 列用同一份 `LabelEncoder` 编码,得到 `Xr_all` (真实) 与 `Xg_all` (合成)。
2. **合成池补齐**: 若合成集小于真实集,有放回上采样到 `n = |Xr_all|`,得到 `Xg_pool`。
3. 每次 repeat (默认 30 次):
   - 随机把真实集打乱后切 `member_frac` (默认 0.5) 为 holdout,剩下为 member;
   - 仅用 member 拟合 `StandardScaler` (攻击者只见自己掌握的训练分布,不偷看 holdout);
   - 从 `Xg_pool` 随机抽 `k = min(|gen|, |member|)` 条做平衡集;
   - 训练 `HistGradientBoostingClassifier`: 正类 = member,负类 = gen;
   - 用该分类器在 member、holdout 上分别预测 `P(member|x)`,得到 `prob_member`, `prob_holdout`;
   - 以这两组概率为 score、member=1/holdout=0 为标签,算 **AUC** (越接近 0.5 越安全);
   - 同时算这两组概率分布的 **Jensen-Shannon divergence**、**均值差 |mean_gap|** 与 **KS 检验 p-value**,作为更细的诊断量。

最终对 `repeats` 次结果取均值/方差。脚本还报告 **`exact_matches`** —— 合成集中与真实表逐行字符串相同的行数,作为最暴力的 "记忆" 指标。

## Score 的含义
直觉:如果生成器记住了 member,那么 member 与合成数据的 *分布* 重叠会比 holdout 与合成数据的 *分布* 重叠更高,所以 "判别器认为 x 是 member 的概率" 在 member 上会系统性偏高。脚本中 `prob_member > prob_holdout` 的倾向越强,attack AUC 离 0.5 越远,说明生成器泄漏越严重。

## 关键代码位置 (路径以 `eval/attack/` 为根)
- `utils.py` `safe_prepare` — 对齐列、标签编码;
- `attack_classifier.py` `evaluate_one_gen` — 一次完整的 repeats 评测;
- `attack_classifier.py` 内 `HistGradientBoostingClassifier(...)` 训练 + `clf.predict_proba(...)`;
- `utils.py` `summarize_scores` — 计算 AUC / JS / mean_gap / KS;
- `attack_classifier.py` 末尾 `run_dataset_loop(...)`:对四个数据集分别评测 baseline / ours 并落盘 `eval/attack/results/<dataname>_attack_classifier_results.txt`。

## 运行
```
# 模块方式 (推荐,从 repo 根目录运行)
python -m eval.attack.attack_classifier --repeats 30
# 或直接运行
python eval/attack/attack_classifier.py --repeats 30
```

## 它衡量的是什么
**群体级 (population-level) 泄漏**: 判别器是在 "member 群体 vs gen 群体" 上训练的,score 反映 "x 长得像不像 member 这批人"。它的优点是噪声小、统计功效高,缺点是对单条样本的可识别性 (per-sample identifiability) 不直接说话。

## 与其它两种 MIA 的关系
- 与 `attack_nn.py` (最近邻距离): 后者无需训练,直接看 "到合成集的最近距离",粒度是 per-sample 几何距离;
- 与 `attack_domias.py` (DOMIAS 密度比): 后者显式估 `p_gen(x)/p_ref(x)`,从生成模型的密度过拟合角度刻画泄漏。

三者从 **判别式 / 几何式 / 密度式** 三个相互独立的角度,共同对一个合成表的隐私性下结论。

## 输出示例 (落盘到 `eval/attack/results/<dataname>_attack_classifier_results.txt`)
```
=== Baseline 评估 (two-sample classifier MIA) ===
avg_attack_auc: ...
avg_js: ...
avg_mean_score_gap: ...
exact_matches: ...
```
AUC 越靠近 0.5、JS 越靠近 0、mean_gap 越靠近 0、KS p-value 越大,代表合成数据越不泄漏。
