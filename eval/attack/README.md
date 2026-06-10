# `eval/attack/` — 表格合成数据的 Membership-Inference 评测

本目录提供 **two-sample classifier MIA** 对 TabDDPM 合成数据的隐私评测。
能横向比较 baseline (`tabddpm_<ds>_ori.csv`) 与我们方法 (`tabddpm_<ds>_new.csv`)
在四个数据集 (`adult / shoppers / cardio / default`) 上的攻击 AUC。

## 文件

| 文件 | 作用 |
|---|---|
| `attack_classifier.py` | 攻击主程序 |
| `attack_classifier.md` | 攻击设计与协议说明 |
| `utils.py` | 共享工具 (预处理、AUC/JS/KS、main loop、路径解析) |
| `results/` | 每个数据集一份 `*_attack_classifier_results.txt`,以及 `SUMMARY.md` 汇总 |

## 运行 (从 repo 根目录,需 conda 环境 `tame`)

```bash
python -m eval.attack.attack_classifier --repeats 30
# 子集运行:
python -m eval.attack.attack_classifier --repeats 30 --datasets shoppers cardio
```

## 数据来源
四个数据集 `adult / shoppers / cardio / default`,真实表与合成表来自
`rebuttle/test_data/<ds>/{train.csv, tabddpm_<ds>_ori.csv, tabddpm_<ds>_new.csv}`。
路径在 `utils.resolve_data_paths` 中基于 `__file__` 解析,与运行目录无关。

## 输出
每个数据集写到 `eval/attack/results/<ds>_attack_classifier_results.txt`,
字段含义见 `attack_classifier.md`。整体汇总在 `results/SUMMARY.md`。
