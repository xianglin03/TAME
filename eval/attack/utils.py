"""
attack/utils.py

三种表格 MIA (classifier / nn / domias) 共享的工具:
- safe_prepare: 列对齐 + LabelEncoder + 丢 NA
- exact_row_matches: 行级精确匹配计数
- js_divergence_hist: 直方图 Jensen-Shannon
- summarize_scores: 统一算 AUC / JS / mean-gap / KS p-value
- resolve_data_paths: 基于 __file__ 解析 rebuttle/test_data 真实路径
- run_dataset_loop: 标准 main loop (跑四数据集, baseline vs ours, 保存 txt)
"""

from __future__ import annotations

import os
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


# -----------------------
# Data preparation
# -----------------------
def safe_prepare(real_df: pd.DataFrame, gen_df: pd.DataFrame):
    """对齐列、去 NA、object 列同一 LabelEncoder,返回 float ndarray。"""
    common = [c for c in real_df.columns if c in gen_df.columns]
    if not common:
        raise ValueError("No common columns between real and generated tables.")
    R = real_df[common].copy().reset_index(drop=True).dropna().reset_index(drop=True)
    G = gen_df[common].copy().reset_index(drop=True).dropna().reset_index(drop=True)

    for col in R.columns:
        if R[col].dtype == 'object' or G[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(pd.concat([R[col].astype(str), G[col].astype(str)], axis=0))
            R[col] = le.transform(R[col].astype(str))
            G[col] = le.transform(G[col].astype(str))

    return R, G, R.values.astype(float), G.values.astype(float), common


def exact_row_matches(real_df: pd.DataFrame, gen_df: pd.DataFrame) -> int:
    s_real = real_df.astype(str).apply(lambda r: "|".join(r.values), axis=1)
    s_gen = gen_df.astype(str).apply(lambda r: "|".join(r.values), axis=1)
    return len(set(s_real).intersection(set(s_gen)))


def upsample_gen_pool(Xg_all: np.ndarray, n_target: int, rng: np.random.RandomState) -> np.ndarray:
    """合成集小于真实集时,有放回上采样到 n_target。"""
    if Xg_all.shape[0] >= n_target:
        return Xg_all
    idx = rng.randint(0, Xg_all.shape[0], size=n_target)
    return Xg_all[idx]


# -----------------------
# Metric helpers
# -----------------------
def js_divergence_hist(s1: np.ndarray, s2: np.ndarray, bins: int = 50,
                       fixed_range: tuple | None = None) -> float:
    """
    两组 score 的 JS divergence (基于直方图)。
    - fixed_range=(lo,hi) 时直接用该范围 (适合 [0,1] 概率);
    - 否则对两组并集做 min-max 归一化再 [0,1] 分箱。
    """
    if fixed_range is None:
        lo = float(min(s1.min(), s2.min()))
        hi = float(max(s1.max(), s2.max()))
        if hi - lo < 1e-12:
            return 0.0
        s1 = (s1 - lo) / (hi - lo)
        s2 = (s2 - lo) / (hi - lo)
        rng = (0.0, 1.0)
    else:
        rng = fixed_range
    h1, _ = np.histogram(s1, bins=bins, range=rng, density=True)
    h2, _ = np.histogram(s2, bins=bins, range=rng, density=True)
    h1 = h1 + 1e-12
    h2 = h2 + 1e-12
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    return float(jensenshannon(h1, h2) ** 2)


def summarize_scores(score_member: np.ndarray, score_nonmember: np.ndarray,
                     js_fixed_range: tuple | None = None) -> Dict[str, float]:
    """三种攻击通用: 给定 member / nonmember 的 score,统一算评估量。"""
    y = np.hstack([np.ones(len(score_member)), np.zeros(len(score_nonmember))])
    s = np.hstack([score_member, score_nonmember])
    try:
        auc = float(roc_auc_score(y, s))
    except Exception:
        auc = 0.5
    js = js_divergence_hist(score_member, score_nonmember, bins=50, fixed_range=js_fixed_range)
    gap = float(abs(score_member.mean() - score_nonmember.mean()))
    try:
        ks_p = float(ks_2samp(score_member, score_nonmember).pvalue)
    except Exception:
        ks_p = 1.0
    return {"auc": auc, "js": js, "gap": gap, "ks_p": ks_p}


def aggregate_repeats(per_repeat: list[Dict[str, float]], exact_matches: int,
                      repeats: int) -> Dict[str, float]:
    """对多次 repeat 的 dict 列表做均值/方差汇总。"""
    keys = ["auc", "js", "gap", "ks_p"]
    arr = {k: np.array([d[k] for d in per_repeat], dtype=float) for k in keys}
    return {
        "avg_attack_auc": float(arr["auc"].mean()),
        "std_attack_auc": float(arr["auc"].std()),
        "avg_js": float(arr["js"].mean()),
        "std_js": float(arr["js"].std()),
        "avg_mean_score_gap": float(arr["gap"].mean()),
        "std_mean_score_gap": float(arr["gap"].std()),
        "avg_ks_pval": float(arr["ks_p"].mean()),
        "exact_matches": int(exact_matches),
        "repeats": int(repeats),
    }


# -----------------------
# Path resolution & main loop
# -----------------------
def _repo_root() -> str:
    """eval/attack/<file>.py 向上两层 = repo root。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..'))


def resolve_data_paths(dataname: str) -> tuple[str, str, str]:
    """返回 (train, gen_baseline, gen_ours) 的绝对路径,基于 rebuttle/test_data。"""
    root = _repo_root()
    base = os.path.join(root, 'rebuttle', 'test_data', dataname)
    return (
        os.path.join(base, 'train.csv'),
        os.path.join(base, f'tabddpm_{dataname}_ori.csv'),
        os.path.join(base, f'tabddpm_{dataname}_new.csv'),
    )


def _format_block(title: str, res: Dict[str, float]) -> str:
    lines = [f"=== {title} ==="]
    for k, v in res.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def run_dataset_loop(attack_name: str,
                     evaluate_one_gen: Callable[[pd.DataFrame, pd.DataFrame, int], Dict[str, float]],
                     repeats: int,
                     dataname_set: tuple[str, ...] | list[str] = ('adult', 'shoppers', 'cardio', 'default'),
                     out_suffix: str | None = None) -> None:
    """
    统一的 main loop: 三个攻击脚本只需要传入自己的 evaluate_one_gen 即可。
    txt 结果落盘到 eval/attack/results/<dataname>_attack_<suffix>_results.txt。
    """
    suffix = out_suffix or attack_name
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)

    for dataname in dataname_set:
        train_p, base_p, ours_p = resolve_data_paths(dataname)
        real = pd.read_csv(train_p)
        gen_base = pd.read_csv(base_p)
        gen_ours = pd.read_csv(ours_p)

        print("\n==============================")
        print(f"Evaluating dataset: {dataname}  ({attack_name})")

        print("[Baseline 评估]")
        res_base = evaluate_one_gen(real, gen_base, repeats)
        print(res_base)
        print("\n[Our method 评估]")
        res_ours = evaluate_one_gen(real, gen_ours, repeats)
        print(res_ours)

        print(f"\n=== 对比 ({attack_name}) ===")
        print(f"Exact matches: baseline {res_base['exact_matches']}  vs ours {res_ours['exact_matches']}")
        print(f"Avg attack AUC: baseline {res_base['avg_attack_auc']:.4f}  vs ours {res_ours['avg_attack_auc']:.4f}")
        print(f"Avg JS on scores: baseline {res_base['avg_js']:.6f}  vs ours {res_ours['avg_js']:.6f}")
        print(f"Avg score gap: baseline {res_base['avg_mean_score_gap']:.6f}  vs ours {res_ours['avg_mean_score_gap']:.6f}")
        print(f"Avg KS p-value: baseline {res_base['avg_ks_pval']:.4f}  vs ours {res_ours['avg_ks_pval']:.4f}")

        out_path = os.path.join(out_dir, f"{dataname}_attack_{suffix}_results.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(_format_block(f"Baseline 评估 ({attack_name})", res_base) + "\n\n")
            f.write(_format_block(f"Our method 评估 ({attack_name})", res_ours) + "\n\n")
            f.write("=== 对比 ===\n")
            f.write(f"Exact matches: baseline {res_base['exact_matches']}  vs ours {res_ours['exact_matches']}\n")
            f.write(f"Avg attack AUC: baseline {res_base['avg_attack_auc']:.4f}  vs ours {res_ours['avg_attack_auc']:.4f}\n")
            f.write(f"Avg JS on scores: baseline {res_base['avg_js']:.6f}  vs ours {res_ours['avg_js']:.6f}\n")
            f.write(f"Avg score gap: baseline {res_base['avg_mean_score_gap']:.6f}  vs ours {res_ours['avg_mean_score_gap']:.6f}\n")
            f.write(f"Avg KS p-value: baseline {res_base['avg_ks_pval']:.4f}  vs ours {res_ours['avg_ks_pval']:.4f}\n")
