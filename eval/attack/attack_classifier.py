"""
attack_classifier.py  (原 attack.py,加了 _classifier 后缀以与 _nn / _domias 区分)

Two-sample classifier MIA on synthetic tables.

思路: 训练一个 HistGradientBoosting 区分 (member, gen),
再用它输出的 P(member|x) 在 (member, holdout) 上算 AUC / JS / mean-gap。

更多细节见同目录的 attack_classifier.md。

Usage:
    python -m eval.attack.attack_classifier --repeats 30
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from .utils import (
        safe_prepare,
        exact_row_matches,
        upsample_gen_pool,
        summarize_scores,
        aggregate_repeats,
        run_dataset_loop,
    )
except ImportError:  # 允许 `python attack_classifier.py` 直接运行
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import (
        safe_prepare,
        exact_row_matches,
        upsample_gen_pool,
        summarize_scores,
        aggregate_repeats,
        run_dataset_loop,
    )


def evaluate_one_gen(real_df: pd.DataFrame, gen_df: pd.DataFrame,
                     repeats: int = 30, member_frac: float = 0.5,
                     random_state: int = 42) -> dict:
    rng = np.random.RandomState(random_state)

    R, G, Xr_all, Xg_all, _ = safe_prepare(real_df, gen_df)
    exact = exact_row_matches(R, G)

    n = Xr_all.shape[0]
    Xg_pool = upsample_gen_pool(Xg_all, n, rng)

    per_repeat = []
    for _ in range(repeats):
        idxs = np.arange(n)
        rng.shuffle(idxs)
        split = int(np.floor((1.0 - member_frac) * n))
        member_idx = idxs[:split]
        holdout_idx = idxs[split:]

        X_member = Xr_all[member_idx]
        X_holdout = Xr_all[holdout_idx]

        scaler = StandardScaler().fit(X_member)
        Xm = scaler.transform(X_member)
        Xh = scaler.transform(X_holdout)
        Xg = scaler.transform(Xg_pool)

        # 平衡 two-sample 训练集: member vs gen
        k = min(Xg.shape[0], Xm.shape[0])
        gen_idxs = rng.choice(Xg.shape[0], size=k, replace=False)
        X_train_two = np.vstack([Xm[:k], Xg[gen_idxs]])
        y_train_two = np.hstack([np.ones(k), np.zeros(k)])

        clf = HistGradientBoostingClassifier(random_state=rng.randint(0, 10000))
        clf.fit(X_train_two, y_train_two)

        prob_member = clf.predict_proba(Xm)[:, 1]
        prob_holdout = clf.predict_proba(Xh)[:, 1]

        # 概率天然在 [0,1],JS 用固定 range 更精确
        per_repeat.append(summarize_scores(prob_member, prob_holdout, js_fixed_range=(0.0, 1.0)))

    return aggregate_repeats(per_repeat, exact_matches=exact, repeats=repeats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--datasets", nargs="+",
                        default=['adult', 'shoppers', 'cardio', 'default'])
    args = parser.parse_args()
    run_dataset_loop(
        attack_name="two-sample classifier MIA",
        evaluate_one_gen=evaluate_one_gen,
        repeats=args.repeats,
        dataname_set=tuple(args.datasets),
        out_suffix="classifier",
    )
