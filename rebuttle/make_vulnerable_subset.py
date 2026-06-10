#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_PATTERNS = [
    r"(^|_)id($|_)", r"uuid", r"guid", r"email", r"name", r"phone",
    r"ssn", r"address", r"user", r"patient", r"customer", r"account"
]


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def drop_identifier_like_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped = []
    keep = []
    for c in df.columns:
        lc = c.lower()
        if any(re.search(p, lc) for p in ID_PATTERNS):
            dropped.append(c)
        else:
            keep.append(c)
    return df[keep].copy(), dropped


def build_features(df: pd.DataFrame):
    num_cols, cat_cols = infer_column_types(df)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    if not transformers:
        raise ValueError("No usable columns found after preprocessing.")

    ct = ColumnTransformer(transformers)
    X = ct.fit_transform(df)
    return X, num_cols, cat_cols


def qi_uniqueness_score(df: pd.DataFrame, qi_cols: List[str]) -> np.ndarray:
    if not qi_cols:
        return np.zeros(len(df), dtype=float)
    key_df = df[qi_cols].astype(str).fillna("<NA>")
    freq = key_df.value_counts(dropna=False)
    row_freq = key_df.merge(freq.rename("_freq"), left_on=qi_cols, right_index=True, how="left")["_freq"].to_numpy()
    return 1.0 / np.maximum(row_freq, 1)


def rarity_score(df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
    if not cat_cols:
        return np.zeros(len(df), dtype=float)
    scores = np.zeros(len(df), dtype=float)
    for c in cat_cols:
        probs = df[c].astype(str).fillna("<NA>").value_counts(normalize=True, dropna=False)
        row_p = df[c].astype(str).fillna("<NA>").map(probs).fillna(1.0).to_numpy()
        scores += (1.0 - row_p)
    return scores / max(len(cat_cols), 1)


def knn_isolation_score(X: np.ndarray, k: int) -> np.ndarray:
    n = len(X)
    if n < 2:
        return np.zeros(n, dtype=float)
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    mean_dist = dists[:, 1:].mean(axis=1)
    return mean_dist


def minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.min(arr), np.max(arr)
    if np.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser(description="Extract the most vulnerable rows from a CSV for worst-case privacy auditing.")
    ap.add_argument("csv_path", help="Input CSV file")
    ap.add_argument("--out_csv", default=None, help="Output vulnerable subset CSV path")
    ap.add_argument("--score_csv", default=None, help="Output row score CSV path")
    ap.add_argument("--meta_json", default=None, help="Output metadata JSON path")
    ap.add_argument("--top_frac", type=float, default=0.1, help="Fraction of most vulnerable rows to keep (default: 0.1)")
    ap.add_argument("--top_n", type=int, default=None, help="Keep exactly top_n rows instead of top_frac")
    ap.add_argument("--k", type=int, default=5, help="k for kNN isolation score (default: 5)")
    ap.add_argument("--qi_cols", type=str, default="", help="Comma-separated quasi-identifier columns, e.g. age,sex,zip")
    ap.add_argument("--drop_identifier_like", action="store_true", help="Drop obvious identifier-like columns before scoring")
    args = ap.parse_args()

    df_raw = pd.read_csv(args.csv_path)
    df_score = df_raw.copy()

    dropped_identifier_like = []
    if args.drop_identifier_like:
        df_score, dropped_identifier_like = drop_identifier_like_cols(df_score)

    if len(df_score.columns) == 0:
        raise ValueError("No columns left for scoring after dropping identifier-like columns.")

    qi_cols = [c.strip() for c in args.qi_cols.split(",") if c.strip()]
    qi_cols = [c for c in qi_cols if c in df_score.columns]

    X, num_cols, cat_cols = build_features(df_score)
    knn_score = minmax(knn_isolation_score(X, args.k))
    uniq_score = minmax(qi_uniqueness_score(df_score, qi_cols))
    rare_score = minmax(rarity_score(df_score, cat_cols))

    # Weighted combination: isolation dominates, then uniqueness, then categorical rarity.
    vulnerability = 0.6 * knn_score + 0.3 * uniq_score + 0.1 * rare_score

    scores = pd.DataFrame({
        "original_index": df_raw.index,
        "vulnerability_score": vulnerability,
        "knn_isolation_score": knn_score,
        "qi_uniqueness_score": uniq_score,
        "categorical_rarity_score": rare_score,
    })

    if args.top_n is not None:
        keep_n = max(1, min(int(args.top_n), len(df_raw)))
    else:
        keep_n = max(1, min(int(np.ceil(len(df_raw) * args.top_frac)), len(df_raw)))

    selected = scores.sort_values("vulnerability_score", ascending=False).head(keep_n)
    vulnerable_df = df_raw.iloc[selected["original_index"].to_numpy()].copy()

    stem, _ = os.path.splitext(args.csv_path)
    out_csv = args.out_csv or f"{stem}_vulnerable_subset.csv"
    score_csv = args.score_csv or f"{stem}_row_scores.csv"
    meta_json = args.meta_json or f"{stem}_vulnerable_subset_meta.json"

    vulnerable_df.to_csv(out_csv, index=False)
    scores.sort_values("vulnerability_score", ascending=False).to_csv(score_csv, index=False)

    meta = {
        "input_csv": args.csv_path,
        "output_csv": out_csv,
        "score_csv": score_csv,
        "rows_input": int(len(df_raw)),
        "rows_output": int(len(vulnerable_df)),
        "top_frac": None if args.top_n is not None else args.top_frac,
        "top_n": keep_n,
        "k": args.k,
        "qi_cols_used": qi_cols,
        "numeric_cols_used": num_cols,
        "categorical_cols_used": cat_cols,
        "dropped_identifier_like_cols": dropped_identifier_like,
        "score_weights": {
            "knn_isolation_score": 0.6,
            "qi_uniqueness_score": 0.3,
            "categorical_rarity_score": 0.1,
        },
        "notes": [
            "Higher vulnerability means more isolated / unique / rare rows.",
            "This script only extracts the most vulnerable subset and does not modify values.",
            "For privacy auditing, compare attack success on this subset across methods.",
        ],
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved vulnerable subset to: {out_csv}")
    print(f"Saved row scores to: {score_csv}")
    print(f"Saved metadata to: {meta_json}")
    print(f"Kept {keep_n} / {len(df_raw)} rows.")


if __name__ == "__main__":
    main()
# python make_vulnerable_subset.py shoppers.csv --top_frac 0.1