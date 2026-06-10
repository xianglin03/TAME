"""
Run attack-classifier evaluation on DP and DP-SGD generated data for shoppers.

Usage:
    python -m eval.attack.run_dp_attack --dp
    python -m eval.attack.run_dp_attack --dpsgd

Results are appended to eval/attack/results/shoppers_attack_classifier_results.txt
"""

from __future__ import annotations

import argparse
import os
import glob
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attack_classifier import evaluate_one_gen


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..'))


def extract_param_from_filename(filename: str, prefix: str) -> str:
    """Extract parameter value from filename like shoppers_dp_1.csv -> 1"""
    basename = os.path.basename(filename)
    # basename is like shoppers_dp_1.csv or shoppers_sgd_dp_16.csv
    # Remove .csv and prefix
    name = basename.replace('.csv', '')
    # name is like shoppers_dp_1 or shoppers_sgd_dp_16
    # The param is the last part after the last underscore
    return name.split('_')[-1]


def run_dp_evaluation(repeats: int = 30):
    root = _repo_root()
    real_path = os.path.join(root, 'rebuttle', 'test_data', 'shoppers', 'train.csv')
    real = pd.read_csv(real_path)

    csv_pattern = os.path.join(root, 'sample_end_csv', 'shoppers_dp_*.csv')
    csv_files = sorted(glob.glob(csv_pattern))

    out_dir = os.path.join(root, 'eval', 'attack', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'shoppers_attack_classifier_results.txt')

    results = []
    for csv_file in csv_files:
        param = extract_param_from_filename(csv_file, 'shoppers_dp_')
        gen = pd.read_csv(csv_file)
        print(f"\nEvaluating DP epsilon={param}")
        res = evaluate_one_gen(real, gen, repeats=repeats)
        print(res)
        results.append((param, res))

    with open(out_path, 'a', encoding='utf-8') as f:
        f.write("\n\n")
        f.write("=== DP 评估 ===\n")
        for param, res in results:
            f.write(f"\n--- DP epsilon={param} ---\n")
            for k, v in res.items():
                f.write(f"{k}: {v}\n")

    print(f"\nResults appended to {out_path}")


def run_dpsgd_evaluation(repeats: int = 30):
    root = _repo_root()
    real_path = os.path.join(root, 'rebuttle', 'test_data', 'shoppers', 'train.csv')
    real = pd.read_csv(real_path)

    csv_pattern = os.path.join(root, 'sample_end_csv', 'shoppers_sgd_dp_*.csv')
    csv_files = sorted(glob.glob(csv_pattern))

    out_dir = os.path.join(root, 'eval', 'attack', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'shoppers_attack_classifier_results.txt')

    results = []
    for csv_file in csv_files:
        param = extract_param_from_filename(csv_file, 'shoppers_sgd_dp_')
        gen = pd.read_csv(csv_file)
        print(f"\nEvaluating DP-SGD mode_num={param}")
        res = evaluate_one_gen(real, gen, repeats=repeats)
        print(res)
        results.append((param, res))

    with open(out_path, 'a', encoding='utf-8') as f:
        f.write("\n\n")
        f.write("=== DP-SGD 评估 ===\n")
        for param, res in results:
            f.write(f"\n--- DP-SGD dp_mode_num={param} ---\n")
            for k, v in res.items():
                f.write(f"{k}: {v}\n")

    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", action="store_true", help="Run DP evaluation")
    parser.add_argument("--dpsgd", action="store_true", help="Run DP-SGD evaluation")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if args.dp:
        run_dp_evaluation(repeats=args.repeats)
    elif args.dpsgd:
        run_dpsgd_evaluation(repeats=args.repeats)
    else:
        print("Please specify --dp or --dpsgd")
