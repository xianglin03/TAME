import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.eval_feature_task import (  # noqa: E402
    LABEL_COL,
    RANDOM_STATE,
    TEST_SIZE,
    USE_EXTERNAL_TEST,
    eval_classification_target,
    eval_regression_target,
    get_base_dir,
    get_real_test_path,
    infer_task_type,
    summarize_results,
)


DATASET_NAME = "shoppers"
SYN_DATA_PATH = "sample_end_csv/shoppers_sgd_dp_32.csv"
SYN_SETTING_NAME = "DP-SGD -> Real"
OUTPUT_DIR = "eval/result/nonlabel_task_dp"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_dir = get_base_dir()
    print(f"Using BASE_DIR: {base_dir}")

    root_dir = os.path.join(base_dir, DATASET_NAME)
    real_path = os.path.join(root_dir, "train.csv")

    required_paths = [real_path, SYN_DATA_PATH]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("Missing files:")
        for path in missing_paths:
            print(f"  - {path}")
        return

    print(f"\n==================== {DATASET_NAME} ====================")
    print(f"Synthetic file: {SYN_DATA_PATH}")

    real_train_df = pd.read_csv(real_path)
    syn_dp_sgd_df = pd.read_csv(SYN_DATA_PATH)

    label_col = real_train_df.columns[-1] if LABEL_COL is None else LABEL_COL
    cols = list(real_train_df.columns)

    missing_cols = [col for col in cols if col not in syn_dp_sgd_df.columns]
    if missing_cols:
        raise ValueError(f"Synthetic data is missing columns: {missing_cols}")

    syn_dp_sgd_df = syn_dp_sgd_df[cols]

    if USE_EXTERNAL_TEST:
        real_test_path = get_real_test_path(DATASET_NAME)
        if not os.path.exists(real_test_path):
            print(f"[{DATASET_NAME}] real test not found: {real_test_path}, fallback to split train.")
            real_train_df, real_test_df = train_test_split(
                real_train_df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
        else:
            real_test_df = pd.read_csv(real_test_path)[cols]
    else:
        real_train_df, real_test_df = train_test_split(
            real_train_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    settings = {
        "Real -> Real": real_train_df,
        SYN_SETTING_NAME: syn_dp_sgd_df,
    }

    target_cols = [col for col in cols if col != label_col]
    detail_rows = []

    for target_col in target_cols:
        task_type = infer_task_type(real_train_df[target_col])
        for setting_name, train_df in settings.items():
            if task_type == "classification":
                metrics = eval_classification_target(
                    train_df=train_df,
                    test_df=real_test_df,
                    target_col=target_col,
                    label_col=label_col,
                )
                if metrics is None:
                    continue
                detail_rows.append(
                    {
                        "dataset": DATASET_NAME,
                        "setting": setting_name,
                        "task_type": "classification",
                        "target_col": target_col,
                        "f1_macro": metrics["f1_macro"],
                        "accuracy": metrics["accuracy"],
                        "r2": np.nan,
                        "n_classes": metrics["n_classes"],
                    }
                )
            else:
                metrics = eval_regression_target(
                    train_df=train_df,
                    test_df=real_test_df,
                    target_col=target_col,
                    label_col=label_col,
                )
                if metrics is None:
                    continue
                detail_rows.append(
                    {
                        "dataset": DATASET_NAME,
                        "setting": setting_name,
                        "task_type": "regression",
                        "target_col": target_col,
                        "f1_macro": np.nan,
                        "accuracy": np.nan,
                        "r2": metrics["r2"],
                        "n_classes": np.nan,
                    }
                )

    if not detail_rows:
        print(f"[{DATASET_NAME}] no valid tasks, skip saving.")
        return

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize_results(detail_df)

    task_name = Path(SYN_DATA_PATH).stem
    detail_save_path = os.path.join(OUTPUT_DIR, f"feature_task_dp_detail_{task_name}.csv")
    summary_save_path = os.path.join(OUTPUT_DIR, f"feature_task_dp_summary_{task_name}.csv")

    detail_df.to_csv(detail_save_path, index=False)
    summary_df.to_csv(summary_save_path, index=False)

    print("\nFeature-task summary:")
    print(
        summary_df[
            [
                "setting",
                "n_cls_tasks",
                "mean_f1_macro",
                "mean_accuracy",
                "n_reg_tasks",
                "mean_r2",
                "delta_f1_vs_real",
                "delta_r2_vs_real",
            ]
        ]
    )
    print(f"Saved detail:  {detail_save_path}")
    print(f"Saved summary: {summary_save_path}")


if __name__ == "__main__":
    main()
