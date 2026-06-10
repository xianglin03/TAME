import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")


# =========================================================
# 1. Global config
# =========================================================
LABEL_COL = None  # If None, use the last column as label.
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_EXTERNAL_TEST = True

DATASET_LIST = ["shoppers", "adult", "cardio", "default"]
BASE_DIR = "./TAME/rebuttle/test_data"

OUTPUT_DIR = "eval/result/nonlabel_task"


# =========================================================
# 2. Helpers
# =========================================================
def resolve_path(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def get_base_dir() -> str:
    return resolve_path(
        [
            BASE_DIR,
            "./rebuttle/test_data",
        ]
    )


def get_real_test_path(dataname: str) -> str:
    data_name_in_data_dir = "cardio_train" if dataname == "cardio" else dataname
    return resolve_path(
        [
            f"./TAME/data/{data_name_in_data_dir}/test.csv",
            f"./data/{data_name_in_data_dir}/test.csv",
        ]
    )


def build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def infer_task_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "classification"
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)
        valid_num = int(series.notna().sum())
        if nunique <= 15 and (valid_num == 0 or nunique / max(valid_num, 1) < 0.05):
            return "classification"
        return "regression"
    return "classification"


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    label_col: str,
) -> Optional[Dict[str, pd.DataFrame]]:
    feature_cols = [c for c in train_df.columns if c not in [target_col, label_col]]
    if len(feature_cols) == 0:
        return None

    common_cols = feature_cols + [target_col]
    train_used = train_df[common_cols].copy()
    test_used = test_df[common_cols].copy()
    return {
        "X_train": train_used[feature_cols],
        "y_train": train_used[target_col],
        "X_test": test_used[feature_cols],
        "y_test": test_used[target_col],
    }


def eval_classification_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    label_col: str,
) -> Optional[Dict[str, float]]:
    data = prepare_xy(train_df, test_df, target_col, label_col)
    if data is None:
        return None

    y_train_raw = data["y_train"].astype(str)
    y_test_raw = data["y_test"].astype(str)

    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train_raw, y_test_raw], axis=0))
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(data["X_train"])),
            ("model", model),
        ]
    )
    pipe.fit(data["X_train"], y_train)
    pred = pipe.predict(data["X_test"])

    return {
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_classes": int(len(encoder.classes_)),
    }


def eval_regression_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    label_col: str,
) -> Optional[Dict[str, float]]:
    data = prepare_xy(train_df, test_df, target_col, label_col)
    if data is None:
        return None

    y_train = pd.to_numeric(data["y_train"], errors="coerce")
    y_test = pd.to_numeric(data["y_test"], errors="coerce")

    valid_train = y_train.notna()
    valid_test = y_test.notna()

    X_train = data["X_train"].loc[valid_train]
    X_test = data["X_test"].loc[valid_test]
    y_train = y_train.loc[valid_train]
    y_test = y_test.loc[valid_test]

    if len(y_train) < 10 or len(y_test) < 10:
        return None

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return {
        "r2": float(r2_score(y_test, pred)),
    }


def summarize_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for setting in detail_df["setting"].unique():
        setting_df = detail_df[detail_df["setting"] == setting]
        cls_df = setting_df[setting_df["task_type"] == "classification"]
        reg_df = setting_df[setting_df["task_type"] == "regression"]

        summary_rows.append(
            {
                "setting": setting,
                "n_cls_tasks": int(len(cls_df)),
                "mean_f1_macro": float(cls_df["f1_macro"].mean()) if len(cls_df) > 0 else np.nan,
                "mean_accuracy": float(cls_df["accuracy"].mean()) if len(cls_df) > 0 else np.nan,
                "n_reg_tasks": int(len(reg_df)),
                "mean_r2": float(reg_df["r2"].mean()) if len(reg_df) > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["delta_f1_vs_real"] = np.nan
    summary_df["delta_r2_vs_real"] = np.nan

    baseline = summary_df.loc[summary_df["setting"] == "Real -> Real"]
    if len(baseline) == 1:
        base_f1 = baseline["mean_f1_macro"].iloc[0]
        base_r2 = baseline["mean_r2"].iloc[0]
        summary_df["delta_f1_vs_real"] = summary_df["mean_f1_macro"] - base_f1
        summary_df["delta_r2_vs_real"] = summary_df["mean_r2"] - base_r2

    return summary_df


# =========================================================
# 3. Main
# =========================================================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_dir = get_base_dir()
    print(f"Using BASE_DIR: {base_dir}")

    for dataname in DATASET_LIST:
        root_dir = os.path.join(base_dir, dataname)

        real_path = os.path.join(root_dir, "train.csv")
        syn_ori_path = os.path.join(root_dir, f"tabddpm_{dataname}_ori.csv")
        syn_dp_path = os.path.join(root_dir, f"dp_tabddpm_{dataname}.csv")
        syn_new_path = os.path.join(root_dir, f"tabddpm_{dataname}_new.csv")

        required_paths = [real_path, syn_ori_path, syn_dp_path, syn_new_path]
        missing_paths = [p for p in required_paths if not os.path.exists(p)]
        if missing_paths:
            print(f"\n[{dataname}] skip, missing files:")
            for p in missing_paths:
                print(f"  - {p}")
            continue

        print(f"\n==================== {dataname} ====================")
        real_train_df = pd.read_csv(real_path)
        syn_ori_df = pd.read_csv(syn_ori_path)
        syn_dp_df = pd.read_csv(syn_dp_path)
        syn_new_df = pd.read_csv(syn_new_path)

        label_col = real_train_df.columns[-1] if LABEL_COL is None else LABEL_COL
        cols = list(real_train_df.columns)

        syn_ori_df = syn_ori_df[cols]
        syn_dp_df = syn_dp_df[cols]
        syn_new_df = syn_new_df[cols]

        if USE_EXTERNAL_TEST:
            real_test_path = get_real_test_path(dataname)
            if not os.path.exists(real_test_path):
                print(f"[{dataname}] real test not found: {real_test_path}, fallback to split train.")
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
            "TabDDPM -> Real": syn_ori_df,
            "DP-TabDDPM -> Real": syn_dp_df,
            "TabDDPM+TAME -> Real": syn_new_df,
        }

        target_cols = [c for c in cols if c != label_col]
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
                            "dataset": dataname,
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
                            "dataset": dataname,
                            "setting": setting_name,
                            "task_type": "regression",
                            "target_col": target_col,
                            "f1_macro": np.nan,
                            "accuracy": np.nan,
                            "r2": metrics["r2"],
                            "n_classes": np.nan,
                        }
                    )

        if len(detail_rows) == 0:
            print(f"[{dataname}] no valid tasks, skip saving.")
            continue

        detail_df = pd.DataFrame(detail_rows)
        summary_df = summarize_results(detail_df)

        detail_save_path = os.path.join(OUTPUT_DIR, f"feature_task_detail_{dataname}.csv")
        summary_save_path = os.path.join(OUTPUT_DIR, f"feature_task_summary_{dataname}.csv")

        detail_df.to_csv(detail_save_path, index=False)
        summary_df.to_csv(summary_save_path, index=False)

        print("\nFeature-task summary:")
        print(summary_df[["setting", "n_cls_tasks", "mean_f1_macro", "n_reg_tasks", "mean_r2", "delta_f1_vs_real", "delta_r2_vs_real"]])
        print(f"Saved detail:  {detail_save_path}")
        print(f"Saved summary: {summary_save_path}")


if __name__ == "__main__":
    main()
