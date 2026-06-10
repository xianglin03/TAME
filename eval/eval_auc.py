# eval/eval_auc.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. 全局配置
# =========================================================
LABEL_COL = None          # 如果为 None，则默认每个数据集最后一列是 label
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 如果你有真实 test.csv，可以改成 True
USE_EXTERNAL_TEST = True
REAL_TEST_FILENAME = "test.csv"

# 数据集列表
DATASET_LIST = ["shoppers", "adult", "cardio", "default"]

# 数据根目录
BASE_DIR = "./TAME/rebuttle/test_data"


# =========================================================
# 2. 工具函数
# =========================================================
def split_xy(df, label_col):
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()
    return X, y


def build_preprocessor(X_df):
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def make_model_and_grid(num_classes):
    clf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_leaf": [1, 2, 4],
    }

    scoring = "roc_auc" if num_classes == 2 else "roc_auc_ovr"
    return clf, param_grid, scoring


def normalize_adult_label(series):
    """
    把 adult 标签统一成 0/1，兼容空格和句点。
    """
    s = series.astype(str).str.strip()
    s = s.replace({
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1,
        "0": 0,
        "1": 1,
    })
    return s


def maybe_fix_dataset_labels(df, dataname, label_col):
    """
    针对特殊数据集修正 label。
    """
    df = df.copy()

    if dataname == "adult":
        df[label_col] = normalize_adult_label(df[label_col])

    return df


def fit_and_eval_auc(train_df, test_df, label_col, desc):
    X_train, y_train_raw = split_xy(train_df, label_col)
    X_test, y_test_raw = split_xy(test_df, label_col)

    # 标签编码：保证 train/test 使用相同映射
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train_raw, y_test_raw], axis=0).astype(str))

    y_train = label_encoder.transform(y_train_raw.astype(str))
    y_test = label_encoder.transform(y_test_raw.astype(str))

    num_classes = len(label_encoder.classes_)
    if num_classes < 2:
        raise ValueError(f"{desc}: label 只有一个类别，无法计算 AUC。")

    preprocessor = build_preprocessor(X_train)
    model, param_grid, scoring = make_model_and_grid(num_classes)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # 训练集内部做调参
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=3,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    prob = best_model.predict_proba(X_test)

    if num_classes == 2:
        auc = roc_auc_score(y_test, prob[:, 1])
    else:
        auc = roc_auc_score(y_test, prob, multi_class="ovr")

    return {
        "setting": desc,
        "auc": float(auc),
        # "best_params": search.best_params_,
        # "num_classes": int(num_classes),
        # "test_size": int(len(test_df))
    }


# =========================================================
# 3. 主流程
# =========================================================
for dataname in DATASET_LIST:
    root_dir = os.path.join(BASE_DIR, dataname)

    real_path = os.path.join(root_dir, "train.csv")
    syn_ori_path = os.path.join(root_dir, f"tabddpm_{dataname}_ori.csv")
    syn_dp_path = os.path.join(root_dir, f"dp_tabddpm_{dataname}.csv")
    syn_new_path = os.path.join(root_dir, f"tabddpm_{dataname}_new.csv")

    print(f"\n==================== {dataname} ====================")

    # 读数据
    real_df = pd.read_csv(real_path)
    syn_ori_df = pd.read_csv(syn_ori_path)
    syn_new_df = pd.read_csv(syn_new_path)

    # 每个数据集单独确定 label 列
    label_col = real_df.columns[-1] if LABEL_COL is None else LABEL_COL

    # 修正特殊数据集标签
    real_df = maybe_fix_dataset_labels(real_df, dataname, label_col)
    syn_ori_df = maybe_fix_dataset_labels(syn_ori_df, dataname, label_col)
    syn_new_df = maybe_fix_dataset_labels(syn_new_df, dataname, label_col)

    # 保证 synthetic 列顺序和 real 一致
    cols = list(real_df.columns)
    syn_ori_df = syn_ori_df[cols]
    syn_new_df = syn_new_df[cols]

    # 划分真实测试集
    if USE_EXTERNAL_TEST:
        # real_test_path = os.path.join(root_dir, REAL_TEST_FILENAME)
        real_test_path = f"/home/sub4-wy/lxl/TAME/data/{dataname}/test.csv"
        if dataname == "cardio":
            real_test_path = f"/home/sub4-wy/lxl/TAME/data/cardio_train/test.csv"
        real_test_df = pd.read_csv(real_test_path)

        # 若 test 也需要修标签
        real_test_df = maybe_fix_dataset_labels(real_test_df, dataname, label_col)
        real_test_df = real_test_df[cols]
        real_train_df = real_df.copy()
    else:
        real_train_df, real_test_df = train_test_split(
            real_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=real_df[label_col]
        )

    results = []

    # 1) Real -> Real
    results.append(
        fit_and_eval_auc(
            train_df=real_train_df,
            test_df=real_test_df,
            label_col=label_col,
            desc="Real -> Real"
        )
    )

    # 2) TabDDPM -> Real
    results.append(
        fit_and_eval_auc(
            train_df=syn_ori_df,
            test_df=real_test_df,
            label_col=label_col,
            desc="TabDDPM -> Real"
        )
    )

    # 3) TabDDPM+TAME -> Real
    results.append(
        fit_and_eval_auc(
            train_df=syn_new_df,
            test_df=real_test_df,
            label_col=label_col,
            desc="TabDDPM+TAME -> Real"
        )
    )

    # 汇总
    res_df = pd.DataFrame(results)
    real_auc = res_df.loc[res_df["setting"] == "Real -> Real", "auc"].iloc[0]
    res_df["auc_ratio_vs_real"] = res_df["auc"] / real_auc

    print(f"\n===== {dataname} TSTR AUC Results =====")
    # print(res_df[["setting", "auc", "auc_ratio_vs_real", "best_params"]])
    print(res_df[["setting", "auc", "auc_ratio_vs_real"]])

    save_path = f"tstr_auc_results_{dataname}.csv"
    res_df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")