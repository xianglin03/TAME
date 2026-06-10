import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings

warnings.filterwarnings("ignore")


def custom_distance(X, Y, numerical_cols, categorical_cols):

    distances = np.zeros(X.shape[0])

    if numerical_cols:
        X_num = X[numerical_cols].astype(float).values
        Y_num = Y[numerical_cols].astype(float).values.reshape(1, -1)

        num_diff = X_num - Y_num

        euclidean_distances = np.sqrt(np.sum(num_diff ** 2, axis=1))
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
        distances += normalized_distances


    #if categorical_cols:
    #    X_cat = X[categorical_cols].values
    #    #Y_cat = Y[categorical_cols].values.reshape(1, -1)
    #    Y_cat = Y[categorical_cols].values
    #    #for i in xx:
    #    #    组成一个向量
    #    print(Y_cat)
    #    exit(1)

    #    cat_diff = (X_cat != Y_cat).astype(float)
    #    cat_distances = cat_diff.sum(axis=1)
    #    distances += cat_distances



    #average_distances = distances / (len(numerical_cols) + len(categorical_cols))
    average_distances = distances / (len(numerical_cols))
    #print(average_distances)
    #exit(1)

    return average_distances


def cal_memorization(dataname, generated_path, train_data):
    generated_data = pd.read_csv(generated_path)

    assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"

    # print(f"shape: {generated_data.shape}")

    column_indices = {
        'magic': {
            'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'categorical': [10]
        },
        'shoppers': {
            'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'categorical': [10, 11, 12, 13, 14, 15, 16, 17]
        },
        'adult': {
            'numerical': [0, 2, 4, 10, 11, 12],
            'categorical': [1, 3, 5, 6, 7, 8, 9, 13, 14]
        },
        'default': {
            'numerical': [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'categorical': [1, 2, 3, 5, 6, 7, 8, 9, 10, 23]
        },
        'Churn_Modelling': {
            'numerical': [0, 3, 4, 5, 6, 9],
            'categorical': [1, 2, 7, 8, 10]
        },
        'cardio_train': {
            'numerical': [0, 2, 3, 4, 5],
            'categorical': [1, 6, 7, 8, 9, 10, 11]
        },
        'wilt': {
            'numerical': [1, 2, 3, 4, 5],
            'categorical': [0]
        },
        'MiniBooNE': {
            'numerical': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            'categorical': [0]
        }
    }

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()

    replicate_count = 0

    # 初始化一个空的列表来保存所有符合条件的索引
    all_indices_below_threshold = []
    for index, W in train_data.iterrows():
        # 计算当前行的 distances
        distances = custom_distance(train_data, W, numerical_col_names, categorical_col_names)

        # 找到所有小于 0.0002 的索引
        indices_below_threshold = np.where(distances < 0.00003)[0]

        indices_below_threshold = np.setdiff1d(indices_below_threshold, [index])


        # 将当前的 indices_below_threshold 添加到 all_indices_below_threshold 中
        all_indices_below_threshold.extend(indices_below_threshold)
    # 去重操作
    unique_indices_below_threshold = list(set(all_indices_below_threshold))

    # 计算最终符合条件的索引个数
    indices_below_threshold_sum = len(unique_indices_below_threshold)
    print(unique_indices_below_threshold)
    print(indices_below_threshold_sum)

    # 使用 pandas 删除对应索引的行
    train_data_cleaned = train_data.drop(unique_indices_below_threshold)

    # 将删除后的数据保存为新的 CSV 文件
    train_data_cleaned.to_csv("E:/test/shoppers-1/cleaned_train_data.csv", index=False)
    exit()


    replicate_ratio = replicate_count / len(generated_data)
    print(f"{dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Run memorization calculation for a given dataset and model.")
    parser.add_argument("--dataset", type=str, default="shoppers", help="Name of the dataset (e.g., shoppers).")
    parser.add_argument("--model", type=str, default="tabddpm", help="Name of the model (e.g., tabsyn).")

    # 解析命令行参数
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    # 文件路径设置
    generated_path = f'E:/test/shoppers-1/tabddpm.csv'
    train_data_path = f'E:/test/shoppers-1/real.csv'

    # 读取数据
    try:
        train_data = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"Error: Train data file not found at {train_data_path}")
        return

    #try:
    cal_memorization(dataset, generated_path, train_data)
    #except Exception as e:
    #    print(f"Error occurred during memorization calculation: {e}")


if __name__ == "__main__":
    main()