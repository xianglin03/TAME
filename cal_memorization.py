import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
warnings.filterwarnings("ignore")

column_indices = {
    'magic': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'categorical': [10]
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
        'numerical': [0,3,4,5,6,9],
        'categorical': [1,2,7,8,10]
    },
    'cardio_train': {
        'numerical': [0,2,3,4,5],
        'categorical': [1,6,7,8,9,10,11]
    },
    'wilt': {
        'numerical': [1,2,3,4,5],
        'categorical': [0]
    },
    'MiniBooNE': {
        'numerical': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        'categorical': [0]
    },
    'shoppers': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'categorical': [10, 11, 12, 13, 14, 15, 16, 17]
    },
    'beijing': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
        'categorical': [8]
    },
    'news': {
        'numerical': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
        'categorical': []
    },
}

min_val_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
max_val_list = [27.0, 3398.75, 24.0, 2549.375, 705.0, 63973.52223, 0.2, 0.2, 361.7637419, 1.0]

def custom_distance_cat(X, Y, numerical_cols, categorical_cols):

    distances = np.zeros(X.shape[0])

    if categorical_cols:
        X_cat = X[categorical_cols].values
        Y_cat = Y[categorical_cols].values.reshape(1, -1)
        cat_diff = (X_cat != Y_cat).astype(float)
        cat_distances = cat_diff.sum(axis=1)
        distances += cat_distances

    average_distances = distances / len(categorical_cols)

    return average_distances

def cal_cat_ori(dataname, generated_path, train_data):

    generated_data = pd.read_csv(generated_path)
    
    # assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    # assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"
    print(generated_data.shape, train_data.shape)
    
    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()
    print(categorical_col_names)

    replicate_count = 0

    for index, W in generated_data.iterrows():
        distances = custom_distance_cat(train_data, W, numerical_col_names, categorical_col_names)
        # print(distances)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        ratio = min_distance / second_min_distance

        if ratio < 1 / 3:
            replicate_count += 1

    replicate_ratio = replicate_count / len(generated_data)
    print(f"cat-ori: {dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio
    

def custom_distance_num(X, Y, numerical_cols, categorical_cols):

    distance_min = np.inf
    distance_avg = 0

    if numerical_cols:
        X_num = X[numerical_cols].astype(float).values
        Y_num = Y[numerical_cols].astype(float).values
        for i in range(len(X_num)):
            d = 0
            for j in range(len(Y_num)):
                d += abs(X_num[i][j] - Y_num[j]) / (max_val_list[j] - min_val_list[j])
            distance_min = min(distance_min, d)
            distance_avg += d
    
    distance_avg /= len(X_num)
    
    return distance_min/distance_avg

def cal_num_lxl(dataname, generated_path, train_data):

    generated_data = pd.read_csv(generated_path)
    
    assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()

    avg_ratio = 0
    for index, item in generated_data.iterrows():
        avg_score = custom_distance_num(train_data, item, numerical_col_names, categorical_col_names)
        avg_ratio += avg_score
    avg_ratio /= len(generated_data)
    
    print(f"Num-lxl: {dataname.capitalize()} - Percent of cat replicate: {avg_ratio:.2%}")
    return avg_ratio
    
def custom_distance_num_ori(X, Y, numerical_cols, categorical_cols):

    distances = np.zeros(X.shape[0])

    if numerical_cols:
        X_num = X[numerical_cols].astype(float).values
        Y_num = Y[numerical_cols].astype(float).values.reshape(1, -1)
        num_diff = X_num - Y_num

        euclidean_distances = np.sqrt(np.sum(num_diff ** 2, axis=1))
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
        distances += normalized_distances

    average_distances = distances / len(numerical_cols)

    return average_distances

def cal_num_ori(dataname, generated_path, train_data):

    generated_data = pd.read_csv(generated_path)
    train_data = train_data
    
    # assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    # assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()

    replicate_count = 0

    # print(train_data.shape)
    # print(generated_data.shape)
    for index, W in generated_data.iterrows():
        distances = custom_distance_num_ori(train_data, W, numerical_col_names, categorical_col_names)
        # print(distances)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        ratio = min_distance / second_min_distance

        if ratio < 1/3:
            replicate_count += 1

    replicate_ratio = replicate_count / len(generated_data)
    print(f"num-ori {dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio
    
def custom_distance_all(X, Y, numerical_cols, categorical_cols):

    distances = np.zeros(X.shape[0], dtype=np.float64)
    
    if numerical_cols:
        X_num = X[numerical_cols].astype(float).values
        Y_num = Y[numerical_cols].astype(float).values.reshape(1, -1)
        num_diff = X_num - Y_num

        euclidean_distances = np.sqrt(np.sum(num_diff ** 2, axis=1))
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
        distances += normalized_distances

    if categorical_cols:
        X_cat = X[categorical_cols].values
        Y_cat = Y[categorical_cols].values.reshape(1, -1)
        cat_diff = (X_cat != Y_cat).astype(float)
        cat_distances = cat_diff.sum(axis=1)
        distances += cat_distances

    average_distances = distances / (len(numerical_cols) + len(categorical_cols))

    return average_distances

def cal_mem_ori(dataname, generated_path, train_data):

    generated_data = pd.read_csv(generated_path)
    train_data = train_data
    
    assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()

    replicate_count = 0

    # print(train_data.shape)
    # print(generated_data.shape)
    for index, W in generated_data.iterrows():
        distances = custom_distance_all(train_data, W, numerical_col_names, categorical_col_names)
        # print(distances)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        ratio = min_distance / second_min_distance

        if ratio < 1/3:
            replicate_count += 1

    replicate_ratio = replicate_count / len(generated_data)
    print(f"mem-ori {dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio

def cal_mem_weight(dataname, generated_path, train_data):
    cat_mem = cal_cat_ori(dataname, generated_path, train_data)
    num_mem = cal_num_ori(dataname, generated_path, train_data)
    
    a = 0.95
    mem_weight = a * cat_mem + (1 - a) * num_mem
    
    print(f"{cat_mem=}, {num_mem=}, {mem_weight=}")
    return cat_mem, num_mem, mem_weight


# 只计算a1, a2的mem
def cal_cat_ori_shoppers(dataname, generated_path, train_data, a1, a2):

    generated_data = pd.read_csv(generated_path)
    
    # assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    # assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"
    print(generated_data.shape, train_data.shape)
    
    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = [col for col in train_data.columns[numerical_cols] if col in [a1, a2]]
    categorical_col_names = [col for col in train_data.columns[categorical_cols] if col in [a1, a2]]
    print('cat', categorical_col_names)

    replicate_count = 0

    for index, W in generated_data.iterrows():
        distances = custom_distance_cat(train_data, W, numerical_col_names, categorical_col_names)
        # print(distances)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        ratio = min_distance / second_min_distance

        if ratio < 99999999/100000000:
            replicate_count += 1

    replicate_ratio = replicate_count / len(generated_data)
    print(f"cat-ori: {dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio

def cal_num_ori_shoppers(dataname, generated_path, train_data, a1, a2):

    generated_data = pd.read_csv(generated_path)
    train_data = train_data
    
    # assert generated_data.shape == train_data.shape, "Generated data and train data must have the same shape"
    # assert generated_data.columns.tolist() == train_data.columns.tolist(), "Generated data and train data must have the same columns"

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = [col for col in train_data.columns[numerical_cols] if col in [a1, a2]]
    categorical_col_names = [col for col in train_data.columns[categorical_cols] if col in [a1, a2]]
    print('num', numerical_col_names)
    
    replicate_count = 0

    # print(train_data.shape)
    # print(generated_data.shape)
    for index, W in generated_data.iterrows():
        distances = custom_distance_num_ori(train_data, W, numerical_col_names, categorical_col_names)
        # print(distances)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        ratio = min_distance / second_min_distance

        if ratio < 99999999/100000000:
            replicate_count += 1

    replicate_ratio = replicate_count / len(generated_data)
    print(f"num-ori {dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio

def cal_mem_weight_shoppers(dataname, generated_path, train_data, a1, a2):
    cat_mem = cal_cat_ori_shoppers(dataname, generated_path, train_data, a1, a2)
    num_mem = cal_num_ori_shoppers(dataname, generated_path, train_data, a1, a2)
    
    a = 0.95
    mem_weight = a * cat_mem + (1 - a) * num_mem
    
    print(f"{cat_mem=}, {num_mem=}, {mem_weight=}")
    return cat_mem, num_mem, mem_weight

def main():
    dataname = "shoppers"
    train_data = pd.read_csv(f"synthetic/{dataname}/real.csv")
    # generate_data_path = "sample_end_csv/tabddpm_cardio_ori.csv"
    # generate_data_path = "sample_end_csv/tabddpm_Churn_ori.csv"
    paths_list = [
        'sample_end_csv/shoppers_dp_1.csv',
        'sample_end_csv/shoppers_dp_4.csv',
        'sample_end_csv/shoppers_dp_8.csv',
        'sample_end_csv/shoppers_dp_16.csv'
    ]
    # Administrative vs Revenue
    # cal_mem_weight_shoppers(dataname, paths_list[0], train_data, 'Administrative', 'Revenue')
    
    for generate_data_path in paths_list: 
        print(generate_data_path)
        cat_mem, num_mem, mem_weight = cal_mem_weight(dataname, generate_data_path, train_data)
        print(f"cat_mem: {cat_mem}, num_mem: {num_mem}, mem_weight: {mem_weight}")
    
    # cat_mem = cal_cat_ori(dataname, generate_data_path, train_data)
    # print(cat_mem)
    # mem = cal_mem_ori(dataname, generate_data_path, train_data)
    # mem = cal_cat_ori(dataname, generate_data_path, train_data)
    
if __name__ == "__main__":
    main()