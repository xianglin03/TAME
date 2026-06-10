import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_absolute_divergence(real_data, synthetic_data, numerical_cols, categorical_cols):

    real_data_num = real_data.iloc[:, numerical_cols]
    synthetic_data_num = synthetic_data.iloc[:, numerical_cols]

    real_data_cat = pd.get_dummies(real_data.iloc[:, categorical_cols], drop_first=True)
    synthetic_data_cat = pd.get_dummies(synthetic_data.iloc[:, categorical_cols], drop_first=True)


    real_data_combined = pd.concat([real_data_num, real_data_cat], axis=1)
    synthetic_data_combined = pd.concat([synthetic_data_num, synthetic_data_cat], axis=1)

    real_data_combined, synthetic_data_combined = real_data_combined.align(synthetic_data_combined, join='outer',
                                                                           axis=1, fill_value=0)

    real_corr = real_data_combined.corr().values
    synthetic_corr = synthetic_data_combined.corr().values

    divergence = np.abs(real_corr - synthetic_corr)
    return divergence

def plot_heatmaps(datasets, column_indices, real_data_paths, synthetic_data_paths):     
    num_datasets = len(datasets)     
    num_methods = len(synthetic_data_paths[0])
    
    methods = ['TabSyn', 'TabSyn\n+Tame', 'TabDDPM', 'TabDDPM\n+Tame']
    
    # 字体参数配置
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=12)
    plt.rc('figure', titlesize=12)  
    
    # 调整布局参数（增加左侧空间）
    fig, axes = plt.subplots(num_methods, num_datasets,
                             figsize=(20, 8),
                             gridspec_kw={'left': 0.2, 'right': 0.9})  # 增加左侧空间
    # 每一行num_methods旁边放一个method名称作为title
    for row_idx in range(num_methods):
        ax = axes[row_idx, 0]
        ax.annotate(methods[row_idx], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 23, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=22, ha='center', va='center', rotation=90)
    
    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    
    # ========== 设置左侧标题 ==========    
    # for row_idx in range(num_methods):
    #     ax = axes[row_idx, 0]  # 多列情况下，使用该行的第一列坐标轴
        
    #     # 设置垂直标题（关键参数）
    #     ax.set_ylabel(methods[row_idx],
    #                   rotation=90,  # 垂直显示
    #                   fontsize=16,  # 字体大小调整
    #                   labelpad=40,  # 标签与图的间距
    #                   va='center',  # 垂直居中
    #                   ha='center')  # 水平居中
    #     ax.yaxis.set_label_position('left')  # 强制左侧显示
    
    # ========== 设置顶部标题 ==========    
    for col_idx in range(num_datasets):
        ax = axes[0, col_idx] if num_methods > 1 else axes[col_idx]
        ax.set_title(datasets[col_idx].capitalize(), pad=10)
    
    # ========== 主绘图逻辑 ==========    
    for col_idx, dataset in enumerate(datasets):
        real_data = pd.read_csv(real_data_paths[col_idx]) 
        numerical_cols = column_indices[dataset]['numerical']
        categorical_cols = column_indices[dataset]['categorical']
        
        for row_idx in range(num_methods):
            ax = axes[row_idx, col_idx]  # 多列情况下，使用该行该列的坐标轴
            
            # 数据加载与计算
            synthetic_data = pd.read_csv(synthetic_data_paths[col_idx][row_idx])
            divergence = compute_absolute_divergence(real_data, synthetic_data,
                                                     numerical_cols, categorical_cols)
            
            # 热力图绘制
            sns.heatmap(
                divergence,
                ax=ax,
                cmap='Blues',
                cbar=False,
                xticklabels=False,  # 关闭X轴标签
                yticklabels=False   # 关闭Y轴标签
            )
            
            # 保留坐标轴线
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_color('#808080')
            ax.spines['bottom'].set_color('#808080')
    
    # ========== 添加全局colorbar ==========    
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # 右侧定位
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cbar_ax, aspect=10)
    
    plt.savefig('quality_heatmap.pdf', bbox_inches='tight')
    plt.show()
    
# Define datasets and methods
datasets = ['adult', 'default', 'shoppers', 'cardio']

# Define the file paths
real_data_paths = [
    'synthetic/adult/real.csv',
    'synthetic/default/real.csv',
    'synthetic/shoppers/real.csv',
    'synthetic/cardio_train/real.csv'
]

synthetic_data_paths = [
    # adult
    ['sample_end_csv/tabsyn_adult_ori.csv', 'sample_end_csv/tabsyn_adult_ngrad_-5_1_200.csv', 
     'sample_end_csv/tabddpm_adult_ori.csv', 'sample_end_csv/gradadult_-5_5000.csv'],
    # default
    ['sample_end_csv/tabsyn_default_ori.csv', 'sample_end_csv/tabsyn_default_ngrad_-5_5000_1.csv',
     'sample_end_csv/tabddpm_default_ori.csv', 'sample_end_csv/graddefault_-5_5000.csv'],
    # shoppers
    ['sample_end_csv/tabsyn_shoppers_ori.csv', 'sample_end_csv/tabsyn_shoppers_ngrad_-5_5000_1.csv',
     'sample_end_csv/tabddpm_shoppers_ori.csv', 'sample_end_csv/ngradshoppers_-5_600000_1.csv'],
    # cardio_train
    ['sample_end_csv/tabsyn_cardio_ori.csv', 'sample_end_csv/tabsyn_cardio_ngrad_-5_1_50.csv',
     'sample_end_csv/tabddpm_cardio_ori.csv', 'sample_end_csv/gradcardio_-5_500.csv']
]

# Column indices for numerical and categorical data
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
    'cardio': {
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

plot_heatmaps(datasets, column_indices, real_data_paths, synthetic_data_paths)
