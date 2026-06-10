import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import Counter
import math

def _conditional_entropy(x, y):
    """
    计算条件熵 H(y|x)。
    这是一个内部辅助函数。
    """
    contingency_table = pd.crosstab(x, y)
    x_marginals = contingency_table.sum(axis=1)
    
    conditional_entropy_sum = 0
    for x_val, row in contingency_table.iterrows():
        p_x = x_marginals[x_val] / len(x)
        entropy_y_given_x = 0
        for y_count in row:
            if y_count > 0:
                p_y_given_x = y_count / x_marginals[x_val]
                entropy_y_given_x -= p_y_given_x * math.log2(p_y_given_x)
        conditional_entropy_sum += p_x * entropy_y_given_x
    return conditional_entropy_sum

def _entropy(s):
    """
    计算一个Pandas Series的熵。
    这是一个内部辅助函数。
    """
    counts = Counter(s)
    total_count = len(s)
    entropy_val = 0
    if total_count == 0:
        return 0
    for count in counts.values():
        p = count / total_count
        entropy_val -= p * math.log2(p)
    return entropy_val

def _theils_u(x, y):
    """
    计算两个分类列的泰尔U统计量 (Theil's U)。
    这是一个内部辅助函数。
    """
    h_y = _entropy(y)
    if h_y == 0:
        return 1.0
    h_y_given_x = _conditional_entropy(x, y)
    return (h_y - h_y_given_x) / h_y

def _correlation_ratio(categorical_col, numerical_col):
    """
    计算一个分类列和一个数值列的相关比 (Eta)。
    这是一个内部辅助函数。
    """
    groups = numerical_col.groupby(categorical_col)
    group_means = groups.mean()
    group_counts = groups.count()
    
    overall_mean = numerical_col.mean()
    ss_between = sum(group_counts * (group_means - overall_mean)**2)
    ss_total = sum((numerical_col - overall_mean)**2)
    
    if ss_total == 0:
        return 0.0
    eta_squared = ss_between / ss_total
    return math.sqrt(eta_squared)

def cal_table_correlation(T, a1, a2):
    """
    根据论文中描述的方法，衡量一个表格(T)中两个属性(a1, a2)之间的相关性。
    函数会自动判断属性类型并选择合适的计算方法。

    参数:
        T (pd.DataFrame): 包含数据的Pandas DataFrame。
        a1 (str): 第一个属性（列）的名称。
        a2 (str): 第二个属性（列）的名称。

    返回:
        一个元组，包含所使用的方法和计算出的相关系数值。
        对于分类-分类变量，返回一个包含两个方向不对称系数的字典。
    """
    if a1 not in T.columns or a2 not in T.columns:
        raise ValueError("Attribute names not found in the table.")

    col1 = T[a1]
    col2 = T[a2]

    cleaned_data = pd.concat([col1, col2], axis=1).dropna()
    if cleaned_data.empty:
        return "数据不足", 0.0
        
    col1_clean = cleaned_data[col1.name]
    col2_clean = cleaned_data[col2.name]

    col1_is_numeric = pd.api.types.is_numeric_dtype(col1_clean)
    col2_is_numeric = pd.api.types.is_numeric_dtype(col2_clean)

    # 情况 1: 数值 vs 数值
    if col1_is_numeric and col2_is_numeric:
        method = "皮尔逊相关系数 (Pearson Correlation)"
        correlation = pearsonr(col1_clean, col2_clean)[0]
        return method, correlation

    # 情况 2: 数值 vs 分类
    elif col1_is_numeric and not col2_is_numeric:
        method = "相关比 (Correlation Ratio / Eta)"
        correlation = _correlation_ratio(col2_clean, col1_clean)
        return method, correlation
    elif not col1_is_numeric and col2_is_numeric:
        method = "相关比 (Correlation Ratio / Eta)"
        correlation = _correlation_ratio(col1_clean, col2_clean)
        return method, correlation

    # 情况 3: 分类 vs 分类
    elif not col1_is_numeric and not col2_is_numeric:
        method = f"泰尔U统计量 (Theil's U)"
        u_12 = _theils_u(col1_clean, col2_clean)
        u_21 = _theils_u(col2_clean, col1_clean)
        correlation = {f'U({a2}|{a1})': u_12, f'U({a1}|{a2})': u_21}
        return method, correlation
        
    else:
        return "不支持的数据类型", None


if __name__ == '__main__':
    # # --- 使用示例 ---
    # data = {
    #     '年龄': [25, 30, 35, 40, 25, 30, 35, 40, 50, 60],
    #     '薪水': [50000, 60000, 70000, 80000, 55000, 65000, 75000, 85000, 90000, 100000],
    #     '部门': ['HR', 'IT', 'IT', 'HR', 'IT', 'Sales', 'Sales', 'HR', 'IT', 'Sales'],
    #     '绩效': ['良好', '优秀', '优秀', '良好', '良好', '优秀', '一般', '良好', '优秀', '一般']
    # }
    # df = pd.DataFrame(data)

    # print("--- 使用样本数据进行测试 ---\n")

    # # 1. 数值 vs 数值
    # method1, corr1 = cal_table_correlation(df, '年龄', '薪水')
    # print(f"1. '年龄' 和 '薪水' 之间的相关性:")
    # print(f"   方法: {method1}")
    # print(f"   相关系数: {corr1:.4f}\n")

    # # 2. 数值 vs 分类
    # method2, corr2 = cal_table_correlation(df, '薪水', '部门')
    # print(f"2. '薪水' 和 '部门' 之间的相关性:")
    # print(f"   方法: {method2}")
    # print(f"   相关系数: {corr2:.4f}\n")

    # # 3. 分类 vs 分类
    # method3, corr3 = cal_table_correlation(df, '部门', '绩效')
    # print(f"3. '部门' 和 '绩效' 之间的相关性:")
    # print(f"   方法: {method3}")
    # print(f"   系数: {corr3}\n")
    
    att_list = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
    
    dataname = 'shoppers'
    
    df1 = pd.read_csv(f'rebuttle/test_data/{dataname}/train.csv')
    df2 = pd.read_csv(f'rebuttle/test_data/{dataname}/tabddpm_{dataname}_ori.csv')
    df3 = pd.read_csv(f'rebuttle/test_data/{dataname}/tabddpm_{dataname}_new.csv')
    
    a1, a2 = 'Administrative', 'ProductRelated'
    
    method1, corr1 = cal_table_correlation(df1, a1, a2)
    result1 = f"[真实数据] {a1} 和 {a2} 之间的相关性:\n   方法: {method1}\n   相关系数: {corr1}\n"
    print(result1)
    
    method2, corr2 = cal_table_correlation(df2, a1, a2)
    result2 = f"[Baseline生成数据] {a1} 和 {a2} 之间的相关性:\n   方法: {method2}\n   相关系数: {corr2}\n"
    print(result2)
    
    method3, corr3 = cal_table_correlation(df3, a1, a2)
    result3 = f"[Ours生成数据] {a1} 和 {a2} 之间的相关性:\n   方法: {method3}\n   相关系数: {corr3}\n"
    print(result3)
    
    # Save results to file
    output_file = f"rebuttle/{dataname}-{a1}-{a2}.txt"
    with open(output_file, "w") as f:
        f.write(result1)
        f.write(result2)
        f.write(result3)
