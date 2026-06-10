import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

def plot_dp_ablation():
    # 数据来自你提供的图片
    data = {
        'Method': ['DP (Baseline)', 'DP + Attr. Align', 'TAME (Ours)'],
        'Memorization Ratio (%) ↓': [14.05, 15.70, 13.81],
        'MLE Average Score (%) ↑': [64.55, 57.14, 62.26],
        'Trend Score (%) ↑': [85.61, 88.25, 87.59]
    }
    
    df = pd.DataFrame(data)
    
    # 设置画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 颜色方案
    colors = ['#d6d6d6', '#7bcc7b', '#eda1ac'] # Gray (Base), Red (Bad case), Green (Ours)
    
    # --- Plot 1: Memorization (Lower is better) ---
    sns.barplot(data=df, x='Method', y='Memorization Ratio (%) ↓', ax=axes[0], palette=colors)
    axes[0].set_title('Privacy Risk (Memorization)', fontweight='bold')
    axes[0].set_ylabel('Memorization Ratio (%)')
    axes[0].set_xlabel('')
    # 添加数值标签
    for i, v in enumerate(df['Memorization Ratio (%) ↓']):
        axes[0].text(i, v + 0.2, f"{v:.2f}", ha='center', fontweight='bold')
    # 标注：DP+AA 变差了
    axes[0].annotate('Worse Privacy!', xy=(1, 15.70), xytext=(1, 12),
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', color='red')

    # --- Plot 2: MLE Utility (Higher is better) ---
    sns.barplot(data=df, x='Method', y='MLE Average Score (%) ↑', ax=axes[1], palette=colors)
    axes[1].set_title('Downstream Utility (MLE)', fontweight='bold')
    axes[1].set_ylabel('Average Score (%)')
    axes[1].set_ylim(50, 70) # 设置范围突出差异
    axes[1].set_xlabel('')
    for i, v in enumerate(df['MLE Average Score (%) ↑']):
        axes[1].text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
    # 标注：性能下降
    axes[1].annotate('Significant Drop!', xy=(1, 57.14), xytext=(1, 53),
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', color='red')

    # --- Plot 3: Trend Score (Higher is better) ---
    sns.barplot(data=df, x='Method', y='Trend Score (%) ↑', ax=axes[2], palette=colors)
    axes[2].set_title('Correlation Preservation (Trend)', fontweight='bold')
    axes[2].set_ylabel('Trend Score (%)')
    axes[2].set_ylim(80, 95)
    axes[2].set_xlabel('')
    for i, v in enumerate(df['Trend Score (%) ↑']):
        axes[2].text(i, v + 0.2, f"{v:.2f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('dp_ablation_analysis.pdf')
    print("Saved: dp_ablation_analysis.pdf")

if __name__ == "__main__":
    # 确保文件夹存在
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plot_dp_ablation()