import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# 设置论文级绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

def preprocess_for_viz(df, sample_size=1000):
    """预处理：采样、编码、归一化"""
    df_sample = df.sample(n=min(len(df), sample_size), random_state=42).copy()
    le = LabelEncoder()
    scaler = MinMaxScaler()
    
    # 编码分类变量，填充数值变量
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object' or df_sample[col].dtype.name == 'category':
            df_sample[col] = df_sample[col].astype(str)
            df_sample[col] = le.fit_transform(df_sample[col])
        else:
            df_sample[col] = df_sample[col].fillna(df_sample[col].mean())
            
    # 归一化
    data_scaled = scaler.fit_transform(df_sample)
    return df_sample, data_scaled

# ==========================================
# 1. t-SNE 流形可视化 (Manifold Visualization)
# ==========================================
def plot_tsne_manifold(real, ori, new, dataset_name, save_dir):
    # 采样以加快速度并保持图表清晰
    n_samples = 500
    r_df, r_scaled = preprocess_for_viz(real, n_samples)
    o_df, o_scaled = preprocess_for_viz(ori, n_samples)
    n_df, n_scaled = preprocess_for_viz(new, n_samples)
    
    # 合并数据进行统一降维
    combined = np.vstack([r_scaled, o_scaled, n_scaled])
    labels = (['Real'] * len(r_scaled)) + (['TabDDPM'] * len(o_scaled)) + (['TAME (Ours)'] * len(n_scaled))
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 分别画，控制层级和透明度
    # TabDDPM (Red)
    idx_base = len(r_scaled)
    idx_new = len(r_scaled) + len(o_scaled)
    
    plt.scatter(embedded[idx_base:idx_new, 0], embedded[idx_base:idx_new, 1], 
                c='#d62728', label='TabDDPM', alpha=0.3, s=20, marker='x')
    
    # TAME (Green)
    plt.scatter(embedded[idx_new:, 0], embedded[idx_new:, 1], 
                c='#2ca02c', label='TAME', alpha=0.3, s=20, marker='^')
    
    # Real (Blue) - 放在最上面或者最下面看效果，通常真实数据放中间比较好对比
    plt.scatter(embedded[:idx_base, 0], embedded[:idx_base, 1], 
                c='#1f77b4', label='Real Data', alpha=0.2, s=20, marker='o')

    plt.title(f't-SNE Visualization of Data Manifold - {dataset_name}', fontweight='bold')
    plt.legend()
    plt.axis('off') # t-SNE坐标轴没有物理意义，关掉更美观
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_tsne.pdf'))
    plt.close()
    print(f"[{dataset_name}] t-SNE plot saved.")

# ==========================================
# 2. 双变量联合分布 (Bivariate Joint Distribution)
# ==========================================
def plot_bivariate_joint(real, ori, new, dataset_name, save_dir):
    """
    自动找到真实数据中相关性最强的两个数值特征，画出它们的联合分布。
    展示 'Correlation Preservation' 的细节。
    """
    # 仅选择数值列
    num_real = real.select_dtypes(include=[np.number])
    if num_real.shape[1] < 2:
        return # 列不够，跳过

    # 计算相关性矩阵，找绝对值最大但不为1的pair
    corr = num_real.corr().abs()
    np.fill_diagonal(corr.values, 0)
    
    # 找到相关性最强的一对特征
    max_corr = corr.max().max()
    row, col = np.where(corr.values == max_corr)
    feat1 = num_real.columns[row[0]]
    feat2 = num_real.columns[col[0]]
    
    print(f"[{dataset_name}] Strongest pair: {feat1} vs {feat2} (Corr: {max_corr:.2f})")
    
    # 绘图：一行三列 (Real, TabDDPM, TAME)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    # 设置绘图范围，保证对比公平
    x_min = min(real[feat1].min(), ori[feat1].min(), new[feat1].min())
    x_max = max(real[feat1].max(), ori[feat1].max(), new[feat1].max())
    y_min = min(real[feat2].min(), ori[feat2].min(), new[feat2].min())
    y_max = max(real[feat2].max(), ori[feat2].max(), new[feat2].max())
    
    def plot_kde_joint(data, ax, color, title):
        sns.kdeplot(data=data, x=feat1, y=feat2, ax=ax, 
                    fill=True, cmap=color, levels=10, thresh=0.05)
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.3)

    plot_kde_joint(real, axes[0], "Blues", f"Real Data\n({feat1} vs {feat2})")
    plot_kde_joint(ori, axes[1], "Reds", "TabDDPM (Baseline)")
    plot_kde_joint(new, axes[2], "Greens", "TAME (Ours)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_bivariate.pdf'))
    plt.close()
    print(f"[{dataset_name}] Bivariate plot saved.")

# ==========================================
# 3. 统计箱线图 (Statistical Box Plots)
# ==========================================
def plot_boxplots(real, ori, new, dataset_name, save_dir):
    """
    对比前6个数值特征的统计分布（中位数、四分位、离群点）
    """
    num_cols = real.select_dtypes(include=[np.number]).columns
    cols_to_plot = num_cols[:min(len(num_cols), 6)]
    
    if len(cols_to_plot) == 0: return

    # 归一化数据以便在同一张图上展示（可选，这里为了保持原始物理意义，我们分Feature画）
    # 为了省版面，我们做成 2x3 的 Grid
    
    rows = int(np.ceil(len(cols_to_plot) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        # 构造绘图数据
        data_to_plot = [real[col].dropna(), ori[col].dropna(), new[col].dropna()]
        
        # 绘图
        bplot = axes[i].boxplot(data_to_plot, patch_artist=True, 
                                labels=['Real', 'TabDDPM', 'TAME'],
                                medianprops=dict(color="black", linewidth=1.5))
        
        # 填充颜色
        colors = ['#aec7e8', '#ff9896', '#98df8a'] # Light Blue, Light Red, Light Green
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            
        axes[i].set_title(f'Distribution: {col}', fontweight='bold')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_boxplots.pdf'))
    plt.close()
    print(f"[{dataset_name}] Boxplots saved.")

def process_all_datasets(root_dir):
    subdirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    for subdir in subdirs:
        dataset_name = os.path.basename(subdir)
        print(f"\nProcessing dataset: {dataset_name}...")
        
        train_path = os.path.join(subdir, "train.csv")
        ori_files = glob.glob(os.path.join(subdir, "*_ori.csv"))
        new_files = glob.glob(os.path.join(subdir, "*_new.csv"))
        
        if not (os.path.exists(train_path) and ori_files and new_files):
            continue
            
        try:
            df_real = pd.read_csv(train_path)
            df_ori = pd.read_csv(ori_files[0])
            df_new = pd.read_csv(new_files[0])
            
            save_dir = os.path.join(subdir, "analysis_plots")
            os.makedirs(save_dir, exist_ok=True)
            
            # --- 生成新图表 ---
            plot_tsne_manifold(df_real, df_ori, df_new, dataset_name, save_dir)
            plot_bivariate_joint(df_real, df_ori, df_new, dataset_name, save_dir)
            plot_boxplots(df_real, df_ori, df_new, dataset_name, save_dir)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    ROOT_DIR = "test_data"  # 请修改为你的目录
    if os.path.exists(ROOT_DIR):
        process_all_datasets(ROOT_DIR)