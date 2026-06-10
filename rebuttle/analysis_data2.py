import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
import math

# 忽略警告
warnings.filterwarnings("ignore")

# 设置顶刊绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

def load_and_clean_data(path):
    """读取数据并去除重复列"""
    try:
        df = pd.read_csv(path)
        # 去除重复列名 (这是导致 Adult 报错的主要原因)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def preprocess_simple(df):
    """基础预处理：区分数值和类别"""
    df_clean = df.copy()
    num_cols = []
    cat_cols = []
    
    for col in df_clean.columns:
        # 判断是否为数值类型
        if pd.api.types.is_numeric_dtype(df_clean[col]) and df_clean[col].dtype != 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            if df_clean[col].nunique() < 20: # 数值较少时也视为类别（如 education-num）
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            df_clean[col] = df_clean[col].astype(str)
            cat_cols.append(col)
            
    return df_clean, num_cols, cat_cols

def preprocess_numeric_matrix(df, sample_size=2000):
    """全数值化处理，用于 PCA/t-SNE"""
    df_sample = df.sample(n=min(len(df), sample_size), random_state=42).copy()
    le = LabelEncoder()
    scaler = MinMaxScaler()
    
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object' or df_sample[col].dtype.name == 'category':
            df_sample[col] = le.fit_transform(df_sample[col].astype(str))
        else:
            df_sample[col] = df_sample[col].fillna(df_sample[col].mean())
            
    data_scaled = scaler.fit_transform(df_sample)
    return df_sample, data_scaled

# ==========================================
# 1. PCA & t-SNE (Global Structure)
# ==========================================
def plot_manifold_learning(real, ori, new, dataset_name, save_dir):
    n_samples = 1000
    _, r_scaled = preprocess_numeric_matrix(real, n_samples)
    _, o_scaled = preprocess_numeric_matrix(ori, n_samples)
    _, n_scaled = preprocess_numeric_matrix(new, n_samples)
    
    # Combined data
    combined = np.vstack([r_scaled, o_scaled, n_scaled])
    
    # --- PCA ---
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(combined)
    
    plt.figure(figsize=(8, 8))
    idx1 = len(r_scaled)
    idx2 = idx1 + len(o_scaled)
    
    plt.scatter(pca_res[:idx1, 0], pca_res[:idx1, 1], c='lightgray', label='Real', alpha=0.5, s=30)
    plt.scatter(pca_res[idx1:idx2, 0], pca_res[idx1:idx2, 1], c='#d62728', label='TabDDPM', alpha=0.6, marker='x', s=30)
    plt.scatter(pca_res[idx2:, 0], pca_res[idx2:, 1], c='#2ca02c', label='TAME', alpha=0.6, marker='^', s=30)
    
    plt.title(f'PCA Projection - {dataset_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_pca.pdf'))
    plt.close()

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_res = tsne.fit_transform(combined)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_res[:idx1, 0], tsne_res[:idx1, 1], c='#1f77b4', label='Real', alpha=0.2, s=20)
    plt.scatter(tsne_res[idx1:idx2, 0], tsne_res[idx1:idx2, 1], c='#d62728', label='TabDDPM', alpha=0.3, marker='x', s=20)
    plt.scatter(tsne_res[idx2:, 0], tsne_res[idx2:, 1], c='#2ca02c', label='TAME', alpha=0.3, marker='^', s=20)
    
    plt.title(f't-SNE Manifold - {dataset_name}')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_tsne.pdf'))
    plt.close()
    print(f"[{dataset_name}] PCA & t-SNE saved.")

# ==========================================
# 2. CDF Curves (Statistical Fidelity)
# ==========================================
def plot_cdf_curves(real, ori, new, dataset_name, save_dir):
    _, num_cols, _ = preprocess_simple(real)
    cols_to_plot = num_cols[:min(len(num_cols), 6)]
    if not cols_to_plot: return

    rows = int(np.ceil(len(cols_to_plot) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if len(cols_to_plot)==1 else axes

    for i, col in enumerate(cols_to_plot):
        try:
            sns.ecdfplot(data=real, x=col, ax=axes[i], color='black', label='Real', linestyle='--')
            sns.ecdfplot(data=ori, x=col, ax=axes[i], color='#d62728', label='TabDDPM')
            sns.ecdfplot(data=new, x=col, ax=axes[i], color='#2ca02c', label='TAME')
            axes[i].set_title(f'CDF: {col}')
            if i == 0: axes[i].legend()
        except:
            pass
    
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_cdf.pdf'))
    plt.close()
    print(f"[{dataset_name}] CDF saved.")

# ==========================================
# 3. Categorical Bars (Fixing the Bug)
# ==========================================
def plot_categorical_bars(real, ori, new, dataset_name, save_dir):
    # 1. 预处理
    _, _, cat_cols = preprocess_simple(real)
    if not cat_cols: return
    
    # 只取前4个特征
    cols_to_plot = cat_cols[:min(len(cat_cols), 4)]
    n_total = len(cols_to_plot)
    
    # 2. 布局策略：
    # 如果只有 1-2 个特征 (如 Cardio 可能很少)，强制用单行，不要换行
    # 如果有 3-4 个特征，用 2x2
    if n_total <= 2:
        n_cols = n_total
        n_rows = 1
    else:
        n_cols = 2
        n_rows = math.ceil(n_total / n_cols)
    
    # 3. 关键修改：减小高度，增加宽度比例
    # 之前的 5 * n_rows 太高了，改为 3.5 * n_rows
    # 宽度设为 12 或 14 都可以
    fig_height = 3.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, fig_height))
    
    # 设置绘图风格，字体适中
    sns.set_context("paper", font_scale=1.4)
    
    # 展平 axes
    if n_total > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        
        # 计算频率
        def get_freq(df, c):
            if c not in df.columns: return pd.DataFrame()
            val_cnt = df[c].value_counts(normalize=True)
            return pd.DataFrame({'Category': val_cnt.index, 'Frequency': val_cnt.values})

        r_freq = get_freq(real, col); r_freq['Source'] = 'Real'
        o_freq = get_freq(ori, col); o_freq['Source'] = 'TabDDPM'
        n_freq = get_freq(new, col); n_freq['Source'] = 'TAME'
        
        combined = pd.concat([r_freq, o_freq, n_freq], ignore_index=True)
        
        # 排序
        if not r_freq.empty:
            order = r_freq.sort_values('Frequency', ascending=False)['Category'].tolist()[:10]
        else:
            order = combined['Category'].unique().tolist()[:10]
        
        # 4. 关键修改：解决柱子太宽/看不清差距的问题
        # width=0.5 让柱子变瘦，看起来更精致
        sns.barplot(data=combined, x='Category', y='Frequency', hue='Source', 
                    ax=ax, order=order, palette=['#d6d6d6', '#7bcc7b', '#eda1ac'],
                    edgecolor='black', linewidth=0.6, width=0.6) 
        
        # 5. Y轴优化：
        # 对于二分类或分布极端的，Log坐标轴有时候会掩盖差异。
        # 如果最小值 > 0.01，可以考虑不用Log，或者手动设置下限。
        ax.set_yscale('log')
        # 设置一个合理的下限，防止柱子看起来“悬空”太高
        ax.set_ylim(bottom=1e-3) 
        
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Frequency (Log)' if i % n_cols == 0 else '')
        
        # 旋转X轴标签，防止重叠
        ax.tick_params(axis='x', rotation=25, labelsize=10)
        
        # 图例处理
        if i == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=9, ncol=1)
        else:
            if ax.get_legend(): ax.get_legend().remove()

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name}_categorical.pdf')
    # 6. 去除白边
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[{dataset_name}] Saved optimized bar chart.")

# ==========================================
# 4. Bivariate & Boxplots
# ==========================================
def plot_bivariate_and_box(real, ori, new, dataset_name, save_dir):
    real_num = real.select_dtypes(include=[np.number])
    if real_num.shape[1] >= 2:
        # Bivariate
        corr = real_num.corr().abs()
        np.fill_diagonal(corr.values, 0)
        r, c = np.unravel_index(corr.values.argmax(), corr.shape)
        f1, f2 = real_num.columns[r], real_num.columns[c]
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # 限制范围避免KDE报错
        xmin, xmax = real_num[f1].quantile(0.01), real_num[f1].quantile(0.99)
        ymin, ymax = real_num[f2].quantile(0.01), real_num[f2].quantile(0.99)
        
        for data, axi, color, title in zip([real, ori, new], ax, ['Blues', 'Reds', 'Greens'], ['Real', 'TabDDPM', 'TAME']):
            try:
                sns.kdeplot(data=data, x=f1, y=f2, ax=axi, fill=True, cmap=color, thresh=0.05)
                axi.set_xlim(xmin, xmax)
                axi.set_ylim(ymin, ymax)
                axi.set_title(title)
            except: pass
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_bivariate.pdf'))
        plt.close()
        print(f"[{dataset_name}] Bivariate saved.")

    # Boxplots
    cols = real_num.columns[:6]
    if len(cols) > 0:
        fig, axes = plt.subplots(int(np.ceil(len(cols)/2)), 2, figsize=(12, 4*int(np.ceil(len(cols)/2))))
        axes = axes.flatten()
        for i, col in enumerate(cols):
            data_plot = [real[col].dropna(), ori[col].dropna(), new[col].dropna()]
            axes[i].boxplot(data_plot, labels=['Real', 'Base', 'TAME'], patch_artist=True)
            axes[i].set_title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_boxplots.pdf'))
        plt.close()
        print(f"[{dataset_name}] Boxplots saved.")

def process_full_suite(root_dir):
    subdirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    for subdir in subdirs:
        name = os.path.basename(subdir)
        train_path = os.path.join(subdir, "train.csv")
        ori_files = glob.glob(os.path.join(subdir, "*_ori.csv"))
        new_files = glob.glob(os.path.join(subdir, "*_new.csv"))
        
        if not (os.path.exists(train_path) and ori_files and new_files): continue
        
        print(f"\nProcessing {name}...")
        df_real = load_and_clean_data(train_path)
        df_ori = load_and_clean_data(ori_files[0])
        df_new = load_and_clean_data(new_files[0])
        
        if df_real is None: continue

        save_dir = os.path.join(subdir, "analysis_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # plot_manifold_learning(df_real, df_ori, df_new, name, save_dir)
        # plot_cdf_curves(df_real, df_ori, df_new, name, save_dir)
        plot_categorical_bars(df_real, df_ori, df_new, name, save_dir)
        # plot_bivariate_and_box(df_real, df_ori, df_new, name, save_dir)

if __name__ == "__main__":
    ROOT_DIR = "test_data" # 请根据实际情况修改
    if os.path.exists(ROOT_DIR):
        process_full_suite(ROOT_DIR)