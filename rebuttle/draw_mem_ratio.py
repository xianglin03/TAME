import matplotlib.pyplot as plt
import numpy as np

def remove_outliers_iqr(x, y, k=1.5):
    """
    Remove outliers based on IQR (Interquartile Range) for both x and y.
    
    Args:
        x (array-like): X values.
        y (array-like): Y values.
        k (float): IQR multiplier, typically 1.5 or 2.0.
    
    Returns:
        (x_filtered, y_filtered): Arrays with outliers removed.
    """
    x = np.array(x)
    y = np.array(y)

    x_q1, x_q3 = np.percentile(x, [25, 75])
    y_q1, y_q3 = np.percentile(y, [25, 75])
    x_iqr = x_q3 - x_q1
    y_iqr = y_q3 - y_q1

    mask = (
        (x >= x_q1 - k * x_iqr) & (x <= x_q3 + k * x_iqr) &
        (y >= y_q1 - k * y_iqr) & (y <= y_q3 + k * y_iqr)
    )
    return x[mask], y[mask]


def plot_mem_vs_co_ratio(file_path, poly_degree=2, iqr_k=1.5):
    """
    Reads a file, removes outliers (IQR), and plots a fitted curve between memorization and co_ratio.

    Args:
        file_path (str): Path to the input file.
        poly_degree (int): Degree of polynomial for fitting (default=2).
        iqr_k (float): IQR multiplier for outlier removal (default=1.5).
    """
    mem_values = []
    co_ratio_values = []

    # Step 1: 读取文件
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("step:"):
                parts = line.split(", ")
                try:
                    mem = float(parts[1].split(": ")[1])
                    co_ratio = float(parts[4].split(": ")[1])
                    mem_values.append(mem)
                    co_ratio_values.append(co_ratio)
                except Exception:
                    continue

    mem_values = np.array(mem_values)
    co_ratio_values = np.array(co_ratio_values)

    # Step 2: 使用IQR去除离群点
    mem_clean, co_ratio_clean = remove_outliers_iqr(mem_values, co_ratio_values, k=iqr_k)
    print(f"✅ Removed {len(mem_values) - len(mem_clean)} outliers using IQR (k={iqr_k})")

    # Step 3: 多项式拟合
    coeffs = np.polyfit(mem_clean, co_ratio_clean, deg=poly_degree)
    poly_func = np.poly1d(coeffs)

    # Step 4: 生成平滑曲线
    x_fit = np.linspace(min(mem_clean), max(mem_clean), 300)
    y_fit = poly_func(x_fit)

    # Step 5: 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(mem_clean, co_ratio_clean, color='blue', alpha=0.6, label='Filtered Data (IQR)')
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Polynomial Fit (deg={poly_degree})')
    plt.xlabel("Memorization", fontsize=14)
    plt.ylabel("Correlation Ratio", fontsize=14)
    plt.title("Relationship between Memorization and Correlation Ratio", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mem_vs_co_ratio_iqr_fit.png", dpi=300)
    plt.close()

    print(f"✅ Fitted coefficients (degree={poly_degree}): {coeffs}")
    print("📊 Saved figure as mem_vs_co_ratio_iqr_fit.png")


# Example usage
if __name__ == "__main__":
    file_path = "eval/result/co_ratio_a1_a2.txt"
    plot_mem_vs_co_ratio(file_path, poly_degree=2, iqr_k=1.5)
