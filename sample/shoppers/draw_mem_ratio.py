import pandas as pd
import matplotlib.pyplot as plt

def plot_mem_ratio(csv_file_list):
    
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ['m', 'green', 'red', 'orange', 'm', 'y', 'k']  # 示例颜色列表
    
    # 画离散类型的图像
    # plt.subplot(1, 2, 1)
    for i, csv_file in enumerate(csv_file_list):
        # 读取csv文件
        df = pd.read_csv(csv_file)
        # 从文件名中提取数字
        size = csv_file.split('/')[-1].split('_')[-1].replace('.csv', '')
        plt.plot(df['Epoch'], df['cat_mem'], 
                label=size,  # 只使用数字作为标签
                color=colors[i],
                linewidth=2,
                alpha=0.8)
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('cat ratio')
        plt.title('shoppers-cat')
        
    # 修改图例样式
    legend = plt.legend(title='Train Size Percent',  # 添加标题
                       fontsize=12,
                       frameon=True,
                       facecolor='white',
                       edgecolor='gray',
                       loc='lower right')
    legend.get_title().set_fontsize(12)  # 设置标题字体大小
    # legend.get_title().set_fontweight('bold')  # 设置标题字体粗细
    
    # 画连续类型的图像
    # plt.subplot(1, 2, 2)
    # 假设 csv_file_list 和 colors 已经定义

    # 设置图像大小

    # for i, csv_file in enumerate(csv_file_list):
    #     # 读取csv文件
    #     df = pd.read_csv(csv_file)
    #     # 绘制图像
    #     label = csv_file.split('/')[-1].replace('.csv', '').replace('shoppers_', '') + '%'
    #     plt.plot(df['Epoch'], df['num_mem'], label=label, color=colors[i % len(colors)])

    # plt.grid(True)
    # plt.xlabel('Epoch')
    # plt.ylabel('Num ratio')
    # plt.title('∑0~n Norm(min_i)/Norm(avg_i)) / n')
    # plt.legend(fontsize=12)

    # 调整布局以减少空白
    plt.tight_layout(pad=5)
    plt.savefig('sample/shoppers/mem_ratio.png')
    # plt.show()
    
    
# 调用函数绘制图像
csv_file_list = [
    'sample/shoppers/size_10.csv',
    'sample/shoppers/size_30.csv',
    'sample/shoppers/size_50.csv',
    'sample/shoppers/size_100.csv',
]
plot_mem_ratio(csv_file_list)