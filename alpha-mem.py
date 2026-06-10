import pandas as pd

df = pd.read_csv("alpha-mem.csv")

# 打印列标签以检查正确的列名
print(df.columns)

# 假设第一列的列名是 'column_name'
df = df.sort_values(by='alpha')
df.to_csv("alpha-mem.csv", index=False)