import pandas as pd

# 加载数据
df = pd.read_csv('Processed_DatasetAll.csv')

# 找到缺失值的位置
missing_values = df.isnull()

# 遍历数据帧，打印缺失值的位置
for col in missing_values.columns:
    if missing_values[col].any():
        rows_with_missing = missing_values.index[missing_values[col]].tolist()
        print(f"在列 '{col}' 中缺失数据的行为: {rows_with_missing}")
