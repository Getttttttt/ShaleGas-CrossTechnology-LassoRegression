import pandas as pd
import chardet    
import numpy as np

def ipc_count(ipc_value):
    if pd.isnull(ipc_value) or ipc_value == '-':
        return 0
    else:
        return len(ipc_value.split(' | '))

# 自定义函数处理多国家/地区的虚拟变量转换
def expand_multivalue_columns(row, col_prefix, all_possible_values):
    for value in all_possible_values:
        if value in row:
            row[f"{col_prefix}_{value}"] = 1
        else:
            row[f"{col_prefix}_{value}"] = 0
    return row

if __name__ == '__main__':
    encoding = 'gb2312'  # 假设已通过chardet检测得到
    df = pd.read_csv('./DatasetAll.csv', encoding=encoding)

    # 步骤1: 修改IPC数量计算逻辑
    df['IPCNumbers'] = df['IPC'].apply(ipc_count)

    # 步骤2: 转化具有多个值的列为虚拟变量
    multivalue_columns = ['priorityCountryRegion', 'supplementaryPriorityCountryRegion', 'legalStatusEvent', 'patentType']
    for col in multivalue_columns:
        # 获取所有可能的值（分隔并去重）
        all_possible_values = set(val for sublist in df[col].dropna().unique() for val in sublist.split(' | '))
        all_possible_values.discard('-')  # 移除表示空值的'-'
        # 应用转换
        df = df.apply(expand_multivalue_columns, args=(col, all_possible_values), axis=1)

    # 步骤3: 转化日期列，处理异常值
    for date_col in ['applicationDate', 'publicationDate']:
        df[date_col] = pd.to_datetime(df[date_col].replace('-', np.nan), errors='coerce')
        # 将日期转换为距离某个固定日期的天数
        df[date_col] = (df[date_col] - pd.Timestamp('1970-01-01')).dt.days

    # 保存处理后的数据
    df.to_csv('./Processed_DatasetAllAnother.csv', index=False)