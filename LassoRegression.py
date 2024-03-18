import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 检查是否存在Images文件夹，不存在则创建
images_dir = "Images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# 加载数据
data = pd.read_csv('Processed_DatasetAll.csv')

# 移除不需要的列
data = data.drop(['publicationNumber'], axis=1)

data.fillna(data.mean(), inplace=True)

# 设置目标变量列表
target_vars = ['IPCNumbers']

# 初始化一个空DataFrame来存储模型评价结果
evaluation_metrics = pd.DataFrame()

# 初始化一个空Series来存储所有特征的系数
all_feature_coef = pd.Series()

for target in target_vars:
    # 分离特征和目标变量
    X = data.drop(target_vars, axis=1)
    y = data[target]
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3407)
    
    # 使用带交叉验证的Lasso模型进行训练
    lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000000)
    lasso_cv.fit(X_train, y_train)
    
    lasso = Lasso(alpha=lasso_cv.alpha_)
    lasso.fit(X_train, y_train)
    
    y_pred_all = lasso_cv.predict(X_scaled)
    y_pred_train = lasso_cv.predict(X_train)
    y_pred_test = lasso_cv.predict(X_test)
    
    # 计算评价指标
    evaluation_metrics[f'{target}-Alpha'] = [lasso_cv.alpha_]
    
    evaluation_metrics[f'{target}-Train R^2'] = [r2_score(y_train, y_pred_train)]
    evaluation_metrics[f'{target}-Test R^2'] = [r2_score(y_test, y_pred_test)]
    evaluation_metrics[f'{target}-All R^2'] = [r2_score(y, y_pred_all)]
    evaluation_metrics[f'{target}-Train MSE'] = [mean_squared_error(y_train, y_pred_train)]
    evaluation_metrics[f'{target}-Test MSE'] = [mean_squared_error(y_test, y_pred_test)]
    evaluation_metrics[f'{target}-All MSE'] = [mean_squared_error(y, y_pred_all)]
    evaluation_metrics[f'{target}-Train RMSE'] = [np.sqrt(mean_squared_error(y_train, y_pred_train))]
    evaluation_metrics[f'{target}-Test RMSE'] = [np.sqrt(mean_squared_error(y_test, y_pred_test))]
    evaluation_metrics[f'{target}-All RMSE'] = [np.sqrt(mean_squared_error(y, y_pred_all))]
    
    # 保存特征系数
    feature_coef = pd.DataFrame({
        'Feature': X.columns,
        'Coefficients (B)': lasso.coef_
    })
    
    feature_coef.to_csv('./Coefficients/'+target+'FeaturesCoefficients.csv', header=True)
    
    # 确定最长Feature名称的长度
    max_length = feature_coef['Feature'].str.len().max()

    # 将所有Feature名称统一为相同的长度
    feature_coef['Feature'] = feature_coef['Feature'].apply(lambda x: x.ljust(max_length))

    
    feature_coef['Representative Name'] = feature_coef.apply(lambda x: ' ' if x['Coefficients (B)'] == 0 else x['Feature'], axis=1)
    
    # 计算绝对值的系数并排序，选取Top 10特征
    top_features_abs_sorted = feature_coef['Coefficients (B)'].abs().sort_values(ascending=False).head(10)
    top_features = feature_coef.loc[top_features_abs_sorted.index]  # 选取Top 10特征

    # 对选定的特征按原始系数值排序，以确保图表反映实际的正负值和大小
    top_features_sorted = top_features.sort_values(by='Coefficients (B)')

    # 配置绘图样式和字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = plt.Normalize(-0.4, .4)
    cmap = plt.cm.coolwarm

    # 计算颜色
    colors = cmap(norm(top_features_sorted['Coefficients (B)'].values))

    # 手动绘制每个条形
    for i, (coef, name) in enumerate(zip(top_features_sorted['Coefficients (B)'], top_features_sorted['Representative Name'])):
        ax.barh(y=i, width=coef, color=colors[i])

    # 设置y轴的刻度标签
    ax.set_yticks(range(len(top_features_sorted)))
    ax.set_yticklabels(top_features_sorted['Representative Name'])

    # 设置其他绘图属性
    plt.xlabel('Coefficients', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title(f'Top Features Selected by Lasso for {y.name}', fontsize=16)
    plt.tight_layout()

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Coefficient Value')

    # 保存图像
    plt.savefig(f'{images_dir}/FeatureCoefficients{y.name}.jpg', format='jpg')
    plt.close()



# 保存模型评价结果
evaluation_metrics = evaluation_metrics.T
evaluation_metrics.columns = ['Value']
evaluation_metrics.to_csv('ModelEvaluationMetrics.csv')
