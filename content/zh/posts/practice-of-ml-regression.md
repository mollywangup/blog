---
title: "机器学习例子 - 回归问题"
date: 2023-08-04T08:09:47Z
draft: false
description: 直接使用 sklearn.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Machine Learning
- sklearn
- Regression
categories:
- Practice
libraries:
- mathjax
---

## 线性回归

### 一元线性回归

以下示例来源于 sklearn 的糖尿病数据集。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集：仅取其中一个特征，并拆分训练集/测试集（7/3）
features, target = load_diabetes(return_X_y=True)
feature = features[:, np.newaxis, 2]
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=8)
print('特征数量：{} 个（原始数据集共 {} 个特征）\n总样本量：共 {} 组，其中训练集 {} 组，测试集 {} 组'.format(feature.shape[1], features.shape[1], target.shape[0], X_train.shape[0], X_test.shape[0]))

# 创建线性回归模型并拟合数据
model = LinearRegression()
model.fit(X_train, y_train)

# 获取模型参数
w = model.coef_
b = model.intercept_
print('模型参数：w={}, b={}'.format(w, b))

# 衡量模型性能：R2 和 MSE
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# R2（决定系数，1最佳），计算等同于 r2_score(y_true, y_pred)
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)
# MSE（均方误差）
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print('模型性能：\n  训练集：R2={:.3f}, MSE={:.3f}\n  测试集：R2={:.3f}, MSE={:.3f}'.format(r2_train, mse_train, r2_test, mse_test))

# 绘图
plt.title('LinearRegression (One variable)')
plt.scatter(X_train, y_train, color='red', marker='X')
plt.plot(X_test, y_pred, linewidth=3)
plt.legend(['training points', 'model: $y={:.2f}x+{:.2f}$'.format(w[0], b)])
plt.savefig('LinearRegression_diabetes.svg')
```
<img src='https://user-images.githubusercontent.com/46241961/273402064-fdd2a737-a691-45bc-8c17-6f921e02d487.svg' alt='一元线性回归-糖尿病数据集' width=80%>

### 多元线性回归

以下示例来源于 sklearn 的糖尿病数据集，选取了所有的特征，并对比了普通最小二乘/Lasso/Ridge 三种回归模型的性能。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集：取所有特征，并拆分训练集/测试集（7/3）
features, target = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=8)
print('特征数量：{} 个\n总样本量：共 {} 组，其中训练集 {} 组，测试集 {} 组'.format(features.shape[1], target.shape[0], X_train.shape[0], X_test.shape[0]))

def _models(alpha=1):
    lr = LinearRegression().fit(X_train, y_train) # 第一种：普通最小二乘回归
    lasso = Lasso(alpha=alpha).fit(X_train, y_train) # 第二种：Lasso/L1/套索回归
    ridge = Ridge(alpha=alpha).fit(X_train, y_train) # 第三种：Ridge/L2/岭回归
    return lr, lasso, ridge

# 对比四组 alpha 取值
alphas_list = [0.05, 0.1, 0.5, 1]

for i in range(len(alphas_list)):
    alpha = alphas_list[i]
    print('\n======== alpha={} ========'.format(alpha))
    
    # 对比三种线性模型
    models = _models(alpha=alpha)
    for model in models:    
        # 模型参数
        w = model.coef_
        b = model.intercept_

        # 模型性能：R2 和 MSE
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)
        mse_train = mean_squared_error(y_train, model.predict(X_train))
        mse_test = mean_squared_error(y_test, model.predict(X_test))
    
        # 打印
        model_name = model.__class__.__name__
        print('{}：\n  模型参数：w={}, b={:.3f}\n  训练集：R2={:.3f}, MSE={:.3f}\n  测试集：R2={:.3f}, MSE={:.3f}'.format(model_name, w, b, r2_train, mse_train, r2_test, mse_test))
```