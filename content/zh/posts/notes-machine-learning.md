---
title: "学习笔记：吴恩达机器学习"
date: 2023-08-04T08:09:47Z
draft: false
description: 线性回归，多项式回归，逻辑回归，SVM，朴素贝叶斯等。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Machine Learning
- sklearn
categories:
- Notes
libraries:
- mathjax
---

本笔记基于以下学习资料（侧重实际应用）：
> 入门机器学习：[(强推|双字)2022吴恩达机器学习Deeplearning.ai课程](https://www.bilibili.com/video/BV1Pa411X76s/)
> Python 代码库：[scikit-learn 官网](https://scikit-learn.org/stable/index.html)
> 复习线性代数：3Blue1Brown 的 [线性代数的本质 - 系列合集](https://www.bilibili.com/video/BV1ys411472E/)

## 统一口径

### 术语

- 特征（`feature`）：指输入变量；
- 标签（`label`）：指输出值，可以是实际值（`target`），也可以是预测值（`prediction`）；
- 训练集（`training set`）：指用于训练模型的数据集；
- 测试集（`test set`）：指用于测试模型的数据集；
- 训练示例（`training example`）：指训练集中的一组数据；
- 训练模型（`model`）：指拟合函数；
- 模型参数（`parameters`）：调整模型的本质是调整模型参数；
- 特征工程（`feature engineering`）：指从原始数据中选择、提取和转换最相关的若干个特征，以提高机器学习模型的准确性；

### 数学表达式

约定如下：
1. `m` 个训练示例，`n` 个特征；
2. 向量是一维数组，使用小写字母表示，且默认为列向量；矩阵是二维数组，使用大写字母表示；
3. 非代码部分从 `1` 开始计数；

<br>具体符号：
- $x$ 表示特征变量，$w$ 表示回归系数，$y$ 表示实际值，$\hat{y}$ 表示预测值，都是列向量；
- $X$ 表示训练示例组成的矩阵，$(X|y)$ 表示带标签的训练示例组成的增广矩阵。注意区分：
  - $x^{(i)}$ 表示第 $i$ 个训练示例的特征，是个列向量（矩阵 $X$ 的第 $i$ 行）；
  - $x_j$ 表示第 $j$ 个特征，是个列向量（矩阵 $X$ 的第 $j$ 列）；
  - $x_j^{(i)}$ 表示第 $i$ 个训练示例的第 $j$ 个特征，是个标量；
  - $y^{(i)}$ 和 $\hat{y}^{(i)}$ 分别表示第 $i$ 个训练示例的实际值和预测值，都是标量；

$$
x = \begin{bmatrix}x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{bmatrix}
\space
w = \begin{bmatrix}w_1 \\\\ w_2 \\\\ \vdots \\\\ w_n \end{bmatrix}
\space
y = \begin{bmatrix}y^{(1)} \\\\ y^{(2)} \\\\ \vdots \\\\ y^{(m)} \end{bmatrix}
\space
\hat{y} = \begin{bmatrix}\hat{y}^{(1)} \\\\ \hat{y}^{(2)} \\\\ \vdots \\\\ \hat{y}^{(m)} \end{bmatrix}
\space
$$

$$
(X|y) = \left [
\begin{array}{cccc|c}
  x_1^{(1)} & x_2^{(1)} & \dots & x_n^{(1)} & y^{(1)} \\\\ 
  x_1^{(2)} & x_2^{(2)} & \dots & x_n^{(2)} & y^{(2)} \\\\ 
  \vdots & \vdots & \ddots & \vdots & \vdots \\\\ 
  x_1^{(m)} & x_2^{(m)} & \dots & x_n^{(m)} & y^{(m)} 
\end{array}
\right ]
\space
x^{(i)} = \begin{bmatrix}x_1^{(i)} \\\\ x_2^{(i)} \\\\ \vdots \\\\ x_n^{(i)} \end{bmatrix}
\space
x_j = \begin{bmatrix}x_j^{(1)} \\\\ x_j^{(2)} \\\\ \vdots \\\\ x_j^{(m)} \end{bmatrix}
$$

## 概述

机器学习解决的问题是：给定训练集，通过机器学习算法生成最佳训练模型，最终应用于预测新特征对应的输出值。

### 分类

根据训练集中包含标签的情况，机器学习可分为以下三类（本文仅涉及前两类）：

- 监督学习（Supervised Learning）：训练集中包含标签，分为：
  - **`回归（Regression）`**
  - **分类（Classification）**
- 无监督学习（Unsupervised Learning）：训练集中不包含标签，分为：
  - **聚类（Clustering）**
  - **降维（Dimensionality reduction）**
- 强化学习（Reinforcement Learning）：有延迟和稀疏的反馈标签；

### 核心思路

#### 对于监督学习

划重点：**确定训练模型和成本函数，找到一组模型参数，使得成本函数最小化**。

Step1：确定训练模型（Model），其中模型包括若干个特征和若干个模型参数；
Step2：确定成本函数（Cost function），用于衡量预测值与实际值之间的差异程度，是关于若干个模型参数的函数；
Step3：求解**使得成本函数最小化**（Goal）的一组参数值，其中可使用梯度下降算法；

关于损失函数和梯度下降的具体数学原理，详见结尾的附录部分。

#### 对于无监督学习


## 监督学习算法

如果训练集中**包含标签**，则属于监督学习，即 `(features, labels) -> Model` 问题。

监督学习共以下两类学习任务：
- 回归：预测值为**连续型**，可应用于趋势预测、价格预测、流量预测等；
- 分类：预测值为**离散型**，可应用于构建用户画像、用户行为预测、图像识别分类等；

### 线性回归

线性回归（Linear Regression），解决线性的**回归**问题。包含一元线性回归和多元线性回归两类情况。

<img src='https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png' alt='One variable Linear Regression Example' width='60%'>

#### 原理

目标：求解一组模型参数 $(w,b)$ 使得成本函数 $J$ 最小化。

##### 模型

$$ 
f_{w,b}(x) = w \cdot x + b = \begin{bmatrix}w_1 \\\\ w_2 \\\\ \vdots \\\\ w_n \end{bmatrix} \cdot \begin{bmatrix}x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{bmatrix} + b = \sum_{j=1}^{n} w_j \cdot x_j + b \tag{Model}
$$

说明：当 n = 1 时，对应一元线性回归；当 n >= 2 时，对应多元线性回归；

##### 模型参数

$w = \begin{bmatrix}w_1 \\\\ w_2 \\\\ \vdots \\\\ w_n \end{bmatrix}$，回归系数，分别对应 n 个特征的权重（weights）或系数（coefficients）；
$b$：偏差（bias）或截距（intercept）；

##### 成本函数

$$ J(w,b) = \frac{1}{2} MSE = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1} $$

$$ J(w,b) = \frac{1}{2} MSE + \alpha {\lVert w \rVert}_1 = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \sum_{j=1}^{n} {\lvert w_j \rvert} \tag{2} $$

$$ J(w,b) = \frac{1}{2} MSE + \alpha {\lVert w \rVert}_2^2 = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \sum_{j=1}^{n} w_j^2 \tag{3} $$

说明：
1. 回归系数 $w$ 在模型 $f_{w,b}(x)$ 中是参数，在成本函数 $J(w,b)$ 中属于变量；
2. 成本函数 $(1)$ 对应`普通最小二乘回归（OLS）`（Ordinary Least Squares）；
3. 成本函数 $(2)$ 对应 `套索回归（Lasso）`，是在最小二乘的基础上，添加了回归系数的 `L1 范数`作为惩罚项，目的是进行**特征选择**（即让 $w$ 中的部分取零）；
4. 成本函数 $(3)$ 对应 `岭回归（Ridge）`，在最小二乘的基础上，添加了回归系数的 `L2 范数`作为惩罚项，目的是**防止过拟合**；
5. $\alpha$，伸缩系数，非负标量，为了控制惩罚项的大小。

##### 目标

$$ \min_{w,b} J(w,b) \tag{Goal} $$

#### 示例

##### 一元线性回归

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

##### 多元线性回归

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

### 多项式回归

多项式回归（Polynomial regression），解决非线性的**回归**问题。

核心思想是将非线性问题转化为线性问题。

#### 原理

目标：求解一组模型参数 $(\vec{w},b)$ 使得成本函数 $J$ 最小化。


$$ f_{w,b}(x) = w_1x + w_2x^2 + b \tag{Model1} $$
$$ f_{w,b}(x) = w_1x + w_2x^2 + w_3x^3 + b \tag{Model2} $$
$$ f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_1x_2 + w_4x_1^2 + w_5x_2^2 + b \tag{Model3} $$

$$ J(w,b) =  \tag{Cost function}$$

$$ \min_{\vec{w},b} J(w,b) \tag{Goal} $$

其中，模型参数如下:
- $w$：分别对应各项的权重（weights）或系数（coefficients）；
- $b$：偏差（bias）或截距（intercept）；

说明：上述 Model1、Model2、Model3 依次是一元二次多项式、一元三次多项式、二元二次多项式。

#### 示例

以下示例为一元三次多项式。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.RandomState(0)

# 数据集
x = np.linspace(-3, 7, 10)
y = np.power(x, 3) + np.power(x, 2) + x + 1 + rng.randn(1)
X = x[:, np.newaxis]

# 绘制训练集
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='red', marker='X')
legend_names = ['training points']

# 多项式特征的线性回归模型
for degree in range(10):
    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # 创建线性回归模型：X_poly 与 y 为线性关系
    model = LinearRegression()
    model.fit(X_poly, y)

    # 使用模型预测
    y_pred = model.predict(X_poly)
    
    # 获取模型参数和性能指标
    w = model.coef_
    b = model.intercept_
    mse = mean_squared_error(y, y_pred) # 均方误差
    r2 = r2_score(y, y_pred) # 决定系数
    print('当 degree 取 {} 时，mse={}, r2={}, 模型参数 w={}, b={:.4f}'.format(degree, round(mse, 3), r2, w, b))

    # 绘图
    plt.plot(X, y_pred)
    legend_names.append('degree {}: mse {}, r2 {}'.format(degree, round(mse, 3), r2))

# 添加图例
plt.legend(legend_names)
plt.savefig('PolynomialFeatures_LinearRegression.svg')
```
<img src='https://user-images.githubusercontent.com/46241961/272204746-6f8c1665-2d34-40fc-ae86-29e8d0d7a942.svg' alt='PolynomialFeatures_LinearRegression' width='80%'>

### 逻辑回归

logistic regression，解决**分类**问题。

（binary classification）

true: 1, positive class
false: 0, negative class

logistic/sigmoid function

$$
z = \vec{w} \cdot \vec{x} + b \\\\
g(z) = \frac{1}{1+e^{-z}}
$$

$$
g(z) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}} = P(y=1|x;\vec{w},b)
$$

### 决策树

Decison tree，解决**分类**问题。

- 根节点：无入多出
- 内部节点：一入多出
- 叶子结点：一入无出

熵

基尼系数

### 随机森林

Random forest，解决**分类**问题。

回归问题：求均值
分列问题：求众数

### SVM

支持向量机，解决**分类**问题。

属于线性分类器。非线性问题，可通过 kernal SVM 解决（映射到高维）；

超平面：
- 决策分界面（decision boundary）
- 边界分界面（margin boundary）

Hard-margin SVM
Soft-margin SVM：加入了容错率

### 朴素贝叶斯

Nbayes，解决**分类**问题。

### K近邻

KNN (K-Nearest Neighbors)，解决**分类+回归**问题。

### 神经网络

Neural Network，解决**分类+回归**问题。

## 无监督学习算法

### 概述

训练集中**不包含标签**，则属于无监督学习，即 `(features) -> Model`.

共分为两大类任务：
- 聚类（Clustering）
- 降维（Dimensionality reduction）

### K-means

解决**聚类**问题。

- K-means：将 n 个点分为 k 个簇，使得簇内具有较高的相似度，簇间具有较低的相似度；（欧氏距离）
- 
### DBSCAN

解决**聚类**问题。

- DBSCAN（密度聚类）：将 n 个点分为三类，然后删除噪音点；（曼哈顿距离）
  - 核心点：在半径 eps（两个样本被看做邻域的最大举例） 内的点的个数超过 min_samples（簇的样本数）；
  - 边界点：在半径 eps 内的点的个数不超过 min_samples，但落在核心点的邻域内；
  - 噪音点：既不是核心点，也不是边界点；

### PCA

解决**降维**问题。

- PCA：主成分分析；

## 模型评估

模型评估的目标是**选出泛化能力最优的模型**。

### 评估方法

留出法（Hold-out）：拆分训练集和测试集

交叉验证法（Cross Validation）：将数据集分成 N 块，使用 N-1 块进行训练，再用最后一块进行测试；

自助法（Bootstrap）：

### 回归评估指标

#### MAE

MAE（Mean Absolute Error），平均绝对误差。

$$ MAE = \frac{1}{m} \sum_{i=1}^{m} \lvert \hat{y}^{(i)} - y^{(i)} \rvert $$

#### MAPE

MAPE（Mean Absolute Percentage Error），平均绝对百分误差。

$$ MAPE = \frac{100}{m} \sum_{i=1}^{m} \lvert \frac{y^{(i)} - \hat{y}^{(i)}}{y^{(i)}} \rvert $$

#### MSE

MSE（Mean Squared Error），均方误差。

$$ MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$

应用：常用作线性回归模型的成本函数，但机器学习中经验使用 $\frac{1}{2} MSE$，即除以 `2m` 而不是 ~~`m`~~，目的是在求导数/偏导数时消去常数2，并不影响结果。

#### RMSE

RMSE（Root Mean Square Error），均方根误差。

$$ RMSE = \sqrt{MSE} $$

#### R<sup>2</sup>

R<sup>2</sup> (coefficient of determination)，决定系数。衡量**总误差（客观存在且无关回归模型）中可以被回归模型解释的比例**，即拟合程度。

$$ R^2 = \frac{SSR}{SST} = 1- \frac{SSE}{SST} $$

说明：
当 $R^2 \to 1$ 时，表明拟合程度越好，因为此时 SSR 趋向于 SST（或 SSE 趋向于 0）；
当 $R^2 \to 0$ 时，表明拟合程度越差，因为此时 SSR 趋向于 0（或 SSE 趋向于 SST）；

{{< expand "关于 SST/SSR/SSE">}}

{{< boxmd >}}
助记小技巧：**T** is short for total, **R** is short for regression, **E** is short for error.
{{< /boxmd >}}

SST (sum of squares total)，总平方和，用以衡量**实际值**相对**均值**的离散程度。SST 客观存在且与回归模型无关；

$$ SST = \sum_{i=1}^{m} (y^{(i)} - \bar{y})^2 $$

SSR (sum of squares due to regression)，回归平方和，用于衡量**预测值**相对**均值**的离散程度。当 SSR = SST 时，回归模型完美；

$$ SSR = \sum_{i=1}^{m} (\hat{y}^{(i)} - \bar{y})^2 $$

SSE (sum of squares error)，误差平方和，用于衡量**预测值**相对**实际值**的离散程度；

$$ SSE = \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$

且三者之间的关系是 $SST = SSR + SSE$.

{{< /expand >}}

<img src='https://user-images.githubusercontent.com/46241961/273468625-e2263610-af8d-4ada-9cf9-9c25eef6c3c3.svg' alt='LinearRegression_SST_SSR_SSE' width='80%'>

### 分类评估指标

#### 混淆矩阵

（confusion matrix）

用于分类模型的效果评估。以下以二分类模型为例：

| 预测/实际&nbsp;&nbsp;&nbsp; | Positive&nbsp;&nbsp;&nbsp; | Negative&nbsp;&nbsp;&nbsp; |
| ---------- | ---------- | ---------- |
| **Positive** | TP  | FP&nbsp;&nbsp;&nbsp; | 
| **Negative** | FN  | TN&nbsp;&nbsp;&nbsp; | 

- 准确率（accuracy）：指预测正确的比例，即 $\frac{TP+TN}{TP+TN+FP+FN}$
- 精确率（precision）：也称作查准率，指预测为正中实际为正的比例，即 $\frac{TP}{TP+FP}$
- 召回率（recall）：也称作查全率，指实际为正中预测为正的比例，即 $\frac{TP}{TP+FN}$
- F1：$\frac{2 \times	 精确率 \times 召回率}{精确率 + 召回率}$

#### ROC

[深入介紹及比較ROC曲線及PR曲線](https://medium.com/nlp-tsupei/roc-pr-%E6%9B%B2%E7%B7%9A-f3faa2231b8c)

用于分类模型的效果评估，以可视化的方式。

## 特征工程

### 特征缩放

特征缩放（Feature Scaling）是一种用于**标准化自变量或特征范围**的方法。

背景：不同特征之间的取值范围差异较大，导致梯度下降运行低效。特征缩放使得不同特征之间的取值范围差异，降低至可比较的范围。
  - 除上限，如 [200, 1000] -> [0.2, 1]

目标：为了使梯度下降运行的更快，最终提高模型训练性能。

经验值：
- 太大或者太小都需要：如[-0.001, 0.001]、[-100, 100]；
- 通常[-3, 3]范围内，不需要；

#### 均值归一化

Mean Normalization，与均值的差异 / 上下限的整体差异：

$$
x^{\prime} = \frac{x - \mu}{max(x) - min(x)}
$$

#### Z 分数归一化

Z-score normalization，与均值的差异 / 标准差：

$$
x^{\prime} = \frac{x - \mu}{\sigma}
$$

其中标准差（Standard Deviation）$\sigma$ 计算公式如下：

$$
\sigma = \sqrt{\frac{\sum {(x - \mu)}^2}{n}}
$$

## 恶补高数与线代

### 梯度

给定任意 $n$ 元**可微**函数 $$f(x_1, x_2,..., x_n)$$

则函数 $f$ 的**偏导数构成的向量**，称为梯度，记作 $grad f$ 或 $\nabla f$，即：

$$
grad f = \nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},..., \frac{\partial f}{\partial x_n})
$$

梯度的几何意义是，<mark>**梯度方向**是函数值上升最快的方向，反方向为下降最快的方向</mark>，因此可应用于求解多元函数的极值。

{{< expand "关于偏导数">}}

函数 $f$ 对自变量 $x_i$ 的偏导数，指保持其他自变量不变，当 $x_i$ 发生增量 $\Delta x_i$ 且趋向于零即 $\displaystyle \lim_{{\Delta x_i} \to 0} $ 时，函数 $f$ 的`瞬时变化率`：

$$ \frac{\partial f}{\partial x_i} = \lim_{{\Delta x_i} \to 0} \frac{\Delta f}{\Delta x_i} = \lim_{{\Delta x_i} \to 0} \frac{f(x_i + {\Delta x_i}, ...) - f(x_i, ...)}{\Delta x_i}
$$

注意，可微一定可导，即任意给定点的邻域内所有偏导数存在且连续。

{{< /expand >}}

### 向量乘法

点积（Dot product），也称作点乘、内积，运算结果是一个标量。

$$
\begin{bmatrix}a \\\\ b \\\\ c \end{bmatrix}
\cdot
\begin{bmatrix}d \\\\ e \\\\ f \end{bmatrix} =
ad + be + cf 
$$

叉积（Cross product），也称作叉乘、外积，运算结果是一个向量。

$$
\begin{bmatrix}a \\\\ b \\\\ c \end{bmatrix}
\times
\begin{bmatrix}d \\\\ e \\\\ f \end{bmatrix} =
\left|
  \begin{matrix}
  \vec{i} & \vec{j} & \vec{k} \\\\ 
  a & b & c \\\\
  d & e & f
  \end{matrix}
\right| =
(bf-ce)\vec{i} - (af-cd)\vec{j} + (ae-bd)\vec{k} = 
\begin{bmatrix}bf-ce \\\\ -(af-cd) \\\\ ae-bd \end{bmatrix}
$$

### 矩阵向量乘法

**矩阵是一组线性变换的组合**。

理解：将矩阵的列向量看作一组新的基向量 $(\vec{u}, \vec{v}, \vec{w})$（非严谨说法，帮助理解），则矩阵向量乘法的几何意义是，该向量在新基向量下的新向量，也就是发生了一次线性变化。

$$
\begin{bmatrix}a & b & c \\\\ d & e & f \\\\ g & h & i \end{bmatrix}
\begin{bmatrix}x \\\\ y \\\\ z \end{bmatrix} = 
x \begin{bmatrix}a \\\\ d \\\\ g \end{bmatrix} + 
y \begin{bmatrix}b \\\\ e \\\\ h \end{bmatrix} + 
z \begin{bmatrix}c \\\\ f \\\\ i \end{bmatrix} = 
\begin{bmatrix}ax+by+cz \\\\ dx+ey+fz \\\\ gx+hy+iz \end{bmatrix}
$$

特别的，当矩阵取单位矩阵时，该向量经过线性变化后，仍等于该向量。

$$
\begin{bmatrix}1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix}x \\\\ y \\\\ z \end{bmatrix} = 
x \begin{bmatrix}1 \\\\ 0 \\\\ 0 \end{bmatrix} + 
y \begin{bmatrix}0 \\\\ 1 \\\\ 0 \end{bmatrix} + 
z \begin{bmatrix}0 \\\\ 0 \\\\ 1 \end{bmatrix} = 
\begin{bmatrix}x \\\\ y \\\\ z \end{bmatrix}
$$

### 矩阵矩阵乘法

理解：多次线性变化的叠加。

## 附

两点之间距离的计算方式（相似度衡量）：
- 欧氏距离：差的平方和的平方根；
- 曼哈顿距离：差的绝对值的和；
- 马氏距离：？？协方差距离
- 余弦相似度（cosine similarity）：用两个向量夹角的余弦值衡量两个样本差异的大小；（越接近于1，说明夹角越接近于0，表明越相似）


一些术语概念：
- 方差：分散程度。样本和样本均值的差的平方和的均值；
- 协方差：线性相关性程度。若协方差为0则线性无关；
- 特征向量：矩阵的特征向量。数据集结构的非零向量；空间中每个点对应的一个坐标向量。

### 成本函数

损失函数（Loss function）用于衡量预测值与实际值之间的差异程度，一般使用 $L$ 表示：

$$ L(f_{w,b}(x^{(i)}), y^{(i)}) $$

成本函数（Cost function）也称作代价函数，用于评估模型的**拟合程度**。一般使用 $J$ 表示：

$$
J(w,b) = \displaystyle \frac{1}{m} \sum_{i=1}^{m} L(f_{w,b}(x^{(i)}), y^{(i)})
$$

#### MSE Cost Function


#### Logistic loss function

适用于逻辑回归模型。

$$
L(f_{w,b}(x^{(i)}), y^{(i)}) = 
\begin{cases}
-log\left(f_{w,b}(x^{(i)})\right) & if\ y^{(i)} = 1 \\\\
-log\left(1-f_{w,b}(x^{(i)})\right) & if\ y^{(i)} = 0 \\\\
\end{cases}
$$
即
$$
-y^{(i)}log(f_{w,b}(x^{(i)}) - (1-y^{(i)})log(f_{w,b}(x^{(i)})
$$

### 梯度下降

#### 梯度下降算法

梯度下降（Gradient Descent）是一种迭代优化算法，用于求解任意一个可微函数的**局部最小值**。在机器学习中，常用于**最小化成本函数**，即最大程度减小预测值与实际值之间的误差。即：

给定成本函数 $J(w_1,w_2,...,w_n)$，求解一组 $(w_1,w_2,...,w_n)$，使得
$$ \min_{w_1,w_2,...,w_n} J(w_1,w_2,...,w_n) $$

实现的核心原理：<mark>**沿着梯度反方向，函数值下降最快**。</mark>

选定初始位置 $(w_1,w_2,...,w_n)$，通过重复以下步骤，直至收敛，即可得到局部最小值的解：

$$
\begin{equation} 
  \begin{pmatrix}
    w_1 \\\\
    w_2 \\\\
    \vdots \\\\
    w_n \\\\
  \end{pmatrix}
    \rightarrow
  \begin{pmatrix}
    w_1 \\\\
    w_2 \\\\
    \vdots \\\\
    w_n \\\\
  \end{pmatrix}
    - \alpha
  \begin{pmatrix}
    \frac{\partial J}{\partial w_1} \\\\
    \frac{\partial J}{\partial w_2} \\\\
    \vdots \\\\
    \frac{\partial J}{\partial w_n} \\\\
  \end{pmatrix}
\end{equation}
$$

其中：
- $\alpha$ 指学习率（Learning rate），也称作步长，决定了迭代的次数。注意 $\alpha \geq 0$，因为需要沿着梯度反方向迭代；
- 假设 $\vec{w}$ 表示点坐标对应的向量，则上述迭代步骤可使用梯度简写为：
  $$
  \vec{w} \rightarrow \vec{w} - \alpha \nabla J
  $$

##### 选择学习率

方法：给定不同 $\alpha$ 运行梯度下降时，绘制 $J$ 和 迭代次数的图，通过观察 $J$ **是否单调递减直至收敛**来判断 $\alpha$ 的选择是否合适；
  - 单调递增或有增有减：$\alpha$ 太大，步子迈大了，应该降低 $\alpha$；
  - 单调递减但未收敛：$\alpha$ 太小，学习太慢，应该提升 $\alpha$；

经验值参考：[0.001, 0.01, 0.1, 1] 或者 [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]


##### 梯度分类

- 批量梯度下降（Batch Gradient Descent）：使用训练集中的所有数据
- 随机梯度下降（SGD）：？？根据每个训练样本进行参数更新

### 过拟合

解决过拟合的方法：
1. 收集更多的训练示例；
2. 特征值选择；
3. 正则化；


<img src='https://www.nvidia.cn/content/dam/en-zz/Solutions/gtcf20/data-analytics/nvidia-ai-data-science-workflow-diagram.svg'>

<img src='https://easyai.tech/wp-content/uploads/2022/08/523c0-2019-08-21-application.png.webp'>

<img src='https://www.tibco.com/sites/tibco/files/media_entity/2021-05/random-forest-diagram.svg'>

<img src='https://miro.medium.com/v2/resize:fit:1204/format:webp/1*iWHiPjPv0yj3RKaw0pJ7hA.png'>