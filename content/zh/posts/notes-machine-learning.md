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

本笔记基于以下学习资料（侧重于实际应用，及直接使用数学结论）：
> 入门机器学习：[(强推|双字)2022吴恩达机器学习Deeplearning.ai课程](https://www.bilibili.com/video/BV1Pa411X76s/)
> Python 代码库：[scikit-learn 官网](https://scikit-learn.org/stable/index.html)
> 复习线性代数：3Blue1Brown 的 [线性代数的本质 - 系列合集](https://www.bilibili.com/video/BV1ys411472E/)

统一口径：

- `feature`: 指输入变量，常称作特征；
  - 一元对应 $x$，多元对应 $\vec x$
- `label`: 指输出值，可以是实际值，也可以是预测值，常称作标签；
  - `target`: 指实际输出值 $y$；
  - `prediction`: 指预测输出值 $\hat y$；
- `training set`: 训练集，指用于训练模型的数据集；
- `training example`: 训练示例，指训练集中的一组数据；
  - 一元对应 $x^{(i)}$，多元对应 $\vec x^{(i)}$
- `Model`：训练模型，即最终的拟合函数；
- `Parameters`：模型参数，调整模型的本质是调整模型参数；
- `feature engineering`：特征工程，指从原始数据中选择、提取和转换最相关的若干个特征，以提高机器学习模型的准确性；

## 机器学习概述

机器学习解决的问题是：给定训练集，通过机器学习算法生成最佳训练模型，最终应用于预测新特征对应的输出值。

### 分类

根据训练集中包含标签的情况，可分为以下三类（本文仅涉及前两类）：

- 监督学习（Supervised Learning）：训练集中包含标签，分为：
  - **回归（Regression）**
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


## 监督学习

如果训练集中**包含标签**，则属于监督学习，即 `(features, targets) -> Model` 问题。

共以下两类学习任务：
- 回归：预测值为**连续型**，可应用于趋势预测、价格预测、流量预测等；
- 分类：预测值为**离散型**，可应用于构建用户画像、用户行为预测、图像识别分类等；

### 线性回归

线性回归（Linear Regression），解决线性的**回归**问题。包含一元线性回归和多元线性回归两类情况。

<img src='https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png' alt='One variable Linear Regression Example' width='60%'>

#### 原理

目标：求解一组模型参数 $(\vec{w},b)$ 使得成本函数 $J$ 最小化。

$$ 
f_{\vec{w},b}(\vec{x}) = \sum_{j=1}^{n} w_j x_j + b 
= \begin{bmatrix}w_1 \\\\ w_2 \\\\ \vdots \\\\ w_n \end{bmatrix} \cdot \begin{bmatrix}x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{bmatrix} + b 
= \vec{w} \cdot \vec{x} + b 
\tag{Model}
$$

$$ J(\vec{w},b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2 \tag{Cost function} $$

$$ J(\vec{w},b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2 + \alpha {\lVert \vec{w} \rVert}_1 \tag{Cost function: L1 norm} $$

$$ J(\vec{w},b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2 + \alpha {\lVert \vec{w} \rVert}_2^2 \tag{Cost function: L2 norm} $$

$$ \min_{\vec{w},b} J(\vec{w},b) \tag{Goal} $$

其中，模型参数如下:
- $\vec{w} = \begin{bmatrix}w_1 \\\\ w_2 \\\\ \vdots \\\\ w_n \end{bmatrix}$，分别对应 n 个特征的权重（weights）或系数（coefficients）；
- $b$：偏差（bias）或截距（intercept）；

说明：
- 当 n = 1 时，对应一元线性回归，即 $ f_{w,b}(x) = wx + b $；当 n >= 2 时，对应多元线性回归；
- 对于普通最小二乘法：
  - $MSE = \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$，但机器学习中经验使用 $\frac{1}{2} MSE$，仅用于求导数/偏导数时，计算消去常数2，并不影响结果；
- 三种成本函数分别对应的线性回归模型：
  - 普通最小二乘回归；
  - Lasso 回归（也称作 L1 回归或套索回归）：
    - 作用：可进行特征选择，即让特征系数取零；
    - 方法：在最小二乘法的基础上，添加了 L1 正则项 $\alpha {\lVert \vec{w} \rVert}_1$ 作为惩罚（其中 $\alpha > 0$）；
  - Ridge 回归（也称作 L2 回归或岭回归）：
    - 作用：可防止过拟合；
    - 方法：在最小二乘法的基础上，添加了 L2 正则项即 $\alpha {\lVert \vec{w} \rVert}_2^2$ 作为惩罚（其中 $\alpha > 0$）；

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

以下示例来源于 sklearn 的糖尿病数据集，选取了所有的特征，并对比了普通最小二乘/Lasso/Ridge 三种回归的模型性能。

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

上述模型结果是 $y = x_1 + 2x_2 + 3$

### 多项式回归

多项式回归（Polynomial regression），解决非线性的**回归**问题。

核心思想是将非线性问题转化为线性问题。

#### 原理

目标：求解一组模型参数 $(\vec{w},b)$ 使得成本函数 $J$ 最小化。


$$ f_{\vec{w},b}(x) = w_1x + w_2x^2 + b \tag{Model1} $$
$$ f_{\vec{w},b}(x) = w_1x + w_2x^2 + w_3x^3 + b \tag{Model2} $$
$$ f_{\vec{w},b}(x) = w_1x_1 + w_2x_2 + w_3x_1x_2 + w_4x_1^2 + w_5x_2^2 + b \tag{Model3} $$

$$ J(\vec{w},b) =  \tag{Cost function}$$

$$ \min_{\vec{w},b} J(\vec{w},b) \tag{Goal} $$

其中，模型参数如下:
- $\vec{w}$：分别对应各项的权重（weights）或系数（coefficients）；
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

## 无监督学习

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

### 混淆矩阵

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

### ROC 曲线

[深入介紹及比較ROC曲線及PR曲線](https://medium.com/nlp-tsupei/roc-pr-%E6%9B%B2%E7%B7%9A-f3faa2231b8c)

用于分类模型的效果评估，以可视化的方式。


训练集和测试集
交叉验证时：将数据集分成 N 块，使用 N-1 块进行训练，再用最后一块进行测试；

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

### SST/SSR/SSE/R<sup>2</sup>

助记小技巧：**T** is short for total, **R** is short for regression, **E** is short for error.

<img src='https://user-images.githubusercontent.com/46241961/273396195-6b600d0a-5248-4c07-aa9c-66bbc9e36518.svg' alt='LinearRegression_SST_SSR_SSE' width='80%'>

<br>以下公式统一说明：
$y$：实际值，target
$\hat{y}$：预测值，prediction
$\bar{y}$：平均值，mean

#### SST

SST (sum of squares total)：总平方和，用以衡量**实际值**偏离**均值**的程度；

$$ SST = \sum (y - \bar{y})^2 $$

思考：SST 客观存在，与回归模型无关；

#### SSR

SSR (sum of squares due to regression)：回归平方和，用于衡量**预测值**偏离**均值**的程度；

$$ SSR = \sum (\hat{y} - \bar{y})^2 $$

思考：当 SSR = SST 时，即回归模型进行了完美的预测；

#### SSE

SSE (sum of squares error)：误差平方和，用于衡量**预测值**偏离**实际值**的程度；

$$ SSE = \sum (y - \hat{y})^2 $$

思考：
- SSE 直接决定了回归模型的质量；
- 三者之间的关系是 $SST = SSR + SSE$；

#### R<sup>2</sup>

R<sup>2</sup> (coefficient of determination)：决定系数，通过**回归平方和**占比**总平方和**来衡量回归模型的质量；

$$ R^2 = \frac{SSR}{SST} = 1- \frac{SSE}{SST} $$

思考：
- 当 $R^2 \to 1$ 时，表明模型质量越高，因为此时 $SSR \to SST$，即客观存在的 $SST$，可以近似全部使用 $SSR$ 解释，此时 $SSE \to 0$；
- 当 $R^2 \to 0$ 时，表明模型质量越差，因为此时 $SSE \to SST$，即客观存在的 $SST$，几乎全部来自于 $SSE$；

### 成本函数

损失函数（Loss function）用于衡量预测值与实际值之间的差异程度，一般使用 $L$ 表示：

$$ L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) $$

成本函数（Cost function）也称作代价函数，用于评估模型的**拟合程度**。一般使用 $J$ 表示：

$$
J(\vec{w},b) = \displaystyle \frac{1}{m} \sum_{i=1}^{m} L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})
$$

#### MSE Cost Function

均方误差成本函数（Mean Squared Error Cost Function），适用于线性回归模型。

$$
J(w,b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (\hat y^{(i)} - y^{(i)})^2 
$$
即
$$ 
J(w,b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 
$$

其中 `m` 为训练集中训练示例数量，几何意义上指点的个数。
注意：除以 `2m` 而不是 ~~`m`~~，目的是在不影响结果的前提下，使得求解偏导数更加简洁（仅此而已）；

#### Logistic loss function

适用于逻辑回归模型。

$$
L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) = 
\begin{cases}
-log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) & if\ y^{(i)} = 1 \\\\
-log\left(1-f_{\vec{w},b}(\vec{x}^{(i)})\right) & if\ y^{(i)} = 0 \\\\
\end{cases}
$$
即
$$
-y^{(i)}log(f_{\vec{w},b}(\vec{x}^{(i)}) - (1-y^{(i)})log(f_{\vec{w},b}(\vec{x}^{(i)})
$$

### 梯度下降

#### 梯度定义

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

### 过拟合

解决过拟合的方法：
1. 收集更多的训练示例；
2. 特征值选择；
3. 正则化；


<img src='https://www.nvidia.cn/content/dam/en-zz/Solutions/gtcf20/data-analytics/nvidia-ai-data-science-workflow-diagram.svg'>

<img src='https://easyai.tech/wp-content/uploads/2022/08/523c0-2019-08-21-application.png.webp'>

<img src='https://www.tibco.com/sites/tibco/files/media_entity/2021-05/random-forest-diagram.svg'>

<img src='https://miro.medium.com/v2/resize:fit:1204/format:webp/1*iWHiPjPv0yj3RKaw0pJ7hA.png'>