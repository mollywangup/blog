---
title: "学习笔记：吴恩达机器学习"
date: 2023-08-04T08:09:47Z
draft: false
description: 
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Machine Learning
categories:
- Notes
libraries:
- mathjax
---

本笔记仅涉及监督学习和无监督学习两类。涉及到的数学知识，点到为止，侧重于实际应用（Python）。

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

Linear Regression，解决**回归**问题。包含一元线性回归和多元线性回归两类情况。

<img src='/images/posts/LinearRegression.png' alt='LinearRegression'>

#### 原理

目标：求解一组 $(\vec{w},b)$ 使得成本函数最小化。

$$ f_{w,b}(x) = wx + b \tag{Model1} $$

$$ 
f_{\vec{w}, b}(\vec{x}) = w_1 x_1 + ... + w_n x_n + b 
= \sum_{j=1}^{n} w_j x_j + b \\\\
= \vec{w} \cdot \vec{x} + b \\\\
\tag{Model2}
$$

$$ J(\vec{w},b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2 \tag{Cost function} $$

$$ \min_{\vec{w},b} J(\vec{w},b) \tag{Goal} $$

其中，模型参数如下:
- $\vec{w}$：分别对应 n 个特征的权重，也指系数（coefficients）；
  - 当 n = 1 时，也指斜率（slope）；
- $b$：bias，即偏差，也指截距（intercept）；

说明：上述 Model1、Model2 分别对应一元线性回归、多元线性回归。

#### 示例

##### 一元线性回归

以下示例来源于 sklearn 的糖尿病数据集。

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X, y = datasets.load_diabetes(return_X_y=True)

print(X.shape, y.shape)

# 仅使用其中一个特征
X = X[:, np.newaxis, 2]
print(X.shape)

# 拆分训练集和测试集
X_train = X[:-20]
X_test = X[-20:]
y_train = y[:-20]
y_test = y[-20:]

# 训练模型
reg = LinearRegression().fit(X_train, y_train)

# 使用测试集验证
y_pred = reg.predict(X_test)

# 模型结果
score = reg.score(X_train, y_train)
w = reg.coef_
b = reg.intercept_
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred) # The coefficient of determination: 1 is perfect prediction

# 绘图
plt.xlabel('X')
plt.ylabel('y')
plt.xticks(())
plt.yticks(())

plt.scatter(X_test, y_test, color='red', marker='X')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.savefig('LinearRegression.png')
```

上述模型结果是 $y = 938.24x + 152.92$

##### 多元线性回归

以下示例来源于 Python 源码。

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = np.dot(X, np.array([1, 2])) + 3
y = np.array([6, 8, 9, 11])

reg = LinearRegression().fit(X, y)
reg.score(X, y)
w = reg.coef_
b = reg.intercept_
reg.predict(np.array([[3, 5]]))
```

上述模型结果是 $y = x_1 + 2x_2 + 3$

### 多项式回归

Polynomial regression，解决**回归**问题。

#### 原理

目标：求解一组 $(\vec{w},b)$ 使得成本函数最小化。

$$ f_{\vec{w},b}(x) = w_1x + w_2x^2 + b \tag{Model1} $$
$$ f_{\vec{w},b}(x) = w_1x + w_2x^2 + w_3x^3 + b \tag{Model2} $$
$$ f_{\vec{w},b}(x) = w_1x_1 + w_2x_2 + w_3x_1x_2 + w_4x_1^2 + w_5x_2^2 + b \tag{Model3} $$

$$ J(\vec{w},b) =  \tag{Cost function}$$

$$ \min_{\vec{w},b} J(\vec{w},b) \tag{Goal} $$

其中，模型参数如下:
- $\vec{w}$：分别对应各项的权重，也指系数（coefficients）；
- $b$：bias，即偏差，也指截距（intercept）；

说明：上述 Model1、Model2、Model3 依次是一元二次多项式、一元三次多项式、二元二次多项式。

#### 示例

以下示例来源于 Python 源码。

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)

poly = PolynomialFeatures(2)
poly.fit_transform(X)

poly = PolynomialFeatures(interaction_only=True)
poly.fit_transform(X)
```

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


## 附

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