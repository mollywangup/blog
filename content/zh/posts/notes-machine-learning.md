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
- 标签（`label`）：指输出变量，真实值（`target` 或 `ground truth`），预测值（`prediction`）；
- 训练集（`training set`）：指用于训练模型的数据集；
- 测试集（`test set`）：指用于验证模型的数据集；
- 训练示例（`training example`）：指训练集中的一组数据；
- 模型（`model`）：指拟合函数；
- 模型参数（`parameter`）：调整模型的本质是调整模型参数；
- [损失函数（Loss function）](#LossFunction)：衡量预测值与真实值之间的差异程度，可理解为"单个损失"；
- 成本函数（`Cost function`）：用于评估模型性能，可理解为"总损失"；
- 特征工程（`feature engineering`）：对特征进行选择、提取和转换等操作，用于提高模型性能；

### 符号<a id="符号"></a>

约定如下：
1. `m` 个训练示例，`n` 个特征；
2. 向量是一维数组，使用小写字母表示，且默认为列向量；矩阵是二维数组，使用大写字母表示；
3. 非代码部分从 `1` 开始计数；

<br>具体符号：
- $x$ 表示特征变量，$w$ 表示回归系数，$y$ 表示真实值，$\hat{y}$ 表示预测值，都是列向量；
- $X$ 表示训练示例组成的矩阵，$(X|y)$ 表示带标签的训练示例组成的增广矩阵。注意区分：
  - $x^{(i)}$ 表示第 $i$ 个训练示例的特征，是个列向量（矩阵 $X$ 的第 $i$ 行）；
  - $x_j$ 表示第 $j$ 个特征，是个列向量（矩阵 $X$ 的第 $j$ 列）；
  - $x_j^{(i)}$ 表示第 $i$ 个训练示例的第 $j$ 个特征，是个标量；
  - $y^{(i)}$ 和 $\hat{y}^{(i)}$ 分别表示第 $i$ 个训练示例的真实值和预测值，都是标量；

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

## 监督学习<a id="SupervisedLearning"></a>

{{< alert theme="info" >}}
有标签的是监督学习。预测连续值的是回归任务，预测离散值的是分类任务。
{{< /alert >}}

给定**包含标签**的训练集 $(X|y)$，通过算法构建一个模型，学习如何从 $x$ 预测 $\hat{y}$，则属于监督学习，即：$$ (X|y) \to f \to \hat{y} $$

监督学习分为`回归（Regression）`和`分类（Classification）`两类任务，前者预测**连续值**，后者预测**离散值**。
<!-- - `回归（Regression）`：可用于趋势预测、价格预测、流量预测等； -->
<!-- - `分类（Classification）`：可用于构建用户画像、用户行为预测、图像识别分类等； -->

<!-- 目标：模型应尽可能满足，最大限度地减少预测值与真实值之间的差异程度，但又不能过拟合（泛化能力）； -->

<!-- 思路：先选择一个训练模型，那模型参数如何确定呢？ -->
<!-- 拆解目标：
Step1：选择训练模型：含模型参数；
Step2：评估模型性能：选择合适的损失函数，以衡量模型的预测值与真实值之间的差异程度；确定损失函数：将模型代入损失函数得到成本函数，以量化模型性能；
Step3：求解目标：求成本函数的极小值解。求极小值问题常用到[梯度下降算法](#GD)。 -->

### 线性回归

线性回归（Linear Regression），解决线性的**回归**问题。
<!-- 前提假设是预测值与真实值的误差（error）服从正态分布。 -->

#### 原理

##### 模型

$n$ 元线性回归的模型 $f: \mathbb{R}^n \to \mathbb{R}$ 如下：

$$ 
f_{w,b}(x) = w \cdot x + b = 
\begin{bmatrix}
  w_1 \\\\
  w_2 \\\\
  \vdots \\\\
  w_n 
\end{bmatrix} 
\cdot 
\begin{bmatrix}
  x_1 \\\\
  x_2 \\\\ 
  \vdots \\\\
  x_n 
\end{bmatrix} + b =
\sum_{j=1}^{n} w_j \cdot x_j + b 
$$

其中，模型参数：
$w \in \mathbb{R}^n$：回归系数，分别对应 n 个特征的权重（weights）或系数（coefficients）；
$b \in \mathbb{R}$：偏差（bias）或截距（intercept）；

说明：当 n = 1 时，对应一元线性回归；当 n >= 2 时，对应多元线性回归；

##### 成本函数

使用[最小二乘](#LeastSquaresLoss)损失：

$$
\begin{split}
L(w,b) &= \frac{1}{2} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \\\\
&= \frac{1}{2} (w \cdot x^{(i)} + b - y^{(i)})^2 
\end{split}
$$

基于最小二乘损失，常见的三种成本函数：

$$ J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{OLS} $$

<!-- $$ J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \lambda \lVert w \rVert_1 \tag{Lasso} $$ -->

$$ J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \lvert w_j \rvert \tag{Lasso} $$

<!-- $$ J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \lVert w \rVert_2^2 \tag{Ridge} $$ -->

$$ J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \lvert w_j \rvert^2 \tag{Ridge} $$

说明：
1. 使用 $\frac{1}{2m}$ 取均值，仅是为了在求（偏）导数时消去常数 $2$，不影响结果；
2. `OLS`：普通最小二乘回归；
3. `Lasso`：用于**特征选择**，即让回归系数稀疏（sparse）。是在 OLS 的基础上，添加了 $w$ 的 [L1 范数](#VectorNorms) 作为正则化项；
4. `Ridge`：用于**防止过拟合**。是在 OLS 的基础上，添加了 $w$ 的 [L2 范数](#VectorNorms) 的平方作为正则化项；
5. $\lambda$：正则化项的参数，非负标量，为了控制惩罚项的大小。

{{< expand "矩阵乘向量写法 ">}}

$$
J(w,b) = \frac{1}{2m} \lVert X_{new} \cdot w_{new} - y \rVert_2^2
$$

其中：
$$
(X_{new}|y) = \left [
\begin{array}{ccccc|c}
  1 & x_1^{(1)} & x_2^{(1)} & \dots & x_n^{(1)} & y^{(1)} \\\\ 
  1 & x_1^{(2)} & x_2^{(2)} & \dots & x_n^{(2)} & y^{(2)} \\\\ 
  \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\\\ 
  1 & x_1^{(m)} & x_2^{(m)} & \dots & x_n^{(m)} & y^{(m)} 
\end{array}
\right ]
\space
w_{new} = \begin{bmatrix}b \\\\ w_1 \\\\ w_2 \\\\ \vdots \\\\ w_n \end{bmatrix}
$$

{{< /expand >}}

##### 目标

求解一组模型参数 $(w,b)$ 使得成本函数 $J$ 最小化。

$$ \min_{w,b} J(w,b) $$

#### 代码

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

多项式回归（Polynomial Regression），解决非线性的**回归**问题。

#### 原理

{{< alert theme="info" >}}
核心思想是将非线性问题转化为线性问题，即将多项式视作新的特征。
{{< /alert >}}

以下式 $(1)(2)(3)$ 依次对应一元二次多项式、一元三次多项式、二元二次多项式模型：

$$ f_{w,b}(x) = w_1x + w_2x^2 + b \tag{1} $$

$$ f_{w,b}(x) = w_1x + w_2x^2 + w_3x^3 + b \tag{2} $$

$$ f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_1x_2 + w_4x_1^2 + w_5x_2^2 + b \tag{3} $$

以式 $(1)$ 的模型为例，将非线性的 $f(x) \to y$ 问题，转化为线性的 $f(x,x^2) \to y$ 问题，即将非一次项的 $x^2$ 视作新特征，即可按照线性回归模型训练。

#### 代码

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

逻辑回归（Logistic Regression），解决**二分类**（Binary Classification）问题。

#### 原理

{{< alert theme="info" >}}
核心思想是借助 sigmoid 函数将函数值转化为接近的二分类分布的函数，
{{< /alert >}}

##### 模型

令 $$ z = w \cdot x + b $$ 作为新的输入，通过 [Sigmoid](https://mollywangup.com/posts/notes-deep-learning/#sigmoid) 激活函数，使输出值分布以 $0.5$ 为分界： 

$$
p(y=1|x;w,b) = g(z) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

当 $p \geq 0.5$ 时，取 $1$，否则取 $0$

##### 成本函数

使用[交叉熵损失](#CrossEntropyLoss)：

$$ L(\hat{y}, y) = -y\ln\hat{y} - (1-y)\ln(1-\hat{y}) $$

对应的成本函数：

$$ J(w,b) = \frac{1}{m} \sum_{i=1}^{m} -y^{(i)} \ln \hat y^{(i)} - (1-y^{(i)}) \ln(1 - \hat y^{(i)}) $$

##### 目标

求解一组模型参数 $(w,b)$ 使得成本函数 $J$ 最小化。

$$ \min_{w,b} J(w,b) $$

<!-- true: 1, positive class
false: 0, negative class -->

### 决策树

Decison tree，解决**分类**问题。

- 根节点：无入多出
- 内部节点：一入多出
- 叶子结点：一入无出

熵

基尼系数

### 随机森林

有放回随机抽子集。

Random forest，解决**分类**问题。

回归问题：求均值
分列问题：求众数

### XGBoost


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

## 无监督学习<a id="UnsupervisedLearning"></a>

{{< alert theme="info" >}}
无标签的是无监督学习。
{{< /alert >}}

给定**不包含标签**的训练集 $X$，通过算法构建一个模型，揭示数据的内在分布特性及规律，则属于无监督学习，即：$$ X \to f \to \hat{y} $$

无监督学习主要包括以下两类任务：
- `聚类（Clustering）`
- `降维（Dimensionality reduction）`

<img src='https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png' alt='图源 scikit-learn: 聚类方法对比' width=80%>

### K-means

解决**聚类**问题。

#### 原理

给定训练集 $X \in \mathbb{R}^{m \times n}$，K-means 要实现的是将 $m$ 个点（训练示例）聚类为 $k$ 个簇（Cluster），步骤如下：

步骤一：随机初始化 $k$ 个簇中心，记作 $\mu_j \in \mathbb{R}^n$；
步骤二：为每个点 $x^{(i)}$ 分配距离最近的簇，记作 $c^{(i)}$：$$ c^{(i)} = \displaystyle\min_{j} \lVert x^{(i)} - \mu_j\rVert_2^2 $$
步骤三：为每个簇重新计算簇中心 $\mu_{j}$，方法是该簇中所有点的均值；

重复以上步骤二和步骤三，直至 $k$ 个簇中心不再发生变化（即收敛）。

成本函数可以表示为：

$$
J(c^{(1)}, \cdots, c^{(m)}, \mu_1, \cdots, \mu_k) = \frac{1}{m} \sum_{i=1}^{m} \lVert x^{(i)} - \mu_{c^{(i)}}\rVert_2^2
$$

其中：$\mu_{c^{(i)}}$ 表示 $x^{(i)}$ 所属的簇中心；

优化初始的 k 个簇中心选择：

1. 从 $X$ 中选择；
2. 

#### 代码

```python
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

# 生成测试数据
np.random.seed(0)

batch_size = 45
centers = np.array([[1, 1], [-1, -1], [1, -1]])
n_clusters = centers.shape[0]
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=[0.3, 0.7, 1])

# 使用 K-means 聚类
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

# 校验
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# 绘图
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.01, right=0.98, bottom=0.05, top=0.9)
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
# plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))
```

### DBSCAN

解决**聚类**问题。

- DBSCAN（密度聚类）：将 n 个点分为三类，然后删除噪音点；（曼哈顿距离）
  - 核心点：在半径 eps（两个样本被看做邻域的最大举例） 内的点的个数超过 min_samples（簇的样本数）；
  - 边界点：在半径 eps 内的点的个数不超过 min_samples，但落在核心点的邻域内；
  - 噪音点：既不是核心点，也不是边界点；

### PCA

主成分分析（Principal Component Analysis, PCA），解决**降维**问题。

用最少的特征尽可能解释所有的方差（越离散方差越大）。

用途：可视化，

<!-- ## 强化学习

（Reinforcement Learning）：有延迟和稀疏的反馈标签； -->

## 模型评估

### 评估方法

留出法（Hold-out）：拆分训练集和测试集

交叉验证法（Cross Validation）：将数据集分成 N 块，使用 N-1 块进行训练，再用最后一块进行测试；

自助法（Bootstrap）：

### 回归指标

#### MAE

MAE（Mean Absolute Error），平均绝对误差。

$$ MAE = \frac{1}{m} \sum_{i=1}^{m} \lvert \hat{y}^{(i)} - y^{(i)} \rvert $$

#### MAPE

MAPE（Mean Absolute Percentage Error），平均绝对百分误差。

$$ MAPE = \frac{100}{m} \sum_{i=1}^{m} \lvert \frac{y^{(i)} - \hat{y}^{(i)}}{y^{(i)}} \rvert $$

#### MSE<a id="mse"></a>

MSE（Mean Squared Error），均方误差。最小二乘的均值版，常用于线性回归模型的成本函数。

$$ MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$

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

SST (sum of squares total)，总平方和，用于衡量**真实值**相对**均值**的离散程度。SST 客观存在且与回归模型无关；

$$ SST = \sum_{i=1}^{m} (y^{(i)} - \bar{y})^2 $$

SSR (sum of squares due to regression)，回归平方和，用于衡量**预测值**相对**均值**的离散程度。当 SSR = SST 时，回归模型完美；

$$ SSR = \sum_{i=1}^{m} (\hat{y}^{(i)} - \bar{y})^2 $$

SSE (sum of squares error)，误差平方和，用于衡量**预测值**相对**真实值**的离散程度；

$$ SSE = \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$

且三者之间的关系是 $SST = SSR + SSE$.

{{< /expand >}}

<img src='https://user-images.githubusercontent.com/46241961/273468625-e2263610-af8d-4ada-9cf9-9c25eef6c3c3.svg' alt='LinearRegression_SST_SSR_SSE' width='80%'>

### 分类指标

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

## 损失函数<a id='LossFunction'></a>

{{< alert theme="info" >}}
损失函数用于**衡量预测值与真实值之间的差异程度**，也就是模型的拟合程度。
{{< /alert >}}

给定 $\hat{y},y \in \mathbb{R}$，分别表示预测值和真实值，则损失函数表示为：$$ L(\hat{y}, y) $$

成本函数 $J$ 表示为：

$$
J = \frac{1}{m} \displaystyle \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)
$$

说明：成本函数更灵活，有时会在损失函数的基础上再加上正则项；

### 最小二乘<a id="LeastSquaresLoss"></a>

$$ L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2 $$

### 交叉熵<a id="CrossEntropyLoss"></a>

推导详见[交叉熵](#CrossEntropy)

$$ L(\hat{y}, y) = H(y,\hat{y}) = - \sum_x y \ln \hat{y} $$

对于二分类问题：$$ L(\hat{y}, y) = -y\ln\hat{y} - (1-y)\ln(1-\hat{y}) $$

## 优化算法

### 梯度下降算法<a id="GD"></a>

梯度下降（Gradient Descent, GD）是一种迭代优化算法，用于求解任意一个可微函数的**局部最小值**。在机器学习中，常用于**最小化成本函数**，即最大程度减小预测值与真实值之间的误差。即：

给定成本函数 $J(w,b)$，求解一组 $(w,b)$，使得
$$ \min_{w,b} J(w,b) $$

实现的核心原理：<mark>**沿着梯度反方向，函数值下降最快**。</mark>

选定初始位置 $w$，通过重复以下步骤，直至收敛，即可得到局部最小值的解：

$$
w \leftarrow w - \alpha \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
$$

即：

$$
\begin{equation} 
  \begin{pmatrix}
    w_1 \\\\
    w_2 \\\\
    \vdots \\\\
    w_n \\\\
    b
  \end{pmatrix}
    \leftarrow
  \begin{pmatrix}
    w_1 \\\\
    w_2 \\\\
    \vdots \\\\
    w_n \\\\
    b
  \end{pmatrix}
    - \alpha
  \begin{pmatrix}
    \frac{\partial J}{\partial w_1} \\\\
    \frac{\partial J}{\partial w_2} \\\\
    \vdots \\\\
    \frac{\partial J}{\partial w_n} \\\\
    \frac{\partial J}{\partial b} 
  \end{pmatrix}
\end{equation}
$$

其中：$\alpha$ 指学习率（Learning rate），也称作步长，决定了迭代的次数。注意 $\alpha \geq 0$，因为需要沿着梯度反方向迭代；

#### 选择学习率

方法：给定不同 $\alpha$ 运行梯度下降时，绘制 $J$ 和 迭代次数的图，通过观察 $J$ **是否单调递减直至收敛**来判断 $\alpha$ 的选择是否合适；
  - 单调递增或有增有减：$\alpha$ 太大，步子迈大了，应该降低 $\alpha$；
  - 单调递减但未收敛：$\alpha$ 太小，学习太慢，应该提升 $\alpha$；

经验值参考：[0.001, 0.01, 0.1, 1] 或者 [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

### 批量梯度下降<a id="BGD"></a>

（Batch Gradient Descent, BGD）：使用训练集中的所有数据

### 随机梯度下降<a id="SGD"></a>

（stotastic gradient descent, SGD）：？？根据每个训练样本进行参数更新

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

## 数学基础

### 统计

注意这里不区分**总体**和**样本**。

#### 极差

$$ \max(y) - \min(y) $$

#### 均值

$$ \mu = \frac{1}{m} \sum_{i=1}^{m} y^{(i)} $$

#### 方差

方差（Variance）用于衡量相对均值的离散程度。

$$ \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} \left(y^{(i)} - \mu\right)^2 $$

#### 标准差

标准差（Standard deviation）是方差的平方根。

$$ \sigma = \sqrt{\sigma^2} $$

#### 变异系数

变异系数（coefficient of variation，CV），是标准差的归一化，无量纲。

$$ c_v = \frac{\sigma}{\mu} $$

### 导数

{{< alert theme="info" >}}
**一阶导用于单调性判断，二阶导用于凹凸性判断。**
{{< /alert >}}

给定一元函数 $f: \mathbb{R} \to \mathbb{R}$，则 $f$ 在点 $x$ 处的一阶导数 $f'$ 和二阶导数 $f''$ 的定义分别如下：

$$
f' = \frac{dy}{dx} = \lim_{{\Delta x} \to 0} \frac{f(x + {\Delta x})}{\Delta x}
$$

$$ f'' = (f')' = \frac{d^2y}{dx^2} $$

注意：可导等于可微，可导一定连续；
说明：一阶导表示函数在该点处的`瞬时变化率`；
用途：一阶导用于判断**单调性**；二阶导用于判断**凹凸性**，大于零则凸（U 型），小于零则凹（倒扣的 U 型）。

### 偏导数

给定多元函数 $f: \mathbb{R}^n \to \mathbb{R}$，则 $f$ 对自变量 $x_j$ 的偏导数（partial derivative），指将其他自变量视作常量时，对 $x_j$ 的导数，即：

$$ 
\frac{\partial f}{\partial x_j} = \lim_{{\Delta x_j} \to 0} \frac{f(x_j + {\Delta x_j}, ...) - f(x_j, ...)}{\Delta x_j}
$$

注意：可微一定可导，可微一定连续。

### 梯度

{{< alert theme="info" >}}
**梯度是一个向量，沿着梯度方向函数值上升最快，逆着梯度方向函数值下降最快。**
{{< /alert >}}

给定**可微**函数 $f: \mathbb{R}^n \to \mathbb{R}$，则 $f$ 的**偏导数构成的向量**，称为梯度，记作 $grad f$ 或 $\nabla f$，即：

$$
grad f = \nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},..., \frac{\partial f}{\partial x_n})
$$

用途：[梯度下降算法](#GD)

### 凸函数

如果一个函数满足**任意两点连成的线段都位于函数图形的上方**，则称这个函数为凸函数（Convex function）。

凸函数的局部最小值等于极小值，可作为选择损失函数的重要参考。

### 向量

{{< alert theme="info" >}}
**点积是标量，叉积是向量，外积是矩阵。**
{{< /alert >}}

n 维向量 $x$ 记作：

$$
x = \begin{bmatrix}x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{bmatrix} \in \mathbb{R}^n
$$

说明：本文一律默认列向量，在 Python 中对应一维数组。$x$ 也可视作一个 $n \times 1$ 矩阵。

#### 点积<a id="DotProduct"></a>

点积（Dot product），也称作点乘、内积、数量积。对于 $x,y \in \mathbb{R}^n$：

$$
x \cdot y = x^Ty = \sum_{j=1}^{n} x_jy_j \in \mathbb{R}
$$

注意：相同维数的向量才能进行点积乘法；
说明：几何意义是向量围成的平面的**面积**或空间的**体积**，大小等于 $\lVert x \rVert_2 \lVert y \rVert_2\cos(\theta)$，其中 $\theta$ 为两向量之间的夹角；
用途：[余弦相似度](#CosineSimilarity)

#### 叉积

叉积（Cross product），也称作叉乘、向量积。对于 $x,y \in \mathbb{R}^3$：

$$
\begin{split}
x \times y &= 
\left|
  \begin{matrix}
  \vec{i} & \vec{j} & \vec{k} \\\\ 
  x_1 & x_2 & x_3 \\\\
  y_1 & y_2 & y_3
  \end{matrix}
\right| \\\\
&= (x_2y_3-x_3y_2)\vec{i} - (x_1y_3-x_3y_1)\vec{j} + (x_1y_2-x_2y_1)\vec{k} \\\\
&= \begin{bmatrix}x_2y_3-x_3y_2 \\\\ -(x_1y_3-x_3y_1) \\\\ x_1y_2-x_2y_1 \end{bmatrix} \in \mathbb{R}^3
\end{split}
$$

注意：叉积的概念仅用于三维空间。这里的公式表达使用了[行列式](#Determinant)和代数余子式；
说明：几何意义是**法向量**，大小等于 $\lVert x \rVert_2 \lVert y \rVert_2\sin(\theta)$，其中 $\theta$ 为两向量之间的夹角。

#### 外积

外积（Outer product）。对于 $x \in \mathbb{R}^m, y \in \mathbb{R}^n$：

$$
x \otimes y = xy^T = 
\begin{bmatrix}
  x_1y_1 & x_1y_2 & \dots & x_1y_n \\\\ 
  x_2y_1 & x_2y_2 & \dots & x_2y_n \\\\ 
  \vdots & \vdots & \ddots & \vdots \\\\ 
  x_my_1 & x_my_2 & \dots & x_my_n
\end{bmatrix}
\in \mathbb{R}^{m \times n}
$$

说明：运算结果是个矩阵。

### 矩阵

$m \times n$ 矩阵可理解为 n 个列向量的集合（或 m 个行向量的集合）。如果把每个列向量看作一个**伪基向量**，则矩阵又可理解为 **伪 n 维空间**的一组**伪基向量**的集合。

#### 线性相关

{{< alert theme="info" >}}
**n 个线性无关的向量，可作为基向量，张成一个 n 维空间。**
{{< /alert >}}

对于 n 个向量 $x_1,x_2,...,x_n$，令其线性组合为零向量，即等式

$$ w_1x_1 + w_2x_2 + \cdots + w_nx_n = \vec{0},\space\space (n>=2) $$

其中 $w_j$ 为标量。**如果当且仅当 $w_1 = w_2 = \cdots = w_n = 0$ 即全部系数为零时才成立，则称该 n 个向量线性无关**，否则线性相关。

{{< notice info >}}
线性无关，对于 n 取 2 就是两个向量不共线，对于 n 取 3 就是三个向量不共面。
{{< /notice >}}

#### 秩

{{< alert theme="info" >}}
矩阵的秩等于线性无关的列（行）向量的个数。满秩则线性无关，不满秩则线性相关。
{{< /alert >}}

矩阵的秩（Rank）记作 $rank$，且**秩 = 列秩 = 行秩**。

机器学习中，对于矩阵 $X$（[详见这里](#符号)），由于 $m \gg n$，所以其秩由 $n$ 决定。 且：
当 $rank(X) = n$ 时，即列满秩，说明 n 个特征线性无关；
当 $rank(X) < n$ 时，即列不满秩，说明 n 个特征线性相关；

#### 行列式<a id="Determinant"></a>

行列式（Determinant）针对的是 $n \times n$ 矩阵，也称为 **n 阶方阵**，记作 $\det$.


#### 矩阵乘向量

{{< alert theme="info" >}}
**矩阵是一组线性变换的组合**。
{{< /alert >}}

理解：将矩阵的列向量看作一组新的**伪基向量**，则矩阵乘向量可以理解为**对向量进行一次线性变换**。

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

#### 矩阵乘矩阵

理解：多次线性变化的叠加。

### 范数

{{< alert theme="info" >}}
**范数是一个函数，用于量化向量或矩阵的大小，即将向量或矩阵映射为一个标量。**
{{< /alert >}}

#### 向量范数<a id="VectorNorms"></a>

n 维向量 $x$ 的 p 范数定义如下：

$$ 
L_p(x) = \lVert x \rVert_p = \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^p\right)^{1/p}
$$

则当 p 依次取 $-\infty, 1, 2, +\infty$ 时，分别对应如下范数：

$$ 
\lVert x \rVert_{-\infty} = \lim_{p \to -\infty} \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^p\right)^{1/p} = 
\min_{j} {\lvert x_j \rvert} \tag{$L_{-\infty}$}
$$

$$ 
\lVert x \rVert_1 = \sum_{j=1}^{n} {\lvert x_j \rvert} \tag{$L_1$}
$$

$$ 
\lVert x \rVert_2 = \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^2\right)^{1/2} \tag{$L_2$}
$$

$$ 
\lVert x \rVert_{+\infty} = \lim_{p \to +\infty} \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^p\right)^{1/p} = 
\max_{j} {\lvert x_j \rvert} \tag{$L_{+\infty}$}
$$

补充说明：
1. L1 范数，也称作[曼哈顿距离](#ManhattanDistance)；
2. L2 范数，也称作[欧氏距离](#EuclideanDistance)，可用于计算向量的模；
3. L$+\infty$ 范数，也称作[切比雪夫距离](#ChebyshevDistance)或最大范数；

#### 矩阵范数<a id="MatrixNorms"></a>

### 距离和相似度

两个 n 维向量 $x$ 和 $y$，可通过两个向量之间的**距离**或**相似度**来衡量差异程度。距离越近，则相似性越高，也就是差异程度越小。

以下式 $(1) (2) (3) (4)$ 对应差向量 $x - y$ 的不同范数。

#### 闵可夫斯基距离<a id="MinkowskiDistance"></a>

是含参数 p 的距离函数。当 p 依次取 1, 2, $\infty$ 时，分别对应曼哈顿距离、欧氏距离、切比雪夫距离；

$$ \left(\sum_{j=1}^{n} {\lvert x_j - y_j \rvert}^p\right)^{1/p} \tag{1} $$

#### 曼哈顿距离<a id="ManhattanDistance"></a>

$$ \sum_{j=1}^{n} \lvert x_j - y_j \rvert \tag{2} $$

#### 欧氏距离<a id="EuclideanDistance"></a>

$$ \sqrt{\sum_{j=1}^{n} (x_j - y_j)^2} \tag{3} $$

#### 切比雪夫距离<a id="ChebyshevDistance"></a>

$$ \max_{j} {\lvert x_j - y_j \rvert} \tag{4} $$

#### 海明距离


#### 马氏距离

？？协方差距离


#### 杰卡德距离

#### KL 散度<a id="KLDivergence"></a>

给定随机变量 $x$ 的两个概率分布 $p(x)$ 和 $q(x)$，KL 散度用于衡量两个概率分布之间的差异程度，公式如下：

$$ D_{KL}(p||q) = \sum_x p(x) \ln \frac{p(x)}{q(x)} $$

说明：也称作[相对熵](#KLD)。大于等于零，越小越相似。

#### 余弦相似度<a id="CosineSimilarity"></a>

余弦相似度（Cosine Similarity）使用两个向量夹角的余弦值来衡量相似度，公式如下：

$$ \frac{x \cdot y}{\lVert x \rVert_2 \lVert y \rVert_2} $$

说明：由[向量点积](#DotProduct)计算公式推导而来。越接近于 1，说明夹角越接近于 0，表明越相似。

#### 皮尔逊相关系数


### 贝叶斯定理

#### 基础概念

设 $A$ 和 $B$ 为两个随机事件，则：

**先验概率**：指某个事件发生的概率。$A$ 发生的概率记作 $P(A)$；

**联合概率**：指两个事件同时发生的概率。$A$ 与 $B$ 的联合概率，记作 $P(A,B)$ 或 $P(AB)$ 或 $P(A \cap B)$；

**条件概率**：也称作**后验概率**。$B$ 发生的条件下 $A$ 发生的概率，其中 $P(B) \neq 0$，记作 $P(A|B)$。有：$$ P(A|B) = \frac{P(A,B)}{P(B)} $$

**相互独立**：$A$ 与 $B$ 相互独立，当且仅当 $P(A,B) = P(A)P(B)$ 时成立；

{{< notice info>}}
朴素贝叶斯的**朴素**指的是假设特征之间相互独立。
{{< /notice >}}

#### 贝叶斯定理

贝叶斯定理（Bayes'theorem）公式如下（其中 $P(B) \neq 0$）：

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

可由条件概率推导得到，因为 $P(A,B) = P(A|B)P(B) = P(B|A)P(A)$

### 概率分布

说明：离散型随机变量对应**概率质量函数**（Probability Mass Function, PMF），连续型随机变量对应**概率密度函数**（Probability Density Function, PDF）。

#### 伯努利分布

伯努利分布（Bernoulli distribution），也称作 0-1 分布。离散型随机变量 $X$ 服从参数 $\phi \in (0,1)$ 的伯努利分布，记作：

$$
X \sim Bern(\phi)
$$

其概率质量函数、期望值和方差分别如下：

$$
p(X=x;\phi) = 
\begin{cases}
\phi, & \text{if $x=1$} \\\\
1-\phi, & \text{if $x=0$} 
\end{cases} = 
\phi^x(1-\phi)^{1-x} \tag{PMF}
$$

$$ \mu = \sum_{i} x_i p(x_i) = \phi $$

$$ \sigma^2 = \sum_{i} \left(x_i - \mu\right)^2 p(x_i) = \phi(1-\phi) $$

#### 二项分布

二项分布（Binomial distribution）指进行一组 `n 重伯努利试验`，成功（$x=1$）次数的概率分布。离散型随机变量 $X$ 服从参数 $n, \phi$ 的二项分布，记作：

$$
X \sim B(n, \phi)
$$

其概率质量函数、期望值和方差分别如下，其中 $x \in \lbrace 0, 1, ..., n \rbrace$：

$$
p(X=x;n,\phi) = \frac{n!}{x!(n-x)!} \phi^x (1-\phi)^{n-x}  \tag{PMF}
$$

$$ \mu = n\phi $$

$$ \sigma^2 = n\phi(1-\phi)$$

<img src='https://user-images.githubusercontent.com/46241961/278027246-01e7fc5c-66b1-4b79-b855-002f64756da9.svg' alt='二项分布（10, 0.5）' width=60%>

{{< notice info>}}
1. $X \sim B(1, \phi)$ 等同于 $X \sim Bern(\phi)$；
2. 二项分布的期望值等于 $n$ 倍的伯努利分布的期望值，同理方差；
3. 当 $n \to \infty$ 时，二项分布趋向于正态分布。
4. {{< /notice >}}

#### 高斯分布

高斯分布（Gaussian distribution），也称作正态分布（Normal distribution）。连续型随机变量 $X$ 服从均值 $\mu$，方差 $\sigma^2$ 的正态分布，记作：

$$
X \sim N(\mu, \sigma^2)
$$

其概率密度函数如下：

$$
p(X=x;\mu,\sigma) = \frac{1}{\sigma \sqrt{2 \pi}} \exp\left(-\frac{(x-\mu)^2} {2 \sigma^2}\right)  \tag{PDF}
$$

<img src='https://user-images.githubusercontent.com/46241961/278027298-8e138d61-dc0b-4b40-aeef-9ee3bdb4a493.svg' alt='高斯分布' width=70%>

#### 拉普拉斯分布


<!-- #### 指数分布 -->


<!-- #### 泊松分布 -->


## 附

一些术语概念：
- 协方差：线性相关性程度。若协方差为0则线性无关；
- 特征向量：矩阵的特征向量。数据集结构的非零向量；空间中每个点对应的一个坐标向量。

### 熵<a id="Entropy"></a>

{{< alert theme="info" >}}
**信息量**是信息的大小，**熵**是信息量的期望值，**相对熵**用于衡量两个概率分布之间的差异，**交叉熵**是相对熵的简化版。
{{< /alert >}}

#### 信息量

给定随机变量 $x$ 的概率分布 $p(x)$，则 $x$ 的**信息量**定义如下：$$ I(x) = \ln \frac{1}{p(x)} = - \ln p(x) $$

说明：
概率越小，信息量越大；
对数底数仅影响量化的单位，以 2 为底则单位是比特，以 e 为底则单位是纳特（默认）。

#### 熵

熵（Entropy）等于随机变量 $x$ 的**信息量的期望值**，用于衡量**不确定性**，定义如下：

$$ 
H(p) = E(I(x)) = \sum_x p(x) I(x) = - \sum_x p(x) \ln p(x)
$$

说明：
熵越大，则不确定性越大；
离散型随机变量对应**求和**，连续型随机变量对应**求积分**（已省略）；

#### 相对熵<a id="KLD"></a>

相对熵（Relative Entropy），又称为 KL 散度（Kullback-Leibler divergence），用于**衡量两个概率分布之间的差异程度**。对于两个概率分布 $p(x)$ 和 $q(x)$，其相对熵定义如下：

$$ D_{KL}(p||q) = \sum_x p(x) \ln \frac{p(x)}{q(x)} $$

说明：
相对熵越小，则 $p(x)$ 和 $q(x)$ 分布越接近；
$D_{KL}(p||q) \geq 0$，当且仅当 $p(x) = q(x)$ 时等号成立，且非对称；

{{< expand "证明：相对熵大于等于零" >}}
由于 $\ln(x) \leq x - 1$，则：

$$
\begin{split}
\- D_{KL}(p||q) &= \sum_x p(x) \ln \frac{q(x)}{p(x)} \\\\ 
&\leq \sum_x p(x) (\frac{q(x)}{p(x)} - 1) &= \sum_x (q(x) - p(x)) = 0
\end{split}
$$

因此 $D_{KL}(p||q) \geq 0$，当且仅当 $p(x) = q(x)$ 时为零。
{{< /expand >}}

#### 交叉熵<a id="CrossEntropy"></a>

将上述相对熵公式展开：

$$ 
\begin{split}
D_{KL}(p||q) &= \sum_x p(x) \ln \frac{p(x)}{q(x)} \\\\
&= \sum_x p(x) \ln p(x) - \sum_x p(x) \ln q(x) \\\\
&= -H(p) + H(p,q)
\end{split}
$$

其中，前半部分就是负的 $p(x)$ 的熵，后半部分则就是交叉熵（Cross Entropy）：$$ H(p,q) = - \sum_x p(x) \ln q(x) $$

实际应用中，如果将 $p(x)$ 作为真实值的概率分布，$q(x)$ 作为预测值的概率分布，则由于真实值的熵 $H(p)$ 是一个常数，因此：

$$ D_{KL}(p||q) \simeq H(p,q)$$

### 过拟合

解决过拟合的方法：
1. 收集更多的训练示例；
2. 特征选择；
3. 正则化；


<!-- <img src='https://www.nvidia.cn/content/dam/en-zz/Solutions/gtcf20/data-analytics/nvidia-ai-data-science-workflow-diagram.svg'>

<img src='https://easyai.tech/wp-content/uploads/2022/08/523c0-2019-08-21-application.png.webp'>

<img src='https://www.tibco.com/sites/tibco/files/media_entity/2021-05/random-forest-diagram.svg'>

<img src='https://miro.medium.com/v2/resize:fit:1204/format:webp/1*iWHiPjPv0yj3RKaw0pJ7hA.png'> -->