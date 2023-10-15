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

- 特征（`feature`）：指自变量；
- 标签（`label`）：指因变量，但是是真实值（`target`）；
- 训练集（`training set`）：指用于训练模型的数据集；
- 测试集（`test set`）：指用于验证模型的数据集；
- 训练示例（`training example`）：指训练集中的一组数据；
- 模型（`model`）：指拟合函数；
- 模型参数（`parameter`）：调整模型的本质是调整模型参数；
- [损失函数（Loss function）](#LossFunction)：衡量预测值与真实值之间的差异程度；
- 成本函数（`Cost function`）：用于评估模型性能，可理解为"总损失"；
- 特征工程（`feature engineering`）：对特征进行选择、提取和转换等操作，用于提高模型性能；

### 符号<a id="符号"></a>

约定如下：
1. `m` 个训练示例，`n` 个特征；
2. 向量是一维数组，使用小写字母表示，且默认为列向量；矩阵是二维数组，使用大写字母表示；
3. 非代码部分从 `1` 开始计数；
<!-- 4. 模型默认为标量函数 $f: \mathbb{R}^n \to \mathbb{R}$； -->

<br>具体符号：
- $x$ 表示特征变量，$w$ 表示回归系数，$y$ 表示真实值，$\hat{y}$ 表示预测值，都是列向量；
- $X$ 表示训练示例组成的矩阵，$(X|y)$ 表示带标签的训练示例组成的增广矩阵。注意区分：
  - $x^{(i)}$ 表示第 $i$ 个训练示例的特征，是个列向量（矩阵 $X$ 的第 $i$ 行）；
  - $x_j$ 表示第 $j$ 个特征，是个列向量（矩阵 $X$ 的第 $j$ 列）；
  - $x_j^{(i)}$ 表示第 $i$ 个训练示例的第 $j$ 个特征，是个标量；
  - $y^{(i)}$ 和 $\hat{y}^{(i)}$ 分别表示第 $i$ 个训练示例的真实值和预测值，都是标量；
<!-- 说明：$x \in \mathbb{R}^n, \space w \in \mathbb{R}^n, \space y \in \mathbb{R}, \space \hat{y} \in \mathbb{R}, \space X \in \mathbb{R}^{m \times n}$ -->

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

## 监督学习

{{< alert theme="info" >}}
有标签的是监督学习。预测连续值的是回归任务，预测离散值的是分类任务。
{{< /alert >}}

给定**包含标签**的训练集 $(X|y)$，通过算法构建一个模型，学习如何从 $x$ 预测 $\hat{y}$，则属于监督学习（Supervised Learning），即：$$ (X|y) \to f \to \hat{y} $$

监督学习分为`回归（Regression）`和`分类（Classification）`两类任务，前者预测**连续值**，后者预测**离散值**。
<!-- - `回归（Regression）`：可用于趋势预测、价格预测、流量预测等； -->
<!-- - `分类（Classification）`：可用于构建用户画像、用户行为预测、图像识别分类等； -->

### 算法思路

目标：模型应尽可能满足，最大限度地减少预测值与真实值之间的差异程度，但又不能过拟合（泛化能力）；

<!-- 思路：先选择一个训练模型，那模型参数如何确定呢？ -->
拆解目标：
Step1：选择训练模型：含模型参数；
Step2：评估模型性能：选择合适的损失函数，以衡量模型的预测值与真实值之间的差异程度；确定损失函数：将模型代入损失函数得到成本函数，以量化模型性能；
Step3：求解目标函数：求成本函数的极小值解。求极小值问题常用到[梯度下降算法](#梯度下降算法)。

### 线性回归

线性回归（Linear Regression），解决线性的**回归**问题。

#### 原理

##### 模型

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
$w$：回归系数，分别对应 n 个特征的权重（weights）或系数（coefficients），是向量；
$b$：偏差（bias）或截距（intercept），是标量；

说明：当 n = 1 时，对应一元线性回归；当 n >= 2 时，对应多元线性回归；

##### 成本函数

[MSE](#mse) 指**预测值与真实值之间误差的平方和的均值**，取值越小说明预测越准，模型性能越好。代入线性回归模型，计算公式如下：

<!-- 坑：这里是因为“下划线被解释成Markdown语法了，因此需要加\转义” 参考 https://github.com/theme-next/hexo-theme-next/issues/826 {\lVert w \rVert}\_1 正常不需要加，但为了渲染需要加-->
$$
\begin{split}
MSE &= \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \\\\
&= \frac{1}{m} \sum_{i=1}^{m} (w \cdot x^{(i)} + b - y^{(i)})^2 \\\\
\end{split}
$$

{{< expand "矩阵乘向量形式的写法（手动解的思路） ">}}

$$
MSE = \frac{1}{m} {\lVert X_{new} \cdot w_{new} - y \rVert}\_2^2
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

基于 MSE 共以下三种常见成本函数：

$$ J(w,b) = \frac{1}{2} MSE \tag{普通最小二乘回归} $$

$$ J(w,b) = \frac{1}{2} MSE + \alpha {\lVert w \rVert}\_1 \tag{Lasso 回归} $$

$$ J(w,b) = \frac{1}{2} MSE + \alpha {\lVert w \rVert}\_2^2 \tag{岭回归} $$

说明：
1. 使用 $\frac{1}{2} MSE$，仅是为了在求导数/偏导数时消去常数 2，并不影响结果；
2. $(w, b)$ 在模型 $f_{w,b}(x)$ 中是参数，在成本函数 $J(w,b)$ 中是变量；
3. `套索回归（Lasso）`：用于**特征选择**，即让回归系数稀疏（sparse）。是在普通最小二乘的基础上，添加了回归系数的 [L1 范数](#VectorNorms) 作为惩罚项；
4. `岭回归（Ridge）`：用于**防止过拟合**。是在普通最小二乘的基础上，添加了回归系数的 [L2 范数](#VectorNorms) 的平方作为惩罚项；
5. 参数 $\alpha$：非负标量，作为伸缩系数，为了控制惩罚项的大小。

##### 目标函数

求解一组模型参数 $(w,b)$ 使得成本函数 $J$ 最小化。

$$ \min_{w,b} J(w,b) $$

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

多项式回归（Polynomial Regression），解决非线性的**回归**问题。

#### 原理

{{< alert theme="info" >}}
核心思想是将非线性问题转化为线性问题。
{{< /alert >}}

以下式 $(1)(2)(3)$ 依次对应一元二次多项式、一元三次多项式、二元二次多项式模型：

$$ f_{w,b}(x) = w_1x + w_2x^2 + b \tag{1} $$

$$ f_{w,b}(x) = w_1x + w_2x^2 + w_3x^3 + b \tag{2} $$

$$ f_{w,b}(x) = w_1x_1 + w_2x_2 + w_3x_1x_2 + w_4x_1^2 + w_5x_2^2 + b \tag{3} $$

以式 $(1)$ 的模型为例，将非线性的 $f(x) \to y$ 问题，转化为线性的 $f(x,x^2) \to y$ 问题，即将非一次项的 $x^2$ 视作新特征，即可按照线性回归模型训练。

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

逻辑回归（Logistic Regression），解决**二分类**（Binary Classification）问题。

#### 原理

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

## 无监督学习<a id="UnsupervisedLearning"></a>

{{< alert theme="info" >}}
无标签的是无监督学习。
{{< /alert >}}

给定**不包含标签**的训练集 $X$，通过算法构建一个模型，揭示数据的内在分布特性及规律，则属于无监督学习（Unsupervised Learning），即：$$ X \to f \to \hat{y} $$

无监督学习主要包括以下两类任务：
- `聚类（Clustering）`
- `降维（Dimensionality reduction）`

### 算法思路

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

<!-- ## 强化学习

（Reinforcement Learning）：有延迟和稀疏的反馈标签； -->

## 模型评估

模型评估的目标是**选出泛化能力最优的模型**。

### 评估方法

留出法（Hold-out）：拆分训练集和测试集

交叉验证法（Cross Validation）：将数据集分成 N 块，使用 N-1 块进行训练，再用最后一块进行测试；

自助法（Bootstrap）：

### 损失函数<a id='LossFunction'></a>

{{< alert theme="info" >}}
损失函数可理解为评估"损失"的**方法**，成本函数可理解为最终的"**总损失**"。
{{< /alert >}}

损失函数用于**衡量预测值与真实值之间的差异程度**，一般表示为：$$ L(f(x), y) $$

成本函数用于**评估模型性能**，一般使用 $J$ 表示，且通常有：

$$
J = \displaystyle \sum_{i=1}^{m} L\left(f(x^{(i)}), y^{(i)}\right)
$$

{{< notice info >}}
成本函数更灵活，在上述 J 的基础上，有时会取均值，有时会再加上正则项（防止过拟合）。
{{< /notice >}}

#### 最小二乘误差

适用于线性回归模型。

$$
L(f_{w,b}(x), y) = \left(f_{w,b}(x) - y\right)^2
$$

#### Logistic 损失

适用于逻辑回归模型。

$$
L(f_{w,b}(x), y) = 
\begin{cases}
-log\left(f_{w,b}(x)\right) & if\space y = 1 \\\\
-log\left(1-f_{w,b}(x)\right) & if\space y = 0 \\\\
\end{cases}
$$
即
$$
-ylog(f_{w,b}(x)) - (1-y)log(f_{w,b}(x))
$$ 

### 回归指标

#### MAE

MAE（Mean Absolute Error），平均绝对误差。

$$ MAE = \frac{1}{m} \sum_{i=1}^{m} \lvert \hat{y}^{(i)} - y^{(i)} \rvert $$

#### MAPE

MAPE（Mean Absolute Percentage Error），平均绝对百分误差。

$$ MAPE = \frac{100}{m} \sum_{i=1}^{m} \lvert \frac{y^{(i)} - \hat{y}^{(i)}}{y^{(i)}} \rvert $$

#### MSE<a id="mse"></a>

MSE（Mean Squared Error），均方误差。最小二乘法的均值版，常用于线性回归模型的成本函数。

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

## 恶补数学

### 梯度

{{< alert theme="info" >}}
**梯度是一个向量，沿着梯度方向函数值上升最快，逆着梯度方向函数值下降最快。**
{{< /alert >}}

给定任意 $n$ (>=2) 元**可微**函数 $$ f(x_1, x_2,..., x_n) $$

则函数 $f$ 的**偏导数构成的向量**，称为梯度，记作 $grad f$ 或 $\nabla f$，即：

$$
grad f = \nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},..., \frac{\partial f}{\partial x_n})
$$

[梯度下降算法](#梯度下降算法) 用于求解局部极小值的问题。

{{< expand "关于偏导数">}}

函数 $f$ 对自变量 $x_j$ 的偏导数，指保持其他自变量不变，当 $x_j$ 发生增量 $\Delta x_j$ 且趋向于零即 $\displaystyle \lim_{{\Delta x_j} \to 0} $ 时，函数 $f$ 的`瞬时变化率`：

$$ \frac{\partial f}{\partial x_j} = \lim_{{\Delta x_j} \to 0} \frac{\Delta f}{\Delta x_j} = \lim_{{\Delta x_j} \to 0} \frac{f(x_j + {\Delta x_j}, ...) - f(x_j, ...)}{\Delta x_j}
$$

注意，可微一定可导，即任意给定点的邻域内所有偏导数存在且连续。

{{< /expand >}}

### 向量

基于应用层面，本文一律默认列向量，在 Python 中对应一维数组。

#### 加减法

$$
\begin{bmatrix}a \\\\ b \\\\ c \end{bmatrix}
\pm
\begin{bmatrix}d \\\\ e \\\\ f \end{bmatrix} =
\begin{bmatrix}a \pm d \\\\ b \pm e \\\\ c \pm f \end{bmatrix}
$$

#### 点积

点积（Dot product），也称作点乘、内积，运算结果是一个标量。

$$
\begin{bmatrix}a \\\\ b \\\\ c \end{bmatrix}
\cdot
\begin{bmatrix}d \\\\ e \\\\ f \end{bmatrix} =
ad + be + cf 
$$

#### 叉积

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

### 矩阵

本部分以 m 个训练示例，n 个特征组成的 $m \times n$ 矩阵 $X$（[详见这里](#符号)）为例。

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
矩阵的列秩等于线性无关的列向量的个数。满秩则线性无关，不满秩则线性相关。
{{< /alert >}}

$m \times n$ 矩阵的秩（Rank）可理解为，n 个**伪基向量**中，线性无关的**伪基向量**的个数，记作 $r$，且满足 $r <= \min{(m, n)}$. 比如：

对于矩阵 $A = \begin{bmatrix}1 & 2 \\\\ 2 & 4 \\\\ 0 & 0 \end{bmatrix}$，由于 $\begin{bmatrix}1 \\\\ 2 \\\\ 0 \end{bmatrix}$ 与 $\begin{bmatrix}2 \\\\ 4 \\\\ 0 \end{bmatrix}$ 线性相关，所以 $r(A) = 1$；

对于矩阵 $B = \begin{bmatrix}1 & 2 \\\\ 2 & 3 \\\\ 0 & 0 \end{bmatrix}$，由于 $\begin{bmatrix}1 \\\\ 2 \\\\ 0 \end{bmatrix}$ 和 $\begin{bmatrix}2 \\\\ 3 \\\\ 0 \end{bmatrix}$ 线性无关，所以 $r(B) = 2$；

{{< notice info >}}
实际应用中，对于矩阵 $X$，由于 $m \gg n$，所以其秩 $r(X) <= n$，即 $X$ **的秩由列秩决定**。 且：
当 $r(X) = n$ 时，即列满秩，说明 n 个特征线性无关；
当 $r(X) < n$ 时，即列不满秩，说明 n 个特征线性相关；
{{< /notice >}}

#### 行列式

行列式（Determinant）针对的是 $n \times n$ 矩阵即方阵，也称为 **n 阶方阵**；



#### 矩阵乘向量

{{< alert theme="info" >}}
**矩阵是一组线性变换的组合**。
{{< /alert >}}

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

#### 矩阵乘矩阵

理解：多次线性变化的叠加。

### 范数

{{< alert theme="info" >}}
**范数是一个函数，用于量化向量或矩阵的大小，即将向量或矩阵映射为一个标量。**
{{< /alert >}}

#### 向量范数<a id="VectorNorms"></a>

向量 $x = \begin{pmatrix}x_1 & x_2 & \dots & x_n \end{pmatrix}$ 的 p 范数定义如下：

$$ 
L_p(x) = {\lVert x \rVert}\_p = \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^p\right)^{1/p}
$$

则当 p 依次取 $-\infty, 1, 2, +\infty$ 时，分别对应如下范数：

$$ 
{\lVert x \rVert}\_{-\infty} = \lim_{p \to -\infty} \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^p\right)^{1/p} = 
\min_{j} {\lvert x_j \rvert} 
$$

$$ 
{\lVert x \rVert}\_1 = \sum_{j=1}^{n} {\lvert x_j \rvert} \tag{L1}
$$

$$ 
{\lVert x \rVert}\_2 = \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^2\right)^{1/2} \tag{L2}
$$

$$ 
{\lVert x \rVert}\_{+\infty} = \lim_{p \to +\infty} \left(\sum_{j=1}^{n} {\lvert x_j \rvert}^p\right)^{1/p} = 
\max_{j} {\lvert x_j \rvert}
$$

补充说明：
1. L1 范数，也称作[曼哈顿距离](#ManhattanDistance)；
2. L2 范数，也称作[欧氏距离](#EuclideanDistance)，可用于计算向量的模；
3. L$+\infty$ 范数，也称作[切比雪夫距离](#ChebyshevDistance)或最大范数；

#### 矩阵范数<a id="MatrixNorms"></a>

### 距离函数

{{< alert theme="info" >}}
**距离函数在机器学习中常用于相似性度量，距离越近，则相似性越高。**
{{< /alert >}}

对于 n 维空间中两点 $x = \begin{pmatrix}x_1 & x_2 & \dots & x_n \end{pmatrix}$ 和 $y = \begin{pmatrix}y_1 & y_2 & \dots & y_n \end{pmatrix}$，可将两点间的距离计算问题转化为量化**差向量 $x - y$ 的大小问题**。

以下式 $(1) (2) (3) (4)$ 用到了范数，依次对应 L1、L2、L$\infty$、Lp 范数；

#### 曼哈顿距离<a id="ManhattanDistance"></a>

$$ \sum_{j=1}^{n} \lvert x_j - y_j \rvert \tag{1} $$

#### 欧氏距离<a id="EuclideanDistance"></a>

$$ \sqrt{\sum_{j=1}^{n} (x_j - y_j)^2} \tag{2} $$

#### 切比雪夫距离<a id="ChebyshevDistance"></a>

$$ \max_{j} {\lvert x_j - y_j \rvert} \tag{3} $$

#### 闵可夫斯基距离<a id="MinkowskiDistance"></a>

是含参数 p 的距离函数。当 p 依次取 1, 2, $\infty$ 时，分别对应曼哈顿距离、欧氏距离、切比雪夫距离；

$$ \left(\sum_{j=1}^{n} {\lvert x_j - y_j \rvert}^p\right)^{1/p} \tag{4} $$

#### 马氏距离

？？协方差距离

#### 汉明距离

#### 杰卡德距离


## 附

- 余弦相似度（cosine similarity）：用两个向量夹角的余弦值衡量两个样本差异的大小；（越接近于1，说明夹角越接近于0，表明越相似）

一些术语概念：
- 协方差：线性相关性程度。若协方差为0则线性无关；
- 特征向量：矩阵的特征向量。数据集结构的非零向量；空间中每个点对应的一个坐标向量。

### 梯度下降算法<a id="梯度下降算法"></a>

梯度下降（Gradient Descent）是一种迭代优化算法，用于求解任意一个可微函数的**局部最小值**。在机器学习中，常用于**最小化成本函数**，即最大程度减小预测值与真实值之间的误差。即：

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

#### 选择学习率

方法：给定不同 $\alpha$ 运行梯度下降时，绘制 $J$ 和 迭代次数的图，通过观察 $J$ **是否单调递减直至收敛**来判断 $\alpha$ 的选择是否合适；
  - 单调递增或有增有减：$\alpha$ 太大，步子迈大了，应该降低 $\alpha$；
  - 单调递减但未收敛：$\alpha$ 太小，学习太慢，应该提升 $\alpha$；

经验值参考：[0.001, 0.01, 0.1, 1] 或者 [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]


#### 梯度分类

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