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

✍ 持续更新 ing

统一口径：

- `features`: 指输入值，常称作特征值 $ x^{(i)} $；
- `labels`: 指输出值，可以是实际值，也可以是预测值；
  - `targets`: 指实际输出值 $ y^{(i)} $；
  - `predictions`: 指预测输出值 $ \hat y^{(i)} $；
- `training set`: 训练集，指用于训练模型的数据集；
- `training example`: 训练示例，指训练集中的一组数据；
- `Model`：训练模型，拟合函数
- `Parameters`：模型参数，调整模型的本质是调整模型参数；

## 机器学习分类

机器学习解决的是：给定训练集 -> 生成最佳拟合模型 -> 根据拟合模型进行预测。

根据训练集中是否包含标签，可分为以下四类（本文仅涉及前两类）：

- 监督学习（Supervised Learning）：包含标签；
- 无监督学习（Unsupervised Learning）：不包含标签；
- 半监督学习（Semi-Supervised Learning）：部分包含标签；
- 强化学习（Reinforcement Learning）

## 成本函数

损失函数（Loss function）用于衡量预测值与实际值之间的差异程度，一般使用 $L$ 表示：

$$ L(f_{w_1,w_2,...,w_n,b}(x^{(i)}), y^{(i)}) $$

成本函数（Cost function）也称作代价函数，用于评估模型的**拟合程度**。一般使用 $J$ 表示：

$$
J(w_1,w_2,...,w_n,b) = \displaystyle\sum_{i=1}^{m} L(f_{w_1,w_2,...,w_n,b}(x^{(i)}), y^{(i)})
$$

### Squared error cost function

适用于线性回归模型。

$$
J(w,b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (\hat y^{(i)} - y^{(i)})^2 
$$
即
$$ 
J(w,b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 
$$

其中 `m` 为训练集中训练示例数量，几何意义上指点的个数。
注意：除以 `2m` 而不是 ~~`m`~~，目的是使得求偏导数时，在不影响结果的前提下，为了更加简洁（仅此而已）；

## 梯度下降

### 梯度

给定任意 $n$ 元**可微**函数 $$f(x_1, x_2,..., x_n)$$

则 $f$ 的**偏导数构成的向量**，称为梯度，记作 $grad f$ 或 $\nabla f$，即：

$$
grad f = \nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},..., \frac{\partial f}{\partial x_n})
$$

梯度的几何意义是，<mark>**梯度方向**是函数值上升最快的方向，反方向为下降最快的方向</mark>，因此可应用于求解多元函数的极值。

{{< expand "偏导数">}}

函数 $f$ 对自变量 $x_i$ 的偏导数，指保持其他自变量不变，当 $x_i$ 发生增量 $\Delta x_i$ 且趋向于零即 $\displaystyle \lim_{{\Delta x_i} \to 0} $ 时，函数 $f$ 的`瞬时变化率`：

$$ \frac{\partial f}{\partial x_i} = \lim_{{\Delta x_i} \to 0} \frac{\Delta f}{\Delta x_i} = \lim_{{\Delta x_i} \to 0} \frac{f(x_i + {\Delta x_i}, ...) - f(x_i, ...)}{\Delta x_i}
$$

注意，可微一定可导，即任意给定点的邻域内所有偏导数存在且连续。

{{< /expand >}}

### 梯度下降算法

梯度下降（Gradient Descent）是一种迭代优化算法，用于寻找任意一个可导函数的**局部最小值**。

在机器学习中，常用于**最小化成本函数**，即最大程度减小预测值与实际值之间的误差。即：

<!-- <img src='https://upload.wikimedia.org/wikipedia/commons/7/79/Gradient_descent.png'> -->

{{< boxmd >}}
给定成本函数 $J(w_1,w_2,...,w_n)$，求解一组 $(w_1,w_2,...,w_n)$，使得
$$ \min_{w_1,w_2,...,w_n} J(w_1,w_2,...,w_n) $$
{{< /boxmd >}}

实现的核心原理：<mark>**沿着梯度反方向，函数值下降最快**。</mark>

选定初始位置点 $(w_1,w_2,...,w_n)$，

假定每个变量每次调整相同的幅度 $\alpha$（其中 $\alpha \geq 0$），则此时 $w_i$ 将迭代为以下值：
$$
w_i \rightarrow w_i - \alpha \frac{\partial}{\partial w_i} J(w_1,w_2,...,w_n,b)
$$

重复以上步骤，直至收敛，得到最终的一组值，即局部最优解。

其中 $\alpha$ 指学习率，也称作步长，决定了迭代的次数。

适用于线性回归、神经网络（深度学习）等模型。

分类：
- 批量梯度下降（Batch Gradient Descent）：使用训练集中的所有数据
- 随机梯度下降（SGD）：？？根据每个训练样本进行参数更新

## 监督学习

训练集中**包含标签**，则属于监督学习，即 `(features, targets) -> Model`.

监督学习分类及常见模型如下：

- 回归（Regression）：
  - 线性回归（Linear Regression）
  - 
- 分类（Classification）：

### 回归

回归问题的输出值都是**连续型变量**。

#### 线性回归模型

##### 一元线性回归

给定包含参数的训练模型，找到一组参数，使得成本函数最小化。

Model: 

$$ f_{w,b}(x) = wx + b $$

Parameters:

- w: weight，即权重，也是斜率（slope）；
- b: bias，即偏差；

Cost Function:

$$ J(w,b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$

Goal:

$$ \min_{w,b} J(w,b) $$

##### 多元线性回归


### 分类

分类问题的输出值都是**离散型变量**。

- KNN (K-Nearest Neighbors)：K近邻算法；
- 决策树：
- Nbayes（朴素贝叶斯）：

## 无监督学习

训练集中**不包含标签**，则属于无监督学习，即 `(features) -> Model`.

无监督学习分类及常见模型如下：

- 聚类（Clustering）
  - K-means
  - DBSCAN
- 降维（Dimensionality reduction）
  - PCA

### 聚类

- K-means：将 n 个点分为 k 个簇，使得簇内具有较高的相似度，簇间具有较低的相似度；（欧氏距离）
- DBSCAN（密度聚类）：将 n 个点分为三类，然后删除噪音点；（曼哈顿距离）
  - 核心点：在半径 eps（两个样本被看做邻域的最大举例） 内的点的个数超过 min_samples（簇的样本数）；
  - 边界点：在半径 eps 内的点的个数不超过 min_samples，但落在核心点的邻域内；
  - 噪音点：既不是核心点，也不是边界点；

### 降维

- PCA：主成分分析；


效果评估
- 针对监督学习：
  - 偏差（bias）：偏离程度；
  - 方差（variance）：分散程度；
- 针对分类：
  - 错误率：分类错误的样本数占样本总数的比例。
  - 精确率：分类正确的样本数占样本总数的比例。
  - 查准率（也称准确率），即在检索后返回的结果中，真正正确的个数占你认为是正确的结果的比例。
  - 查全率（也称召回率），即在检索结果中真正正确的个数，占整个数据集（检索到的和未检索到的）中真正正确个数的比例。
  - F1是一个综合考虑查准率与查全率的度量，其基于查准率与查全率的调和平均定义：即：F1度量的一般形式-Fβ，能让我们表达出对查准率、查全率的不同偏好
  


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





<img src='https://www.nvidia.cn/content/dam/en-zz/Solutions/gtcf20/data-analytics/nvidia-ai-data-science-workflow-diagram.svg'>

<img src='https://easyai.tech/wp-content/uploads/2022/08/523c0-2019-08-21-application.png.webp'>

<img src='https://www.tibco.com/sites/tibco/files/media_entity/2021-05/random-forest-diagram.svg'>

<img src='https://miro.medium.com/v2/resize:fit:1204/format:webp/1*iWHiPjPv0yj3RKaw0pJ7hA.png'>