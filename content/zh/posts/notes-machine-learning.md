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
---

✍ 持续更新 ing

统一口径：

- `features`: 指输入值，常称作特征值；
- `labels`: 指输出值，可以是实际值，也可以是预测值；
  - `targets`: 指实际输出值；
  - `predictions`: 指预测输出值；
- `Training set`: 训练集，指用于训练模型的数据集；
- `Single training example`: 训练示例，指训练集中的一组数据；

## 机器学习分类

机器学习解决的是：给定训练集 -> 生成训练模型 -> 根据训练模型进行预测。

根据训练集中是否包含标签，可分为以下四类（本文仅涉及前两类）：

- 监督学习（Supervised Learning）：包含标签；
- 无监督学习（Unsupervised Learning）：不包含标签；
- 半监督学习（Semi-Supervised Learning）：部分包含标签；
- 强化学习（Reinforcement Learning）

## 监督学习

训练集中**包含标签**，则属于监督学习，即 `(features, labels) -> model`.

监督学习分类及常见模型如下：

- 回归（Regression）：
  - 线性回归（Linear Regression）
  - 
- 分类（Classification）：

### 回归

回归问题的输出值都是**连续型变量**。

#### 线性回归模型

{{< boxmd >}}
y = wx + b
{{< /boxmd >}}

其中：
- w: weight，即权重；
- b: bias，即偏差；


### 分类

分类问题的输出值都是**离散型变量**。

- KNN (K-Nearest Neighbors)：K近邻算法；
- 决策树：
- Nbayes（朴素贝叶斯）：

## 无监督学习

训练集中**不包含标签**，则属于无监督学习，即 `(features) -> model`.

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