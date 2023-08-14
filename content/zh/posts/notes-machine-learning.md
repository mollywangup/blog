---
title: "吴恩达机器学习笔记"
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
- 
---

✍ 持续更新。

统一口径：
自变量：(features, labels)


## 机器学习分类

可分为以下4类，但本文以常见的前2类为主：

- **Supervised Learning**：监督学习。从正确答案中（有标记数据集）学习；
- Unsupervised Learning：无监督学习。发现规律（无标记数据集）；
- Semi-Supervised Learning：半监督学习；
- Reinforcement Learning：强化学习。奖惩机制互动式学习；

### Supervised Learning

监督学习。解决


### Unsupervised Learning

无监督学习。

## 常用算法

机器学习算法
- 监督学习：
  - 分类（classification）：输出值是离散值；
    - KNN (K-Nearest Neighbors)：K近邻算法；
    - 决策树：
    - Nbayes（朴素贝叶斯）：
  - 回归（regression）：输出值是连续值；

- 无监督学习：
  - 聚类（clustering）：通过相似性度量方法把一些观测值分成同一类，常用于分析数据集；
    - K-means：将 n 个点分为 k 个簇，使得簇内具有较高的相似度，簇间具有较低的相似度；（欧氏距离）
    - DBSCAN（密度聚类）：将 n 个点分为三类，然后删除噪音点；（曼哈顿距离）
      - 核心点：在半径 eps（两个样本被看做邻域的最大举例） 内的点的个数超过 min_samples（簇的样本数）；
      - 边界点：在半径 eps 内的点的个数不超过 min_samples，但落在核心点的邻域内；
      - 噪音点：既不是核心点，也不是边界点；
  - 降维（Dimensionality reduction）：变量降噪，更利于可视化；
    - PCA：主成分分析；


训练集和测试集
交叉验证时：将数据集分成 N 块，使用 N-1 块进行训练，再用最后一块进行测试；


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