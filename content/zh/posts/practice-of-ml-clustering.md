---
title: "机器学习例子 - 聚类问题"
date: 2023-08-07T18:36:17Z
draft: false
description: 直接使用 sklearn.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Machine Learning
- sklearn
- Clustering
categories:
- Practice
libraries:
- mathjax
---

## K-means

```python
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

# 模拟测试数据
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

## DBSCAN
