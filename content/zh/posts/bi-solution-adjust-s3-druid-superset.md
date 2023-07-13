---
title: "BI 方案：Adjust + S3 + Druid + Superset"
date: 2023-05-07T16:03:28Z
draft: false
description: 将 Adjust 原始数据可视化至 Apache Superset.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
- S3
- Apache Druid
- Apache Superset
- OLAP
- SQL
categories:
- BI
- DB
- Columnar
---

## 背景信息

本文旨在将来自 Adjust 的原始数据可视化在 Superset. 其中，不同的工具分工如下：

- **Adjust**：
  - MMP；
  - 用于收集原始数据；
- **S3**：
  - 云存储，分布式文件系统；
  - 用于存储原始数据；
- **Apache Druid**：
  - 开源的 OLAP 数据库，列式存储，时间序列分析；
  - 可用于批量摄取来自 S3 的原始数据；
  - 可用于将 segments 数据持久化到 S3（建议新建一个专门的存储捅）；
- **Apache Superset**：
  - 开源的可视化工具；
  - 可直接连接 Apache Druid 数据库；

## Step1. 收集原始数据

👉 指路我的另一篇文章 <a href="https://mollywangup.com/posts/tracking-ad-and-iap-revenue-with-adjust-sdk/" target="_blank">使用 Adjust 追踪广告&内购收入</a>

### 事件分类

使用 Adjust 收集原始数据，根据事件类型可分为两类：

普通事件：

收入事件：


### 参数


## Step2. 存储原始数据




## Step3. 转存至数仓




## Step4. 可视化




## 附：原始数据清洗 SQL



