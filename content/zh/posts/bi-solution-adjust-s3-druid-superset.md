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
- MMP
---

🙇‍♀️ 本文是个文章地图索引。

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

本文使用的是 **Adjust**.

👇 指路我的另外两篇文章：
- <a href="https://mollywangup.com/posts/tracking-event-and-revenue-with-adjust-sdk/" target="_blank">使用 Adjust 追踪事件和收入数据</a>
- <a href="https://mollywangup.com/posts/implement-uninstalls-and-reinstalls-with-adjust-and-fcm/" target="_blank">使用 Adjust + FCM 追踪卸载和重装</a>

## Step2. 存储原始数据

本文使用的是 **S3**.

👉 指路我的另外一篇文章 <a href="https://mollywangup.com/posts/two-methods-for-exporting-adjust-raw-data/" target="_blank">将 Adjust 原始数据导出的两种方法</a>

## Step3. 转存至数仓

本文使用的是 **Apache Druid**.

👉 指路我的另外一篇文章 <a href="https://mollywangup.com/posts/ingest-s3-data-with-druid-sql-based-ingestion-task/" target="_blank">使用 Druid SQL-based ingestion 批量摄取 S3 数据</a>

## Step4. 可视化

本文使用的是 **Apache Superset**.

- Docker 部署：[apache/superset](https://hub.docker.com/r/apache/superset)
- 支持的数据库：[Supported Databases](https://superset.apache.org/docs/databases/installing-database-drivers)

## 附：原始数据清洗 SQL

👉 指路我的另外一篇文章 <a href="https://mollywangup.com/posts/common-dimensions-and-metrics-based-on-adjust-raw-data/" target="_blank">基于 Adjust 原始数据的指标体系</a>
