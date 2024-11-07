---
title: "BI 方案：Firebase + GCS + BigQuery + Looker Studio"
date: 2023-02-28T16:02:30Z
draft: false
description: 【太贵弃之】将 Firebase 原始数据可视化至 Looker Studio.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Firebase
- GCS
- BigQuery
- Looker Studio
categories:
- BI
- OLAP
- BAAS
---

🙇‍♀️ 本文是个文章地图索引。

本文旨在将来自 Firebase/BigQuery 的原始数据可视化在 Looker Studio. 其中，不同的工具分工如下：

- **Firebase**：
  - BAAS；
  - 用于收集原始数据；
- **GCS**：
  - 云存储，Google 生态；
  - 用于存储原始数据；
- **BigQuery**：
  - OLAP 数据库，列式存储，Google 生态；
- **Looker Studio**：
  - Google 生态的可视化工具；
  - 可直接连接 BigQuery，Google 生态；

## Step1. 收集原始数据

本文使用的是 **Firebase**.

👇 指路我的另外一篇文章 <a href="https://mollywangup.com/posts/tracking-logevent-and-setuserproperty-with-firebase-sdk/" target="_blank">使用 Firebase 统计事件&设置用户属性</a>

## Step2. 原始数据至数仓

本文使用的是 **GCS + BigQuery**.

仅需在 GA 后台设置导出至 BigQuery，即可实现自动将原始数据存储在 GCS 并存储至 BigQuery.

## Step3. 可视化

本文使用的是 **Looker Studio**.

傻瓜式操作，见 [Connect to Data](https://lookerstudio.google.com/data)

## 附：原始数据清洗 SQL

👉 指路我的另外一篇文章 <a href="https://mollywangup.com/posts/common-dimensions-and-metrics-based-on-bigquery-raw-data/" target="_blank">基于 BigQuery 原始数据的指标体系</a>