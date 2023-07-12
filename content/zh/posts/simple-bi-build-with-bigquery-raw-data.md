---
title: "BI 方案：Firebase + GCS + BigQuery + Looker Studio"
date: 2023-03-06T16:02:30Z
draft: false
description: （deprecated）简易 BI：将 Firebase 原始数据可视化至 Looker Studio.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Firebase
- GCS
- BigQuery
- Looker Studio
- OLAP
- SQL
categories:
- BI
- DB
- Columnar
- BASS
---

## 背景信息

本文旨在将来自 Firebase 的原始数据可视化在 Looker Studio. 其中，不同的工具分工如下：

- **Firebase**：
  - BASS；
  - 用于收集原始数据；
- **GCS**：
  - 云存储，Google 生态；
  - 用于存储原始数据；
- **BigQuery**：
  - OLAP 数据库，列式存储，Google 生态；
- **Looker Studio**：
  - Google 生态的可视化工具；
  - 可直接连接 BigQuery，Google 生态；

## 原始字段



## 加工字段的SQL语句

### 数据源级别（在作为数据源的Query中）

### 数理统计级别（在Looker Studio中）
