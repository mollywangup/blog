---
title: "BI 方案：Adjust + S3 + Druid + Superset"
date: 2023-05-07T16:03:28Z
draft: false
description: 简易 BI：将 Adjust 原始数据可视化至 Apache Superset.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
- S3
- Druid
- Superset
categories:
- OLAP
- BI
---

## 背景信息

本文旨在将来自 Adjust 的原始数据可视化在 Superset;

- Adjust: 收集原始数据；
- S3: 存储原始数据；
- Apache Druid: 
  - 批量摄取来自 S3 的原始数据；
  - 将 segment 数据持久化到 S3（建议新建一个专门的存储捅）；
- Apache Superset: 开源的可视化工具；

## Step1. 收集原始数据



## Step2. 存储原始数据


## Step3. 转存原始数据


## Step4. 可视化




