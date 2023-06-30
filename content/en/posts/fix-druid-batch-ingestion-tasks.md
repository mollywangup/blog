---
title: "Fix Druid Batch Ingestion Tasks"
date: 2023-06-30T07:37:04Z
draft: true
description: Duplicate column entries found, 
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Apache Druid
categories:
- Troubleshooting
---

## Duplicate column entries found

详细报错：`errorMsg: CannotParseExternalData: Duplicate column entries found`

### 问题定位

`Apache Druid` 属于列式存储，出现此问题的原因是，存在名称相同的两列；

### 解决方案

定位到名称相同的两列，并手动修改列名称；

{{< expand "发生在 S3 .csv.gz" >}}

#### Title2

contents2

{{< /expand >}}

## 

### 问题定位

    "errorMsg": "The worker that this task is assigned did not start it in timeout[PT5M]. See overlord and middleMana..."


### 解决方案

一般情况下是因为服务器存储空间不足。

{{< tabs Linux MacOS >}}

  {{< tab >}}

  ### Linux section

  ```bash
  df -h
  du -sh /var/log/* | sort -hr | head -n 10
  ```

  {{< /tab >}}
  {{< tab >}}

  ### MacOS section

  Hello world!
  {{< /tab >}}
{{< /tabs >}}