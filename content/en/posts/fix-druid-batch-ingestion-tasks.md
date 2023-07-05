---
title: "解决 Druid 批量任务中的各种报错"
date: 2023-06-30T07:37:04Z
draft: false
description: Apache Druid batch ingestion tasks, duplicate column entries found
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Apache Druid, SQL-based ingestion
categories:
- Troubleshooting
---

## Duplicate column entries found

- 详细报错：
  ```Prolog
  "errorMsg": "CannotParseExternalData: Duplicate column entries found : [0, Facebook]"
  ```

  {{< expand "踩坑举例：发生在 S3 的 .csv.gz 原始数据" >}}

  - 以下是正常的表头：

    | Inline&nbsp;&nbsp;&nbsp;     | Markdown&nbsp;&nbsp;&nbsp;  | In&nbsp;&nbsp;&nbsp;                | Table      |
    | ---------- | --------- | ----------------- | ---------- |
    | *italics*  | **bold**  | ~~strikethrough~~&nbsp;&nbsp;&nbsp; | `code`     |

    <img src='/images/posts/duplicate_column_entries_normal.png' alt='正常的表头'>

  - 以下是有问题的表头：原始数据表头丢失，导致Druid自动识别到存在两列的列名称都为空；
    
    <img src='/images/posts/duplicate_column_entries_err.png' alt='异常表头'>

  {{< /expand >}}

- 解决方案：
  `Apache Druid` 属于列式存储，出现此问题的根本原因是，**存在名称相同的两列**。需要定位到名称相同的两列，并进行手动调整；

## 

- 详细报错：
  ```Prolog
  "errorMsg": "The worker that this task is assigned did not start it in timeout[PT5M]. See overlord and middleMana..."
  ```

- 解决方案：
  一般情况下是因为服务器存储空间不足（来自小公司的小声bb）。
  以下清理内存的一些常用方法。

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

## Max retries exceeded with url: /druid/v2/sql/task/



## InsertTimeOutOfBounds

Query generated time chunk
