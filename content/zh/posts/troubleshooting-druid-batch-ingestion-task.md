---
title: "踩坑：Druid + S3 批量摄取任务中的各种报错"
date: 2023-06-30T07:37:04Z
draft: false
description: Duplicate column entries found, InsertTimeOutOfBounds, The worker that this task is assigned did not start it in timeout.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Apache Druid
- S3
- SQL-based ingestion
- Batch Ingestion Task
categories:
- Troubleshooting
---

## 背景信息

- Apache Druid: `26.0.0`
- Batch ingestion task informations:
  - <a href="https://druid.apache.org/docs/latest/multi-stage-query/index.html" target="_blank">SQL-based ingestion</a>
  - <a href="https://druid.apache.org/docs/latest/ingestion/native-batch-input-sources.html#s3-input-source" target="_blank">S3 input source</a>

## Duplicate column entries found

### 详细报错

```json
{
  "errorCode": "CannotParseExternalData",
  "errorMessage": "Duplicate column entries found : [0, Facebook]"
}
```

### 解决方案

Druid 属于列式存储，出现此问题的根本原因是，**存在名称相同的两列**。因此需要定位到名称相同的两列，并进行手动调整。

我遇到这个问题，是因为 MMP 方写入到 S3 的一手原始数据本身就是有问题的，具体表现为**原始数据表头丢失，导致 Druid 自动识别到存在三列名称都为空的列**。详见下方：

{{< expand "踩坑举例：发生在 S3 的 .csv.gz 原始数据" >}}

- 以下是正常的表头：

  <img src='/images/posts/duplicate_column_entries_normal.png' alt='正常表头'>

- 以下是有问题的表头：
  
  <img src='/images/posts/duplicate_column_entries_err.png' alt='异常表头'>

{{< /expand >}}

## InsertTimeOutOfBounds

### 详细报错

```json
{
  "errorCode": "InsertTimeOutOfBoundsFault",
  "interval": "2023-06-09T00:00:00.000Z/2023-06-10T00:00:00.000Z",
  "errorMessage": "Query generated time chunk [2023-06-09T00:00:00.000Z/2023-06-10T00:00:00.000Z] out of bounds specified by replaceExistingTimeChunks"
}
```

### 解决方案

此问题一般发生在 [REPLACE specific time ranges](https://druid.apache.org/docs/latest/multi-stage-query/reference.html#replace-specific-time-ranges)，即类似下列的任务中：

```sql
REPLACE INTO <target table>
OVERWRITE WHERE __time >= TIMESTAMP '<lower bound>' AND __time < TIMESTAMP '<upper bound>'
< SELECT query >
PARTITIONED BY <time granularity>
[ CLUSTERED BY <column list> ]
```

出现此问题的原因是，查询生成的时间段超出了由 replaceExistingTimeChunks 指定的边界，因此需要检查并修改日期字段。

我遇到这个问题，是因为在上述任务中的 WHERE 语句中，`MILLIS_TO_TIMESTAMP("{created_at}" * 1000)` 的格式转换有问题（具体是没有*1000就直接转时间戳），导致最终的时间戳对应的是`-146136543-09-08T08:23:32.096Z/146140482-04-24T15:36:27.903Z`

## Worker did not start in timeout

### 详细报错

以下已省略其他敏感信息：

```json
{
  "type": "query_controller",
  "errorMsg": "The worker that this task is assigned did not start it in timeout[PT5M]. See overlord and middleMana..."
}
```

### 解决方案

我遇到这个问题，是直接在 Druid 控制后台运行批量摄取任务时发生的。一般情况下是因为服务器存储空间不足。（🙊 来自小公司的小声bb）

以下清理内存的一些常用方法。

👉 定期清除日志文件，指路我的另一篇文章 <a href="https://mollywangup.com/posts/add-crontab-task-on-linux/" target="_blank">使用 Crontab 添加定时任务</a>

{{< tabs Linux >}}
{{< tab >}}

```shell
# 查看日志内存占用大小
df -h
du -sh /var/log/* | sort -hr | head -n 10
du -sh /opt/druid/apache-druid-26.0.0/log/* | sort -hr | head -n 10

# 移除所有的 Druid 的日志文件
sudo rm /opt/druid/apache-druid-26.0.0/log/*.log
```

{{< /tab >}}
{{< /tabs >}}

<!-- ## Max retries exceeded with url: /druid/v2/sql/task/ -->

未完待续 ...
