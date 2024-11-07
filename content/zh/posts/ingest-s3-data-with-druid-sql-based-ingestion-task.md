---
title: "使用 Druid SQL-based ingestion 批量摄取 S3 数据"
date: 2023-04-16T02:38:06Z
draft: false
description: S3 是 .csv.gz 格式，Druid 是 segments 格式。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Apache Druid
- S3
categories:
- OLAP
---

本文旨在将来自 S3 的 .csv.gz 数据，批量摄取至 Druid. 其中：

- Apache Druid: `26.0.0`
- 参考文档：
  - <a href="https://druid.apache.org/docs/latest/multi-stage-query/index.html" target="_blank">SQL-based ingestion</a>
  - <a href="https://druid.apache.org/docs/latest/ingestion/native-batch-input-sources.html#s3-input-source" target="_blank">S3 input source</a>

## REPLACE all data

```sql
REPLACE INTO <target table>
OVERWRITE ALL
< SELECT query >
PARTITIONED BY <time granularity>
[ CLUSTERED BY <column list> ]
```

## REPLACE specific time ranges

```sql
REPLACE INTO <target table>
OVERWRITE WHERE __time >= TIMESTAMP '<lower bound>' AND __time < TIMESTAMP '<upper bound>'
< SELECT query >
PARTITIONED BY <time granularity>
[ CLUSTERED BY <column list> ]
```