---
title: "学习笔记：ClickHouse"
date: 2024-11-05T07:46:01Z
draft: false
description: 大部分语法兼容 MySQL, 别名查询很方便也是个坑。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- ClickHouse
categories:
- OLAP
- Notes
---

## 数据库

### 创建数据库

```sql
CREATE DATABASE test
```

### 删除数据库

```sql
DROP DATABASE test
```

## 数据表

### CREATE

```sql
CREATE TABLE IF NOT EXISTS test.my_first_table (
    `event_name` String,
    `created_at` DateTime,
    `user_id` UInt32,
    `country` String,
    `login_type` Nullable(Enum('Google' = 1, 'Facebook' = 2, 'Apple' = 3, 'Vistor' = 4)),
    `revenue_usd` Nullable(Float64)
) 
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (created_at, event_name, user_id)
SETTINGS index_granularity = 8192
```

### DROP

```sql
DROP TABLE my_first_table
```

### INSERT

```sql
INSERT INTO my_first_table (*) VALUES ('install', '2024-11-01 13:14:45', 1000034, 'SA', 2, NULL)
```

### ALTER

```sql
-- 新增列
ALTER TABLE my_first_table ADD COLUMN status Nullable(UInt8)
ALTER TABLE my_first_table ADD COLUMN city Nullable(String) AFTER country

-- 修改列类型
ALTER TABLE my_first_table MODIFY COLUMN status Nullable(Enum('离线' = 0, '忙碌' = 1, '空闲' = 2, 'Live' = 3))

-- 修改列取值
ALTER TABLE my_first_table UPDATE login_type = 1 WHERE user_id = 1000034

-- 删除列
ALTER TABLE my_first_table DROP COLUMN city
```

### SELECT

#### 查看分区

```sql
SELECT
    *
FROM system.parts
WHERE table = 'my_first_table'
ORDER BY
    partition DESC
LIMIT 3
FORMAT Vertical
```

#### 查询当前时区

```sql
SELECT timezoneOf(now())
```

#### 查看 Host

```sql
SELECT version(), hostName(), currentDatabase(), currentUser()
```

#### 设置查询超时时长

```sql
SELECT COUNT(*) FROM my_first_table SETTINGS max_execution_time=1
```

#### 检测是否存在某列

```sql
SELECT hasColumnInTable('test', 'my_first_table', 'user_id')
```

## 常用函数

### 字符串函数

```sql
SELECT splitByChar('_', 'abc_def_12_')[3]
```

### 日期时间函数

```sql
SELECT
  toDateTime(0),
  toDateTime(NULL),

  toDateTime(1713265060),
  fromUnixTimestamp(1713265060),
  toDateTime(1713265060, 'Asia/Shanghai'),

  toDate(NOW(), 'Asia/Shanghai'),
  toDateTime(NOW(), 'Asia/Shanghai'),
  toHour(toDateTime(NOW(), 'Asia/Shanghai')),
  NOW() + INTERVAL 8 HOUR, -- 另一种转时区的方法

  toUnixTimestamp(toDateTime('2024-10-16 03:17:47'))
```

### 其他函数


## 特点与坑

### 惊叹的好用

{{< alert theme="warning" >}}
别名既方便，但也容易冲突。
{{< /alert >}}

```sql
SELECT
  now() AS x,
  toDate(x)
```

### 删除行记录

{{< notice warning >}}
删除效率非常低，建议仅追加写入。
{{< /notice >}}

#### 方法一：DROP PARTITION

```sql
ALTER TABLE my_first_table DROP PARTITION 20240418
```

#### 方法二：DELETE FROM

```sql
DELETE FROM my_first_table WHERE created_at >= '2024-04-19'
``` 