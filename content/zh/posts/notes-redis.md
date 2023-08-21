---
title: "学习笔记：Redis"
date: 2023-08-13T13:54:47Z
draft: true
description: 
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Redis
categories:
- DB
- Notes
---

✍ 本文作为学习笔记。

## 安装 

{{< tabs macOS >}}
{{< tab >}}

```shell
brew install redis
redis-cli -v # redis-cli 7.0.12
```

{{< /tab >}}
{{< /tabs >}}

## 常用命令

```shell
redis-cli
redis-cli --raw
```

### Key

```shell
GET name
DEL name
EXISTS name

# 查找所有键
KEYS *
KEYS *me

# 删除所有键
FLUSHALL 

# 查看过期时间
TTL name

# 设置过期时间
EXPIRE name 3600
```

### String

```shell
SET name Molly

# 设置带有过期时间的
SETEX name 3600 Molly
```

## List

列表，有序。

```shell
# 插入
LPUSH letter a
LPUSH letter b c d e f
RPUSH letter g h

# 删除
LPOP letter # 删除左边第1个元素
RPOP letter 3 # 删除右边3个元素
LTRIM letter 0 3 # 删除指定范围内以外的元素，即保留指定范围内的元素

# 查看列表中所有元素
LRANGE letter 0 -1

# 查看长度
LLEN letter
```

## Set

集合，不重复，无序。

```shell
# 添加
SADD course Redis

# 查看集合中所有元素
SMEMBERS course

# 判断元素是否在集合中
SISMEMBER course Redis

# 删除集合中的元素
SREM course Redis

# 集合运算
SINTER
SUNION
SDIFF
```

## SortedSet/ZSet

有序集合，不重复，有序。

```shell
# 添加
ZADD key score member

# 查看有序集合中所有元素
ZRANGE key 0 -1 WITHSCORES

# 查看某个成员的分数
ZSCORE key member

# 查看某个成员的排名
ZRANK key member
ZREVRANK key member
```

## Hash

哈希。

```shell
HSET key field value
HGET key field
HGETALL key
HDEL key field value
HEXISTS key field

# 例子
HSET profile name molly
HGET profile name
HGETALL profile
HDEL profile name molly
HEXISTS profile name
```