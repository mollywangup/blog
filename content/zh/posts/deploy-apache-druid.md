---
title: "部署 Apache Druid"
date: 2023-07-03T10:58:13Z
draft: true
description: Deploy Apache Druid on a Linux system.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Apache Druid
categories:
- OLAP
---

## 背景信息

- Apache Druid: `26.0.0`

## 基础安装

### 安装

- 下载地址: https://downloads.apache.org/druid/26.0.0/apache-druid-26.0.0-bin.tar.gz
- 官方网站: https://druid.apache.org/docs/latest/tutorials/index.html
- OK了: http://52.62.120.83:8888/unified-console.html

```shell
# 安装
sudo mkdir -p /opt/druid
cd /opt/druid
sudo wget https://downloads.apache.org/druid/26.0.0/apache-druid-26.0.0-bin.tar.gz
sudo tar -xzf apache-druid-26.0.0-bin.tar.gz

https://www.apache.org/dyn/closer.cgi?path=/druid/26.0.0/apache-druid-26.0.0-bin.tar.gz

# 可能需要安装的依赖
sudo yum install perl-core
```

### 启动

```shell
# 启动
sudo /opt/druid/apache-druid-26.0.0/bin/start-druid

# 后台启动
nohup sudo /opt/druid/apache-druid-26.0.0/bin/start-druid > /dev/null 2>&1 &

# 重启
ps aux | grep druid
sudo pkill -f druid
sudo /opt/druid/apache-druid-26.0.0/bin/start-druid
```

### 配置

```shell
# 默认配置文件
cd /opt/druid/apache-druid-26.0.0/conf/druid/auto/_common
# common.jvm.config  common.runtime.properties  log4j2.xml
```

### 日志

```shell
# 查看日志
cd /opt/druid/apache-druid-26.0.0/log/
cat middleManager.log
```

### 修改 deep storage

## 应用

### 配置 index_parallel task


### 连接Superset

[Superset] Install Database Drivers：https://superset.apache.org/docs/databases/installing-database-drivers/


### SQL-based ingestion and multi-stage query task API
