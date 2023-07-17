---
title: "使用 Crontab 添加定时任务"
date: 2021-07-11T02:19:20Z
draft: false
description: 使用 Crontab 创建简单的定时任务，适用于 macOS/Linux.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Crontab
- Linux
categories:
- Tool
---

## Step1. 安装 Crontab

macOS 一般系统自带，可以直接下一步。

{{< tabs Linux >}}
{{< tab >}}

```shell
# 安装
sudo yum install cronie

# 启动服务
sudo service crond start

# 开机自启
sudo chkconfig crond on
```

{{< /tab >}}
{{< /tabs >}}

## Step2. 编写定时任务

### 1. 编辑 crontab 文件

```shell
crontab -e
```

### 2. 设置定时任务

示例任务：每天凌晨清除该路径下的日志文件；
其中，前五个位置表示五个时间字段，依次是：分钟、小时、日期、月份、星期几；

```plaintext
0 0 * * * sudo rm /path/*.log
```

### 3. 保存并关闭

<kbd><kbd>ESC</kbd>+<kbd>:wq</kbd></kbd>

### 4. 查看任务列表

```shell
crontab -l
```
