---
title: "在 Linux 上添加定时任务"
date: 2023-07-11T02:19:20Z
draft: false
description: Crontab task.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Crontab
- Linux
- Shell
categories:
- Tool
---

## Step1. 安装Crontab

{{< tabs Linux macOS>}}
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
{{< tab >}}

```shell
```

{{< /tab >}}
{{< /tabs >}}

## Step2. 编写定时任务

### 1. 编辑crontab文件

```shell
crontab -e
```

### 2. 设置定时任务

示例任务：每天凌晨清除该路径下的日志文件；

```crontab
0 0 * * * sudo rm /opt/druid/apache-druid-26.0.0/log/*.log
```

### 3. 保存并关闭
