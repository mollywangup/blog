---
title: "踩坑：使用 SSH 时的各种报错"
date: 2023-08-21T09:21:12Z
draft: true
description: Operation timed out (connect to host github.com port 22), ...
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- SSH
- GitHub
categories:
- Troubleshooting
---

以下基于 macOS 系统。

## SSH 连接至 GitHub 超时

### 详细报错

```plaintext
ssh: connect to host github.com port 22: Operation timed out
```

### 解决方案

参考 [ssh: connect to host github.com port 22: Connection timed out](https://stackoverflow.com/questions/15589682/ssh-connect-to-host-github-com-port-22-connection-timed-out)

打开 SSH 配置文件：

```shell
vim ~/.ssh/config
```

然后添加如下配置：

```plaintext
Host github.com
    Hostname ssh.github.com
    Port 443
```

最后运行：

```shell
ssh -T git@github.com
```