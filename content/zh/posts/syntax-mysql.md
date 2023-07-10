---
title: "MySQL 基本语法"
date: 2021-01-28T06:48:47Z
draft: false
description: MySQL syntax
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- MySQL
categories:
- DB
---

## 初始化登录

### 首次登录并设置密码

```shell
# 首次登录
mysql -uroot
```

```mysql
<!-- 设置密码 -->
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'jywlbj';
```

