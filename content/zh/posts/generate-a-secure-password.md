---
title: "快速生成一个安全的随机密码"
date: 2021-03-16T01:15:43Z
draft: false
description: 选择困难症患者的福音。使用的是 OpenSSL 和 pwgen.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Linux
categories:
- Tool
---

专业密码管理的工具有很多，比如 1Password/LastPass，以下仅仅是为了方便 **快速生成一个安全的随机密码**。

## 使用 OpenSSL

使用以下命令行生成的密码形如：`T1W+MDI0nf1d0XZyiJze1Q==`

```shell
openssl rand -base64 16
```

{{< expand "👇 未安装 OpenSSL 的看这里" >}}

```shell
# for macOS
brew install openssl

# for Debian
sudo apt-get install openssl
```

{{< /expand >}}

## 使用 pwgen

使用以下命令行生成的密码形如：`shohTh7zoYooRi9c`

```shell
pwgen -c -n -B -1 16
```

其中，常用参数如下：

```plaintext
-c：指定生成的密码包含大小写字母。
-n：指定生成的密码包含数字。
-y：指定生成的密码包含符号，例如!@#$%^&*()_+-={}[]|:;"'<>,.?/等。
-B：指定生成的密码不能包含斜杠（/）字符。
-s：指定生成的密码只包含字符，没有数字或符号。
-1：指定生成一行密码，而不是多行密码。
<length>：指定生成密码的长度，默认为8。
```

{{< expand "👇 未安装 pwgen 的看这里" >}}
```shell
# for macOS
brew install pwgen

# for Debian
sudo apt-get install pwgen
```
{{< /expand >}}