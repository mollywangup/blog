---
title: "GP 包签名管理（Release Key）"
date: 2022-08-23T10:43:11Z
draft: false
description: GP 包正式签名（Release Key）的创建、查看哈希值。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- GP
categories:
- 
---

上架 GP 时，除了内测轨道，必须使用正式签名（Release Key）。

## 创建方法

方法一：IDE直接生成；

方法二：命令行生成；（推荐）

```shell
keytool -genkey -v -keystore <RELEASE_KEY_PATH> -alias <RELEASE_KEY_ALIAS> -storepass <STOREPASS> -keypass <KEYPASS> -keyalg RSA -keysize 2048 -validity 36500
```

## 查看方法

```shell
# 需要输入key store的密码
keytool -v -list -keystore <RELEASE_KEY_PATH>
```

## 获取 Key 的哈希值

方法详见 [Create a Release Key Hash](https://developers.facebook.com/docs/android/getting-started/#release-key-hash)

```shell
keytool -exportcert -alias <RELEASE_KEY_ALIAS> -keystore <RELEASE_KEY_PATH> | openssl sha1 -binary | openssl base64
```