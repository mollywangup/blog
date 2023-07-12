---
title: "使用 FCM 实现 Push 消息功能"
date: 2023-02-07T02:03:15Z
draft: false
description: FCM 是一个支持跨平台消息传递的免费解决方案。实现的前提是 FCM SDK 为每个新增设备生成唯一标识 registration token，用以消息定位。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- FCM
- Push Notification
categories:
- SDK
---

## 背景信息

### 目标

向用户/玩家发送 Push 消息，需要兼容以下三种状态：

- 在前台 (`foreground`)
- 在后台 (`background`)
- 游戏进程被杀掉 (`killed`)

### 关于 FCM

[FCM Architectural Overview](https://firebase.google.com/docs/cloud-messaging/fcm-architecture)

<img src='/images/posts/diagram-FCM.png' alt='FCM Architectural Overview'>

