---
title: "使用 Adjust SDK 追踪卸载和重装"
date: 2023-05-04T09:41:20Z
draft: false
description: Tracking uninstall and reinstall with Adjust SDK.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
- FCM
categories:
- SDK
- MMP
---

## 背景信息

需要手动通过 Adjust SDK 追踪卸载和重装，其中 Adjust 需要 Firebase 的 FCM SDK 来实现；

## 方法概述

### 官方文档

[Set up uninstall and reinstall measurement in the Adjust dashboard](https://help.adjust.com/en/article/set-up-uninstalls-reinstalls)

### 准备工作

1. 在游戏内接入Firebase的FCM: [Set up a Firebase Cloud Messaging client app on Android](https://firebase.google.com/docs/cloud-messaging/android/client)
   - 主要目标是：
     - 接入FCM SDK；
     - 获取 registration token，也就是Adjust需要的那个`pushNotificationsToken`；
2. 在 Adjust SDK 设置 [Push tokens](https://help.adjust.com/en/article/push-tokens-android-sdk)；
3. 准备 FCM server key，并配置到 Adjust 后台；
   <img src='/images/posts/FCM-server-key-config.png' alt='FCM server key config example'>