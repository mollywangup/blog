---
title: "使用 Adjust + FCM 追踪卸载和重装"
date: 2023-02-02T09:41:20Z
draft: false
description: Adjust SDK 无法独自实现，需要借助 FCM SDK 的消息推送功能来实现。
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

Adjust SDK 无法独自实现追踪卸载和重装，需要借助 FCM SDK 的消息推送功能。

## 实现原理

官方文档：[Uninstall and reinstall measurement](https://help.adjust.com/en/article/uninstalls-reinstalls)

概括起来如下：

1. FCM 会在新用户首次启动时，为该设备生成一个设备标识符，即 `registration token` 以下称作 `push token`，在此应用场景中的作用是**消息定位**该设备；
2. Adjust 获取上述 FCM 生成的 `push token`，并与自己的设备标识相关联；
3. Adjust 通过 FCM 每天向设备发送一个静默的推送消息用于卸载监听，并根据监听结果来判断用户是否发生了卸载或者重装。

	{{< expand "Adjust 服务器发送监听消息给 FCM" >}}

<br>其中 `pushToken` 用于定位设备：

```json
{
   "to":"pushToken",
   "data":{
      "adjust_purpose":"uninstall detection"
   }
}
```

	{{< /expand >}}

<img src='https://images.ctfassets.net/5s247im0esyq/4Zwu2aRZTQFd4A9zE6xP3a/49661ec90acd2eb1514a403979cee16c/6f0fc1af-71b4-4c5c-b3ce-7b57c11fd4b5.png' alt='Retrieve push token from FCM（图源 Adjust）' width=100%>
<br>
<img src='https://images.ctfassets.net/5s247im0esyq/4kWODOnDFoxu4FHdzLxd5I/16857ac2f57226244e868db17b45842d/3cb1008e-021f-40d3-813f-06161c8a2804.png' alt='Check app status through FCM daily（图源 Adjust）' width=100%>

## 具体步骤

1. 准备 FCM server key，并配置到 Adjust 后台：

	<br><img src='/images/posts/FCM-server-key-config.png' alt='FCM server key config example'><br>

2. 接入 FCM SDK：[Set up a Firebase Cloud Messaging client app on Android](https://firebase.google.com/docs/cloud-messaging/android/client)

3. Adjust SDK 获取 [Push tokens](https://help.adjust.com/en/article/push-tokens-android-sdk)：

	> Push tokens are used for Audience Builder and client callbacks. They are also required for uninstall and reinstall tracking.
	
	{{< expand "setPushToken" >}}
```java
// Send the token with context (recommended)
Adjust.setPushToken(pushNotificationsToken, context);
```
	{{< /expand >}}