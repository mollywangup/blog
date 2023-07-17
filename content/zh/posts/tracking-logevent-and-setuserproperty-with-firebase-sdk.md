---
title: "使用 Firebase 统计事件&设置用户属性"
date: 2022-04-02T06:06:12Z
draft: false
description: 追踪方法，测试方法（竞品调研神器）。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Firebase
categories:
- SDK
- BASS
---

本文旨在使用 Firebase SDK 追踪事件数据。

<br>
{{< notice info >}}
💡 理论上，收入事件 = 设置了金额和币种参数的普通事件，所以额外收费的**广告收入**和**订阅收入**服务，是可以作为一个普通的收入事件上报的（此方法本文已略）。
{{< /notice >}}

## 统计事件

见 [Log events](https://firebase.google.com/docs/analytics/unity/events#log_events_2)

```C#
Firebase.Analytics.FirebaseAnalytics.LogEvent(
  Firebase.Analytics.FirebaseAnalytics.EventSelectContent,
  new Firebase.Analytics.Parameter(
    Firebase.Analytics.FirebaseAnalytics.ParameterItemId, id),
  new Firebase.Analytics.Parameter(
    Firebase.Analytics.FirebaseAnalytics.ParameterItemName, "name"),
  new Firebase.Analytics.Parameter(
    Firebase.Analytics.FirebaseAnalytics.UserPropertySignUpMethod, "Google"),
  new Firebase.Analytics.Parameter(
    "favorite_food", mFavoriteFood),
  new Firebase.Analytics.Parameter(
    "user_id", mUserId)
);
```

## 设置用户属性

见 [Set user properties](https://firebase.google.com/docs/analytics/unity/properties#set_user_properties_2)

```C#
Firebase.Analytics.FirebaseAnalytics.SetUserProperty("favorite_food", "ice cream");
```

## 测试方法

无论哪种方法，都需要先打开测试机的调试模式，方法见 [Enable debug mode](https://firebase.google.com/docs/analytics/debugview#enable_debug_mode)

```shell
adb shell setprop debug.firebase.analytics.app PACKAGE_NAME
```

### 方式一：ADB

在终端打印日志，见 [View events in the log output](https://firebase.google.com/docs/analytics/unity/events#view_events_in_the_log_output)：

```shell
adb shell setprop log.tag.FA VERBOSE
adb shell setprop log.tag.FA-SVC VERBOSE
adb logcat -v time -s FA FA-SVC
```

### 方式二：DebugView

在 Firebase/GA 后台，打开 DebugView，如下图：

<img src='https://firebase.google.com/static/docs/analytics/images/report.png' alt='DebugView report（图源 Firebase）'>

