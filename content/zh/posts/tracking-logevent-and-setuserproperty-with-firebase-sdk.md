---
title: "使用 Firebase 统计事件&设置用户属性"
date: 2022-04-02T06:06:12Z
draft: false
description: 追踪方法，测试方法（竞品调研神器），原始数据结构。
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

本文旨在使用 Firebase SDK 统计事件、设置用户属性，并提供测试方法、附赠原始数据结构举例。

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

{{< alert theme="info" >}}
✍ 😱 emmm... 怎么不算竞品调研神器呢。
{{< /alert >}}

### 方式二：DebugView

在 Firebase/GA 后台，打开 DebugView，如下图：

<img src='https://firebase.google.com/static/docs/analytics/images/report.png' alt='DebugView report（图源 Firebase）'>

## 附：原始数据结构

以下为 `sign_up` 事件对应的 Firebase/BigQuery 原始数据，有助于理解数据结构。

{{< expand "原始数据举例（已脱敏）" >}}

```json
[{
  "event_date": "20230116",
  "event_timestamp": "1673858401132002",
  "event_name": "sign_up",
  "event_params": [{
    "key": "firebase_screen_class",
    "value": {
      "string_value": "UnityPlayerActivity",
      "int_value": null,
      "float_value": null,
      "double_value": null
    }
  }, {
    "key": "method",
    "value": {
      "string_value": "Android",
      "int_value": null,
      "float_value": null,
      "double_value": null
    }
  }, {
    "key": "ga_session_id",
    "value": {
      "string_value": null,
      "int_value": "1673858394",
      "float_value": null,
      "double_value": null
    }
  }, {
    "key": "ga_session_number",
    "value": {
      "string_value": null,
      "int_value": "2",
      "float_value": null,
      "double_value": null
    }
  }, {
    "key": "firebase_screen_id",
    "value": {
      "string_value": null,
      "int_value": "-5164663614086310235",
      "float_value": null,
      "double_value": null
    }
  }, {
    "key": "firebase_event_origin",
    "value": {
      "string_value": "app",
      "int_value": null,
      "float_value": null,
      "double_value": null
    }
  }, {
    "key": "engaged_session_event",
    "value": {
      "string_value": null,
      "int_value": "1",
      "float_value": null,
      "double_value": null
    }
  }],
  "event_previous_timestamp": "1673780157003002",
  "event_value_in_usd": null,
  "event_bundle_sequence_id": "22",
  "event_server_timestamp_offset": "2492698",
  "user_id": null,
  "user_pseudo_id": "8d59ce7133e03f6170eadbce40174c91",
  "privacy_info": {
    "analytics_storage": "Yes",
    "ads_storage": "Yes",
    "uses_transient_token": "No"
  },
  "user_properties": [{
    "key": "ga_session_id",
    "value": {
      "string_value": null,
      "int_value": "1673858394",
      "float_value": null,
      "double_value": null,
      "set_timestamp_micros": "1673858394853000"
    }
  }, {
    "key": "first_open_time",
    "value": {
      "string_value": null,
      "int_value": "1673780400000",
      "float_value": null,
      "double_value": null,
      "set_timestamp_micros": "1673777018672000"
    }
  }, {
    "key": "ga_session_number",
    "value": {
      "string_value": null,
      "int_value": "2",
      "float_value": null,
      "double_value": null,
      "set_timestamp_micros": "1673858394853000"
    }
  }, {
    "key": "player_match_level",
    "value": {
      "string_value": "8",
      "int_value": null,
      "float_value": null,
      "double_value": null,
      "set_timestamp_micros": "1673780429774000"
    }
  }],
  "user_first_touch_timestamp": "1673777018672000",
  "user_ltv": null,
  "device": {
    "category": "mobile",
    "mobile_brand_name": "Xiaomi",
    "mobile_model_name": "M2104K10AC",
    "mobile_marketing_name": "Redmi K40 Gaming Edition",
    "mobile_os_hardware_model": "M2104K10AC",
    "operating_system": "Android",
    "operating_system_version": "Android 11",
    "vendor_id": null,
    "advertising_id": "",
    "language": "zh-cn",
    "is_limited_ad_tracking": "No",
    "time_zone_offset_seconds": "28800",
    "browser": null,
    "browser_version": null,
    "web_info": null
  },
  "geo": {
    "continent": "Asia",
    "country": "China",
    "region": "",
    "city": "",
    "sub_continent": "Eastern Asia",
    "metro": "(not set)"
  },
  "app_info": {
    "id": "PACKAGE_NAME",
    "version": "1.2.8",
    "install_store": null,
    "firebase_app_id": "1:65595447720:android:aa82859441a614a0aba59d",
    "install_source": "com.miui.packageinstaller"
  },
  "traffic_source": {
    "name": "(direct)",
    "medium": "(none)",
    "source": "(direct)"
  },
  "stream_id": "3607414280",
  "platform": "ANDROID",
  "event_dimensions": null,
  "ecommerce": null,
  "items": []
}]
```

{{< /expand >}}