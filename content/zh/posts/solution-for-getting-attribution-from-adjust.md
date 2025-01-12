---
title: "通过 Adjust 获取用户归因的解决方案"
date: 2025-01-12T03:36:40Z
draft: false
description: 共三类：客户端 SDK，服务端回调，设备 API 接口。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
categories:
- MMP
---

## 背景信息

1. 应用有用户注册系统，用户注册成功前/后，需判断/维护用户的归因信息；
2. 用户的归因存在更新的情形，如**再归因**、由 iOS 用户 ATT 授权状态变化导致的的**归因更新**等；
3. 选择方案时，主要考虑准确性+时效性。

## 归因字段说明

| 字段&nbsp;&nbsp;&nbsp; | 说明&nbsp;&nbsp;&nbsp; |
| ---------- | ---------- |
| **{tracker_name}** | 完整归因信息，结构: {network_name}::{campaign_name}::{adgroup_name}::{creative_name} |
| **{network_name}\|{fb_install_referrer_publisher_platform}** | 渠道 |
| **{campaign_name}\|{fb_install_referrer_campaign_group_name}** | 推广计划 |
| **{adgroup_name}\|{fb_install_referrer_campaign_name}** | 推广组 |
| **{creative_name}\|{fb_install_referrer_adgroup_name}** | 推广素材 |

## 三类获取途径

### 方式一：客户端 SDK

- 官方文档：<a href="https://dev.adjust.com/en/sdk/ios/features/attribution" target="_blank">iOS</a>, <a href="https://dev.adjust.com/en/sdk/android/features/attribution" target="_blank">Android</a>
- 延迟说明：官方说法是 Adjust SDK 初始化成功后的 **2~3s**

{{< notice warning >}}
坑：对于 iOS，即使 Adjust 是可以获取到具体归因信息的，但为了响应苹果隐私政策，SDK 返回的归因信息可能是 **No User Consent**，此时可以使用本文其他获取方式作为补救方案。
{{< /notice >}}

### 方式二：服务端回调

- 官方文档：<a href="https://help.adjust.com/en/article/callback-structure-partner" target="_blank">Callback structure</a>
- 回调配置：以下环节均可配置，按需：
  - `安装`
  - `注册成功`（自定义事件）
  - `归因更新`
- 延迟说明：官方说法是 **30s**，但实际操作过程中没有这么久

### 方式三：设备 API 接口

- 官方文档：<a href="https://dev.adjust.com/zh/api/device-api#inspect-device" target="_blank">设备 API</a>
- 延迟说明：官方说法是 **30s**，但实际操作过程中没有这么久
- 例子：

  ```shell
  curl \
  --header "Authorization: Bearer {your_api_token}" \
  -L -X GET "https://api.adjust.com/device_service/api/v2/inspect_device?advertising_id={your_advertising_id}&app_token={your_app_token}"
  ```
  ```json
  {
    "Adid": "acf8534f2f052395e617a38730682ccc",
    "AdvertisingId": "1234-5678-9012-3456",
    "Tracker": "abc123",
    "TrackerName": "Organic",
    "FirstTracker": "zr5vueb",
    "FirstTrackerName": "Organic",
    "Environment": "sandbox",
    "ClickTime": "0001-01-01T00:00:00Z",
    "InstallTime": "2015-08-19T03:42:03Z",
    "LastSessionTime": "2017-07-29T17:29:17Z",
    "LastEventsInfo": {
        "kgfcul": {
          "name": "Copy IDFA",
          "time": "2024-07-18T10:01:16Z"
        },
        "wz9qqz": {
          "name": "Accept Terms",
          "time": "2024-07-01T14:13:47Z"
        },
        "z3f773": {
          "name": "Copy IDFV",
          "time": "2024-07-01T19:01:39Z"
        }
    },
    "LastSdkVersion": "ios4.37.2",
    "State": "installed"
  }
  ```

## 应用 Case

建议实际应用中，以上方式组合使用，以最大化用户归因的准确性。

### Case1: 注册拦截

当需要**根据用户归因信息进行注册拦截**时，可采用如下的方案：

- 优先客户端判断（2~3s）
- 如果客户端有有效返回值，可直接在请求**服务端注册接口**时，携带归因信息
- 如果客户端无有效返回值，此时服务端可通过设备 API 接口实时查询归因信息（30s）
  - 大写加粗强调：如果返回 `No User Consent`（属于设备未授权 ATT 的情形），也属于无有效返回值

### Case2: 注册成功环节获取归因信息

对于安装且注册成功的用户，需要**在数据库维护其归因信息**，可采用如下的方案：

- 先自定义一个 **`注册成功(NEW_USER_REGISTER_SUCCESS)`** 事件
- 然后设置服务端回调，以获取用户的归因信息

可参考下图：

<img src='/images/posts/adjust-callback.png' alt='注册成功事件回调'>

### Case3：iOS ATT 授权导致的归因更新

配置入口如下图，回调参数按需设置：

<img src="/images/posts/adjust-callback-attribution-update.png" alt="归因更新回调配置入口">

## 附：两种 ATT 弹窗形式

### 第一种：先初始化，后弹窗（建议）

先初始化 SDK 再弹 ATT 弹窗

<img src="https://images.ctfassets.net/5s247im0esyq/1cTJmK9fkFRizPLIv0tPDe/c467f82a9ecd5ce17cd166210e799b23/launch_att_consent_sdk_then_ATT-i0259-a02-v01-20220616_zh.png" alt="先初始化，后弹窗（图源 Adjust）">

### 第二种：先弹出，后初始化

先弹 ATT 弹窗再初始化 SDK

<img src="https://images.ctfassets.net/5s247im0esyq/7IRTLpM0CAjj4W1UIbM0fh/8e499a6ffa823d219189d297a0f9d796/launch_att_consent_ATT_then_sdk-i0259-a01-v01-20220616_zh.png" alt="先弹出，后初始化（图源 Adjust）">