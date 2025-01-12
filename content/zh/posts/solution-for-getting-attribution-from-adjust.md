---
title: "Solution for Getting Attribution From Adjust"
date: 2025-01-12T03:36:40Z
draft: false
description: 从 Adjust 获取用户归因的解决方案。
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

1. 应用有用户注册系统，用户注册成功后，内部数据库需存储用户的归因信息；
2. 用户的归因存在更新的情形，如**再归因**、由 iOS 用户 ATT 授权状态变化导致的的**归因更新**等；
3. 选择方案时，主要考虑准确性+时效性。

## 归因字段说明

| 字段&nbsp;&nbsp;&nbsp; | 说明&nbsp;&nbsp;&nbsp; |
| ---------- | ---------- |
| **{tracker_name}** | 完整归因信息，结构: {network_name}::{campaign_name}::{adgroup_name}::{creative_name} |
| **{network_name}\|{fb_install_referrer_publisher_platform}** | 流量来源 |
| **{campaign_name}\|{fb_install_referrer_campaign_group_name}** | 推广计划 |
| **{adgroup_name}\|{fb_install_referrer_campaign_name}** | 推广组 |
| **{creative_name}\|{fb_install_referrer_adgroup_name}** | 推广素材 |

## 三类获取途径

### 方式一：客户端 SDK

- 官方文档： 
  - <a href="https://dev.adjust.com/en/sdk/ios/features/attribution" target="_blank">iOS</a>
  - <a href="https://dev.adjust.com/en/sdk/android/features/attribution" target="_blank">Android</a>
- 延迟说明：官方说法是 Adjust SDK 初始化成功后的 **2~3s**

### 方式二：服务端回调

- 官方文档：<a href="https://help.adjust.com/en/article/callback-structure-partner" target="_blank">Callback structure</a>
- 回调配置：以下环节均可配置，按需：
  - 安装
  - 注册成功：自定义一个注册成功事件，然后
  - 归因更新
- 延迟说明：官方说法是 **30s**，但实际操作过程中没有这么久

### 方式三：设备 API 接口

- 官方文档：<a href="https://dev.adjust.com/zh/api/device-api#inspect-device" target="_blank">设备 API</a>
- 延迟说明：官方说法是 **30s**，但实际操作过程中没有这么久

### 总结

建议实际应用中，以上方式组合使用，以最大化用户归因的准确性。如：

- 优先客户端判断（2~3s）
- 如果客户端有有效返回值，可直接在请求服务端注册接口时，携带归因信息
  - 对于 iOS 应用，unattributed 近似等于 Facebook 广告量
- 如果客户端无有效返回值，此时服务端可通过设备 API 接口实时查询归因信息（30s）
  - 大写加粗强调：如果返回 no user consent（属于设备未授权 ATT 的情形），也属于无有效返回值，仍需要继续服务端尝试调接口获取

## 附：两种 ATT 弹窗形式

