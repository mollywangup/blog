---
title: "优化支付 (IAP) 时的异常处理"
date: 2023-02-22T02:07:40Z
draft: false
description: 内置异常共两种，初始化阶段异常和支付阶段异常。网络异常需要手动处理。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- IAP
categories:
- SDK
---

## 背景信息

本文涉及到的是 Unity IAP, 以下是本文的目标：

1. 支付异常时的捕捉及处理；
2. 处理方案概述（三类）：
   - 网络异常：需要交互；
   - 支付异常：需要交互；
   - 其他（取消/重复购买/仅iOS的特殊异常）：暂不处理；

## 具体实现

### 异常列表完善

1. 内置的异常如下：
   - 初始化阶段：[InitializationFailureReason](https://docs.unity3d.com/Packages/com.unity.purchasing@4.6/api/UnityEngine.Purchasing.InitializationFailureReason.html)（3个）
   - 支付阶段：[PurchaseFailureReason](https://docs.unity3d.com/Packages/com.unity.purchasing@4.6/api/UnityEngine.Purchasing.PurchaseFailureReason.html)（8个）
2. 新增的异常如下（初始化阶段之前）：
   - `NetworkUnavailable`：初始化阶段的第一优先级判断，玩家本地无网络连接时；

因此，最终的异常列表如下（异常描述就省略了，🙊 当然不是因为表格太丑的原因删掉的）：

| 类型 | 具体异常  | 处理方案 |
| ---------- | --------- | ---------- |
| 判断网络<br>（初始化前） | `NetworkUnavailable` | 网络异常 |
| InitializationFailureReason<br>（3个） | `AppNotKnown` | 支付失败 |
|  | `NoProductsAvailable` | 支付失败 |
|  | `PurchasingUnavailable` | 支付失败 |
| PurchaseFailureReason<br>（8个） | `DuplicateTransaction` | / |
|  | `ExistingPurchasePending` | / |
|  | `PaymentDeclined` | / |
|  | `ProductUnavailable` | 支付失败 |
|  | `PurchasingUnavailable` | 支付失败 |
|  | `SignatureInvalid` | 支付失败 |
|  | `Unknown` | 支付失败 |
|  | `UserCancelled` | / |

### 异常处理方案

按照捕捉到的具体异常进行分类处理，共如下三类：


## 附：IAP 官方流程

<img src='/images/posts/PurchaseProcessingResult.Complete.png' alt='PurchaseProcessingResult.Complete'>
<br>
<img src='/images/posts/PurchaseProcessingResult.Pending.png' alt='PurchaseProcessingResult.Pending'>
