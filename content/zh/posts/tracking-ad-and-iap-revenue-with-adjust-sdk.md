---
title: "使用 Adjust SDK 追踪广告&内购收入"
date: 2023-02-02T06:06:12Z
draft: false
description: Tracking Ad and IAP revenue with Adjust SDK.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
categories:
- SDK
- MMP
- S2S
---

## 背景信息

两个目标：
1. 需要在 Adjust 面板查看广告收入数据；
2. 需要在 Adjust 面板查看内购收入数据；

## 追踪广告收入

### 方式一（SDK-to-SDK方式）（推荐）

#### 方法描述

MAX SDK 可获取 [Impression-Level User Revenue](https://dash.applovin.com/documentation/mediation/android/getting-started/advanced-settings#impression-level-user-revenue-api)，通过 SDK-to-SDK 的方式，将 MAX SDK 的 **`ad revenue`** 转发给 Adjust SDK.
参考：
```C#
// Adjust SDK initialization
AdjustConfig adjustConfig = new AdjustConfig("{YourAppToken}", AdjustEnvironment.Sandbox);
adjustConfig.setSendInBackground(true);
Adjust.start(adjustConfig);

// ...

// pass MAX SDK ad revenue data to Adjust SDK
public static void OnInterstitialAdRevenuePaidEvent(string adUnitId)
{
    var info = MaxSdk.GetAdInfo(adUnitId);

    var adRevenue = new AdjustAdRevenue(AdjustConfig.AdjustAdRevenueSourceAppLovinMAX);
    adRevenue.setRevenue(info.Revenue, "USD");
    adRevenue.setAdRevenueNetwork(info.NetworkName);
    adRevenue.setAdRevenueUnit(info.AdUnitIdentifier);
    adRevenue.setAdRevenuePlacement(info.Placement);

    Adjust.trackAdRevenue(adRevenue);
}
```

#### 参考文档

1. Adjust: [Get real-time data using SDK postbacks](https://help.adjust.com/en/article/applovin-max#set-up-tracking-with-applovin)
2. GitHub: [Track AppLovin MAX ad revenue with Adjust SDK](https://github.com/adjust/unity_sdk/blob/master/doc/english/sdk-to-sdk/applovin-max.md)

#### 优缺点

- 优点：实时；
- 缺点：需要开发且发版；

### 方式二（API Key）

#### 方法描述

将MAX后台的`report key`填到 Ajust 后台，本质是通过 API 的形式**每天从 MAX 下载一次数据**，然后同步至 Adjust 面板；

#### 参考文档

Adjust: [Connect Adjust to your AppLovin MAX account](https://help.adjust.com/en/article/applovin-max#set-up-tracking-with-applovin)

<img src='/images/posts/connect-adjust-to-your-applovin-MAX-account.png' alt='Connect Adjust to your AppLovin MAX account'>

#### 优缺点

- 优点：快速，成本低；
- 缺点：非实时；

## 追踪内购收入

### 方式一（SDK方式）（推荐）

#### 方法描述

1. 通过 Adjust SDK 手动统计一个内购事件如`purchase`，并为其设置金额和币种参数。
  参考：
    ```C#
    AdjustEvent adjustEvent = new AdjustEvent("8u8bek");
    adjustEvent.setRevenue(0.01, "USD");
    adjustEvent.setTransactionId("transactionId");
    Adjust.trackEvent(adjustEvent);
    ```
2. 关于事件`purchase`的补充说明：
   - **event token**: `8u8bek` （已在Adjust后台创建）；
    <img src='/images/posts/event-token-8u8bek.png' alt='Event token example'>
   - **`setRevenue`**：币种需要设置为`USD`，即默认币种；
   - **`setTransactionId`**：为了防止重复统计内购收入，可设置为`Google Transaction ID`；即使用谷歌支付时谷歌生成的订单唯一标识；
   - 其余参数：现阶段暂时不加，因为即使加了，面板上也无法查看，只能通过导出raw data的方式；
        > https://help.adjust.com/en/article/raw-data-exports
        > https://github.com/adjust/unity_sdk/blob/master/README.md#et-revenue
        > In addition to the data points the Adjust SDK collects by default, you can use the Adjust SDK to track and add as many custom values as you need (user IDs, product IDs, etc.) to the event or session. Custom parameters are only available as raw data and will not appear in your Adjust dashboard.

#### 参考文档

1. [Track revenue events (with the Adjust SDK)](https://help.adjust.com/en/article/revenue-events#track-revenue-events)
2. [Unity SDK of Adjust](https://github.com/adjust/unity_sdk/blob/master/README.md#et-revenue)

#### 优缺点

- 优点：快速，开发成本低；
- 缺点：断网延迟等；

### 方式二（S2S方式）

#### 方法描述

自备服务器，且需要设置跟 Adjust 沟通的参数（见 [Required parameters](https://help.adjust.com/en/article/server-to-server-events#required-parameters) ），当发生内购事件时，Adjust 服务器发给我们服务器；

#### 参考文档

1. [Track revenue events (server-to-server)](https://help.adjust.com/en/article/revenue-events#track-revenue-events-sts)
2. [Server-to-server (S2S) events](https://help.adjust.com/en/article/server-to-server-events#set-up-s2s-security)

#### 优缺点

- 优点：效率和准确性更高；
- 缺点：需要自备服务器； 