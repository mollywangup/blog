---
title: "使用 Adjust 追踪事件和收入数据"
date: 2023-02-02T06:06:12Z
draft: false
description: 广告收入通过聚合 SDK 转发而来（额外收费），内购收入通过设置带有金额和币种参数的普通事件而来，订阅收入有专门的 subscription API（额外收费）。
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

本文旨在使用 Adjust SDK 追踪以下四类事件数据：

1. 普通事件（指非收入事件）；
2. 广告收入；
3. 内购收入（一次性）；
4. 订阅收入（周期性）；

<br>
{{< notice info >}}
💡 理论上，收入事件 = 设置了金额和币种参数的普通事件，所以额外收费的**广告收入**和**订阅收入**服务，是可以作为一个普通的收入事件上报的（此方法本文已略）。
{{< /notice >}}

## 追踪普通事件

### 方法描述

在 Adjust 后台为每个事件创建一个 event token，然后直接上报即可。

```C#
AdjustEvent adjustEvent = new AdjustEvent("abc123");
Adjust.trackEvent(adjustEvent);
```

### 参考文档

1. [Adjust] [Create an event token](https://help.adjust.com/en/article/basic-event-setup#create-an-event-token)
2. [GitHub] [Track an event](https://github.com/adjust/unity_sdk#track-an-event)


## 追踪广告收入

共两种方式，推荐 SDK-to-SDK 方式。

### 方式一（SDK-to-SDK方式）（推荐）

#### 方法描述

MAX SDK 可获取 [Impression-Level User Revenue](https://dash.applovin.com/documentation/mediation/android/getting-started/advanced-settings#impression-level-user-revenue-api)，通过 SDK-to-SDK 的方式，将 MAX SDK 的 **`ad revenue`** 转发给 Adjust SDK.

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

1. [Adjust] [Get real-time data using SDK postbacks](https://help.adjust.com/en/article/applovin-max#set-up-tracking-with-applovin)
2. [GitHub] [Track AppLovin MAX ad revenue with Adjust SDK](https://github.com/adjust/unity_sdk/blob/master/doc/english/sdk-to-sdk/applovin-max.md)

<!-- #### 优缺点

- 优点：实时；
- 缺点：需要开发且发版； -->

### 方式二（API Key）

#### 方法描述

将 MAX 后台的`report key`填到 Ajust 后台，本质是通过 API 的形式**每天从 MAX 下载一次数据**，然后同步至 Adjust 面板；

#### 参考文档

[Adjust] [Connect Adjust to your AppLovin MAX account](https://help.adjust.com/en/article/applovin-max#set-up-tracking-with-applovin)

<img src='/images/posts/connect-adjust-to-your-applovin-MAX-account.png' alt='Connect Adjust to your AppLovin MAX account'>

<!-- #### 优缺点

- 优点：快速，成本低；
- 缺点：非实时； -->

## 追踪内购收入

### 方式一（SDK方式）

#### 方法描述

创建一个普通事件如 `purchase`，在上报时，为其设置金额和币种参数，即可被 Adjust 识别为收入事件。

```C#
AdjustEvent adjustEvent = new AdjustEvent("abc123");
adjustEvent.setRevenue(0.01, "USD");
adjustEvent.setTransactionId("transactionId");
Adjust.trackEvent(adjustEvent);
```

{{< alert theme="info" >}}
补充说明：
**`setRevenue`**：币种需要设置为`USD`，即默认币种；
**`setTransactionId`**：为了防止重复统计内购收入，可设置为订单唯一标识；
{{< /alert >}}

#### 参考文档

1. [Adjust] [Track revenue events (with the Adjust SDK)](https://help.adjust.com/en/article/revenue-events#track-revenue-events)
2. [GitHub] [Ad revenue tracking](https://github.com/adjust/unity_sdk#ad-revenue-tracking)

<!-- #### 优缺点

- 优点：快速，开发成本低；
- 缺点：断网延迟等； -->

### 方式二（S2S方式）

#### 方法描述

自备服务器，且需要设置跟 Adjust 沟通的参数（见 [Required parameters](https://help.adjust.com/en/article/server-to-server-events#required-parameters) ），当发生内购事件时，Adjust 服务器发给我们服务器；

#### 参考文档

1. [Adjust] [Track revenue events (server-to-server)](https://help.adjust.com/en/article/revenue-events#track-revenue-events-sts)
2. [Adjust] [Server-to-server (S2S) events](https://help.adjust.com/en/article/server-to-server-events#set-up-s2s-security)

<!-- #### 优缺点

- 优点：效率和准确性更高；
- 缺点：需要自备服务器；  -->

## 追踪订阅收入

### 方法描述

构造 `subscription` 对象，直接上报即可。

{{< alert theme="warning" >}}
⚠️ 注意：`price` 为 long 类型，假定订阅价格是 $9.99，则需要上报为 `9.99 * 1000000 = 9990000`，详见 [getPriceAmountMicros](https://developer.android.com/reference/com/android/billingclient/api/ProductDetails.PricingPhase#getPriceAmountMicros())
{{< /alert >}}

```C#
AdjustPlayStoreSubscription subscription = new AdjustPlayStoreSubscription(
    price,
    currency,
    sku,
    orderId,
    signature,
    purchaseToken);
subscription.setPurchaseTime(purchaseTime);

Adjust.trackPlayStoreSubscription(subscription);
```

### 参考文档

1. [Adjust] [Measure subscriptions](https://help.adjust.com/en/article/measure-subscriptions-react-native-sdk)
2. [GitHub] [Subscription tracking](https://github.com/adjust/unity_sdk#subscription-tracking)
