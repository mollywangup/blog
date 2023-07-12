---
title: "通用：上架 GP 常用 SDK/Service 集成需求"
date: 2023-03-29T09:35:45Z
draft: false
description: Firebase, Facebook, MAX, Adjust, Helpshift, APT
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Firebase
- Facebook
- MAX
- Adjust
- IAP & Subscription
- Helpshift
- APT
categories:
- SDK
---

## Firebase SDK

理解：Firebase 项目实际上只是一个启用了额外的 Firebase 特定配置和服务的 Google Cloud 项目；

<img src='/images/posts/firebase-projects-hierarchy_projects-apps-resources.png' alt='Firebase Project' style="max-width:60%; height:auto;">

### 常用功能

1. 分析功能：事件统计和设置用户属性（Log Events & setUserProperty）；
   
   <img src='/images/posts/relationship-between-GA-and-firebase.png' alt='relationship-between-GA-and-firebase'>
2. 统计bug/崩溃等（Firebase Crashlytics），且支持自定义 key 细化发生场景；
3. 远程控制（Remote Config）：
   
   <img src='/images/posts/modify-remote-config-programmatically.png' alt='Remote Config'>
   <img src='/images/posts/propagate-remote-config-updates-in-real-time.png' alt='Cloud Functions'>
4. 收集启动时长，监控网络请求等性能数据（Performance）：
   
   <img src='/images/posts/firebase-performance.png' alt='Performance'>
5. 接入多种登录方式如Facebook/PlayGames等（Firebase Authentication功能）；

### 官方文档

1. [Firebase] Google Analytics for Unity：
   - [Get started](https://firebase.google.com/docs/analytics/unity/start)
   - [Log events](https://firebase.google.com/docs/analytics/unity/events)
   - [Set user properties](https://firebase.google.com/docs/analytics/unity/properties)
2. [Firebase] Firebase Crashlytics：
   - [Get started](https://firebase.google.com/docs/crashlytics/get-started?platform=unity)
   - [Add custom keys](https://firebase.google.com/docs/crashlytics/customize-crash-reports?platform=unity#add-keys)
3. [Firebase] Remote Config：
   - [Get started with Firebase Remote Config](https://firebase.google.com/docs/remote-config/get-started?&platform=android)
4. [Firebase] Performance Monitoring：
   - [Get started with Performance Monitoring for Android](https://firebase.google.com/docs/perf-mon/get-started-android)
5. [Firebase] Firebase Authentication：
   - [Get Started with Firebase Authentication in Unity](https://firebase.google.com/docs/auth/unity/start)
   - [Facebook Login](https://firebase.google.com/docs/auth/unity/facebook-login)
   - [Play Games Login](https://firebase.google.com/docs/auth/unity/play-games)

## Facebook SDK

### 常用功能

1. 事件统计功能（Log Events）：只要接了Facebook SDK，就等同于可实现在Facebook Ads上进行推广；
   - 自动事件统计功能；
   - 手动事件统计功能；
2. 登录功能（Facebook Login）：可直接接入，也可通过其他集成服务如Firebase接；
3. 分享功能（Sharing）；
4. 深度链接（Deep Link）；

### 官方文档

1. [Facebook] ：
   - [Getting Started with the Facebook Unity SDK](https://developers.facebook.com/docs/unity/gettingstarted)
   - [How to Log App Events](https://developers.facebook.com/docs/app-events/unity#get-started---unity)
2. [GitHub] [Facebook SDK for Unity](https://github.com/facebook/facebook-sdk-for-unity)

### 测试方法

面向开发：
- [Enabling Debug Logs](https://developers.facebook.com/docs/app-events/getting-started-app-events-android#enabling-debug-logs)

面向运营：
- 方法一：App Ads Helper
  https://developers.facebook.com/tools/app-ads-helper/?id=790833925449113
- 方法二：Events Manager
  https://business.facebook.com/events_manager2/list/app/790833925449113/overview?act=518122528886487&date=2022-08-22_2022-09-04

### 注意事项

创建 Facebook 开发者账号时，需要以下两个信息（由研发反馈）：
1. GP 正式包的 key 的哈希值；
2. 启动 Facebook SDK 的类名；

## MAX SDK

### 常用功能

1. 作为广告聚合平台（Mediation）：
   - 接入不同的广告格式（Ad Formats）：Rewarded/Interstitial 等；
   - 接入多个广告源（Ad Networks）：
     - 自有广告源：AppLovin；
     - 非自有广告源：AdMob/Meta/Unity/Vungle等几十个；
2. 获取广告收入；

### 官方文档

1. [MAX]：
   - [Integrate MAX for Unity](https://dash.applovin.com/documentation/mediation/unity/getting-started/integration)
   - Ad Formats:
     - [Rewarded](https://dash.applovin.com/documentation/mediation/unity/ad-formats/rewarded-ads)
     - [Interstitial](https://dash.applovin.com/documentation/mediation/unity/ad-formats/interstitials)
   - Ad Networks：
     - [Meta Audience Network](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/facebook)
     - [AdMob bidding](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/google)
     - [AdMob waterfall](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/admob)
     - [Unity Ads](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/unityads)
     - [Liftoff Monetize](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/liftoff) (原Vungle)
     - [Chartboost](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/chartboost)
     - [DT Exchange](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/fyber-marketplace) (原AdColony)
     - [Mintegral](https://dash.applovin.com/documentation/mediation/unity/mediation-setup/mintegral)
2. [MAX] [Impression-Level User Revenue API](https://dash.applovin.com/documentation/mediation/unity/getting-started/advanced-settings#impression-level-user-revenue-api)

### 测试方法

使用 Mediation Debugger 也就是测试套件：

1. 官方网站：[Displaying the Mediation Debugger](https://dash.applovin.com/documentation/mediation/unity/testing-networks/mediation-debugger#displaying-the-mediation-debugger)

    ```C#
    MaxSdkCallbacks.OnSdkInitializedEvent += (MaxSdkBase.SdkConfiguration sdkConfiguration) => {
        // Show Mediation Debugger
        MaxSdk.ShowMediationDebugger();
    };
    ```
2. 线下PDF版：<a href="https://mollywangup.com/pdf/Mediation%20Debugger%20%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E%20-%20v23.2.pdf" target="_blank">Mediation Debugger 使用说明 - v23.2</a>

## Adjust SDK

### 常用功能

👉 指路我的另一篇文章 <a href="https://mollywangup.com/posts/tracking-ad-and-iap-revenue-with-adjust-sdk/" target="_blank">使用 Adjust SDK 追踪广告&内购收入</a>

1. 事件统计功能（Log Events）；
2. 手动统计广告收入（Ad Revenue）：方法是将MAX SDK的广告收入转发给Adjust SDK；
3. 手动统计内购收入（IAP）：方法是自定义一个内购收入事件，然后将收入上报至该事件；
   - 本质上属于手动统计的一个事件。区别于普通事件，收入事件：
     - 有货币单位，一律需要换算为USD；
     - 可通过交易ID进行去重；

### 官方文档

1. [Adjust] ：
   - [Integrate Adjust SDK for Unity](https://help.adjust.com/en/article/get-started-unity-sdk)
2. [GitHub] ：
   - Track events：
     - [Track an event](https://github.com/adjust/unity_sdk/blob/master/README.md#track-an-event)
     - [Add Event parameters](https://github.com/adjust/unity_sdk/blob/master/README.md#event-parameters)
   - Track ad revenue：
     - [Track AppLovin MAX ad revenue with Adjust SDK](https://github.com/adjust/unity_sdk/blob/master/doc/english/sdk-to-sdk/applovin-max.md)
     - [Ad revenue tracking](https://github.com/adjust/unity_sdk/blob/master/README.md#ad-ad-revenue)
   - Track IAP：
     - [Track revenue](https://github.com/adjust/unity_sdk/blob/master/README.md#track-revenue)
     - [Revenue deduplication](https://github.com/adjust/unity_sdk/blob/master/README.md#revenue-deduplication)
     - [Subscription tracking](https://github.com/adjust/unity_sdk/blob/master/README.md#subscription-tracking)

### 测试方法

@运营 配合测试

## IAP & Subscription（Unity）

### 常用功能

1. 非订阅性质的内购；
2. 订阅性质的内购；

### 官方文档

1. [Unity] [Set up and integrating Unity IAP](https://docs.unity3d.com/Packages/com.unity.purchasing@4.7/manual/Overview.html)
2. [Unity] ErrorCode
   - [InitializationFailureReason](https://docs.unity3d.com/Packages/com.unity.purchasing@4.6/api/UnityEngine.Purchasing.InitializationFailureReason.html)
   - [PurchaseFailureReason](https://docs.unity3d.com/Packages/com.unity.purchasing@4.6/api/UnityEngine.Purchasing.PurchaseFailureReason.html)

## Helpshift SDK

### 常用功能

1. 客服；
2. FAQs；
3. ...

### 官方文档

[Helpshift] [Integrating Contact Us & In App Messaging](https://developers.helpshift.com/sdkx-unity/support-tools-android/#conversation-view)

## APT（Android Performance Tuner）

### 常用功能

监控游戏性能；

### 官方文档

[Android] [Overview of Android Performance Tuner (Unity)](https://developer.android.com/games/sdk/performance-tuner/unity)

## Play Install Referrer API

### 常用功能

获取用户来源，仅限安卓；（主要通过解析`referrerUrl`）

👉 指路我的另一篇文章 <a href="https://mollywangup.com/posts/how-to-use-play-install-referrer-api-to-retrieve-traffic-source/" target="_blank">使用 Play Install Referrer API 解析 Facebook Campaign</a>

### 官方文档
1. [Android] [Play Install Referrer Library](https://developer.android.com/google/play/installreferrer/library)
2. [Facebook] [Understand Facebook App Ads Referral URLs](https://developers.facebook.com/docs/app-ads/install-referrer/#step-3--decrypt-your-data)