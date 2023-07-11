---
title: "使用 Play Install Referrer API 解析 Facebook Campaign"
date: 2022-10-25T06:17:06Z
draft: false
description: Use Play Install Referrer API to retrieve traffic source.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Play Install Referrer API
- Facebook Ads
- Firebase
categories:
- Attribution
---

## 背景信息

### 目标

区分用户来源，更具体一点指 **Facebook 推广用户来源于哪一个具体的 Facebook Campaign**；

### 说明

如果已经接了 Adjust/AppsFlyer SDK, 则可直接略过。本文旨在手动解析一手的 Referrer 信息，并设置为 Firebase 用户属性。其中：

- 实现方法：
  - 工具：Play Install Referrer API；
  - 体现：Firebase 的用户属性`campaign_id`；
- 局限性：
  - 仅支持安卓系统；
  - 仅支持 Facebook Ads；

## 具体实现

### 方法概述

共三步：

{{< expand "步骤一：获取 referrerUrl" >}}

1. 先接 [Play Install Referrer](https://developer.android.com/google/play/installreferrer/library) 客户端库；
2. 再通过客户端库的方法获取原始的referrerUrl；

{{< /expand >}}

{{< expand "步骤二：解析 referrerUrl（核心）" >}}

1. 先从`referrerUrl`中获取`utm_content`；
2. 再解密`utm_content`。方法见官方的 [Understand Facebook App Ads Referral URLs](https://developers.facebook.com/docs/app-ads/install-referrer/#step-3--decrypt-your-data)，需要用到Facebook Decryption Key；

{{< /expand >}}

{{< expand "步骤三：处理解析结果" >}}

1. 先从解密后的`utm_content`中获取`campaign_group_id`；
2. 再将`campaign_group_id`设置为用户属性`campaign_id`；

{{< /expand >}}

### 步骤一：获取 referrerUrl

先接Play Install Referrer客户端库：
- 官方文档: https://developer.android.com/google/play/installreferrer/library
- 他人做法参考: https://www.geeksforgeeks.org/how-to-use-google-play-install-referrer-api-in-android/

再获取原始的referrerUrl：
- 官方方法: https://developer.android.com/google/play/installreferrer/library#install-referrer
- 参考：
    ```java
    ReferrerDetails response = referrerClient.getInstallReferrer();
    String referrerUrl = response.getInstallReferrer(); // 就是这个东西，且仅需这一个
    long referrerClickTime = response.getReferrerClickTimestampSeconds();
    long appInstallTime = response.getInstallBeginTimestampSeconds();
    boolean instantExperienceLaunched = response.getGooglePlayInstantParam();
    ```

### 步骤二：解析 referrerUrl

#### referrerUrl 格式说明

#### utm_content 格式说明


#### 解析方法


### 步骤三：处理解析结果


## 测试方法

使用本文中的 *referrerUrl格式说明* 中的例子即可；

## 附

### Firebase User Property 配额限制

user-property 见 https://support.google.com/firebase/answer/9237506?hl=en

- 个数：<= 25；
- 命名长度：<= 24个字符；
- 取值长度：<= 36个字符；

### referrerUrl 格式参考

来自Adjust: https://partners.adjust.com/placeholders/#Referrer

```url
utm_source%3Dmy.apps.com%26utm_campaign%3Dmy_campaign%26utm_content%3D%7B%22key1%22%3A0%2C%22key2%22%3A1623237220%7D
```

<img src='/images/posts/referrer-example.png' alt='Referrer example'>

### GA4 Scopes

[[GA4] Scopes of traffic-source dimensions](https://support.google.com/analytics/answer/11080067?hl=en#zippy=%2Cin-this-article)