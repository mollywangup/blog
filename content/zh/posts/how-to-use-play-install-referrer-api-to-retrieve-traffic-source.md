---
title: "使用 Play Install Referrer API 解析Facebook Campaign信息"
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

区分用户来源，更具体一点指 **Facebook推广用户来源于哪一个具体的Facebook Campaign**；

### 说明

如果已经接了 Adjust SDK, 则直接略过。相当于把 Adjust SDK 已实现做的事情，自己再手动做一遍。其中：

- 实现方法：
  - 工具：Play Install Referrer API；
  - 体现：Firebase的用户属性`campaign_id`；
- 局限性：
  - 仅支持安卓系统；
  - 仅支持Facebook Ads；

## 具体实现

### 方法概述

共三步：
1. **获取 referrerUrl**：
    包含以下2个步骤：
   - 先接 [Play Install Referrer](https://developer.android.com/google/play/installreferrer/library) 客户端库；
   - 再通过客户端库的方法获取原始的referrerUrl；
1. 解析referrerUrl（核心）：
  - 包含以下2个步骤：
    - 先从`referrerUrl`中获取`utm_content`；
    - 再解密`utm_content`。方法见官方的 [Understand Facebook App Ads Referral URLs](https://developers.facebook.com/docs/app-ads/install-referrer/#step-3--decrypt-your-data)，需要用到Facebook Decryption Key；
2. 处理解析结果：
  - 包含以下2个步骤：
    - 先从解密后的`utm_content`中获取`campaign_group_id`：
    - 再将`campaign_group_id`设置为用户属性`campaign_id`；
  
### 步骤一：获取 referrerUrl

1. 先接Play Install Referrer客户端库：
  - 官方文档: https://developer.android.com/google/play/installreferrer/library
  - 他人做法参考: https://www.geeksforgeeks.org/how-to-use-google-play-install-referrer-api-in-android/
2. 再获取原始的referrerUrl：
  - 官方方法：https://developer.android.com/google/play/installreferrer/library#install-referrer
    参考：
```java
ReferrerDetails response = referrerClient.getInstallReferrer();
String referrerUrl = response.getInstallReferrer(); // 就是这个东西，且仅需这一个
long referrerClickTime = response.getReferrerClickTimestampSeconds();
long appInstallTime = response.getInstallBeginTimestampSeconds();
boolean instantExperienceLaunched = response.getGooglePlayInstantParam();
```

### 步骤二：解析referrerUrl

#### referrerUrl格式说明

#### utm_content格式说明


#### 解析方法


### 步骤三：处理解析结果


## 测试方法

使用本文中的 *referrerUrl格式说明* 中的例子即可；

## 附

### Firebase对User properties的配额限制

见 https://support.google.com/firebase/answer/9237506?hl=en
- 个数：<= 25；
- 命名长度：<= 24个字符；
- 取值长度：<= 36个字符；

### referrerUrl格式参考

来自Adjust: https://partners.adjust.com/placeholders/#Referrer

```url
utm_source%3Dmy.apps.com%26utm_campaign%3Dmy_campaign%26utm_content%3D%7B%22key1%22%3A0%2C%22key2%22%3A1623237220%7D
```

<img src='/images/posts/referrer-example.png' alt='Referrer example'>

### GA4 Scopes

[[GA4] Scopes of traffic-source dimensions](https://support.google.com/analytics/answer/11080067?hl=en#zippy=%2Cin-this-article)