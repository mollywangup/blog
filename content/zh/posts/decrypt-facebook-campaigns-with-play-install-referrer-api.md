---
title: "使用 Play Install Referrer API 解密 Facebook Campaign"
date: 2022-10-25T06:17:06Z
draft: false
description: 手动解密 Facebook Campaign.
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

本文旨在手动解析一手的 Referrer 信息，并设置为 Firebase 用户属性。如果已经接了 MMP，可直接略过。（🤝 感兴趣的话也可以了解下）

- 实现方法：
  - 工具：Play Install Referrer API；
  - 体现：Firebase 的用户属性`campaign_id`；
- 局限性：
  - 仅支持安卓系统；
  - 仅支持 Facebook Ads；

## 方法概述

共三步：

{{< expand "Step1. 获取 referrerUrl" >}}

1. 先接 [Play Install Referrer](https://developer.android.com/google/play/installreferrer/library) 客户端库；
2. 再通过客户端库的方法获取原始的 referrerUrl；

{{< /expand >}}

{{< expand "Step2. 解析 referrerUrl（核心）" >}}

1. 先从`referrerUrl`中获取`utm_content`；
2. 再解密`utm_content`。方法见官方的 [Understand Facebook App Ads Referral URLs](https://developers.facebook.com/docs/app-ads/install-referrer/#step-3--decrypt-your-data)，需要用到 Facebook Decryption Key；

{{< /expand >}}

{{< expand "Step3. 处理解析结果" >}}

1. 先从解密后的`utm_content`中获取`campaign_group_id`；
2. 再将`campaign_group_id`设置为用户属性`campaign_id`；

{{< /expand >}}

## 具体实现

### Step1. 获取 referrerUrl

1. 先接 Play Install Referrer 客户端库：
   - 官方文档：[Play Install Referrer Library](https://developer.android.com/google/play/installreferrer/library)
   - 他人做法参考：[How to Use Google Play Install Referrer API in Android?](https://www.geeksforgeeks.org/how-to-use-google-play-install-referrer-api-in-android/)

2. 再获取原始的 referrerUrl：
   - 官方方法：[Getting the install referrer](https://developer.android.com/google/play/installreferrer/library#install-referrer)
      ```java
      ReferrerDetails response = referrerClient.getInstallReferrer();
      String referrerUrl = response.getInstallReferrer(); // 就是这个东西，且仅需这一个
      long referrerClickTime = response.getReferrerClickTimestampSeconds();
      long appInstallTime = response.getInstallBeginTimestampSeconds();
      boolean instantExperienceLaunched = response.getGooglePlayInstantParam();
      ```

### Step2. 解析 referrerUrl

#### referrerUrl 格式说明

格式（以下使用的是同一个例子）：

{{< tabs 原始格式 decode后的格式 >}}
{{< tab >}}

```plaintext
utm_source%3Dutm_source_xxx%26utm_campaign%3Dutm_campaign_xxx%26utm_medium%3Dutm_medium_xxx%26utm_content%3D%7B%22source%22%3A%20%7B%22data%22%3A%20%223154158d7cfc829685fab52df9b47ba67b89947743514445d11ad23788bb6467fcf3775aa3c7e87e47db0bc38a6ddd4a0cd49b0100bc036ec10b1082714416132495ac4cc09953805ab282865f2d2620a0914496188f15c649424752fa8a6edd78b6c85f2dc1c1de175c29a3efaf47b14afda86826fe1adbfe170ed1759186cbee98944c539641f55e0f42937ae4c1a6f84d4b9335087306d9af8c3d7379ad56bcfe1e021b93da20595f3ba14500c3056508fc154dac3175db2f5f45756afc914f9d910cd867e23b1d430158690dbc53b9aa098bbb056f8152502dcdb64d6ec96eccd908895f34262ce5c5068fb64cdb4595d6eb44553acc1bd56b40789192de7cf78f0c951a0aab2ede8a9eae23b60f95e26ca14c9c84076ab73927c88bf5d496c5cf4fe642d5e550add78fa84796383cb1c71f062a39f5297fb8e4a4717d13f2d7a3c738d37303b5080bdcb08a%22%2C%20%22nonce%22%3A%20%22ee8501a143b5d3950cf820b1ee1c4f9f%22%7D%7D
```

{{< /tab >}}
{{< tab >}}

```plaintext
utm_source=utm_source_xxx&utm_campaign=utm_campaign_xxx&utm_medium=utm_medium_xxx&utm_content={"source": {"data": "3154158d7cfc829685fab52df9b47ba67b89947743514445d11ad23788bb6467fcf3775aa3c7e87e47db0bc38a6ddd4a0cd49b0100bc036ec10b1082714416132495ac4cc09953805ab282865f2d2620a0914496188f15c649424752fa8a6edd78b6c85f2dc1c1de175c29a3efaf47b14afda86826fe1adbfe170ed1759186cbee98944c539641f55e0f42937ae4c1a6f84d4b9335087306d9af8c3d7379ad56bcfe1e021b93da20595f3ba14500c3056508fc154dac3175db2f5f45756afc914f9d910cd867e23b1d430158690dbc53b9aa098bbb056f8152502dcdb64d6ec96eccd908895f34262ce5c5068fb64cdb4595d6eb44553acc1bd56b40789192de7cf78f0c951a0aab2ede8a9eae23b60f95e26ca14c9c84076ab73927c88bf5d496c5cf4fe642d5e550add78fa84796383cb1c71f062a39f5297fb8e4a4717d13f2d7a3c738d37303b5080bdcb08a", "nonce": "ee8501a143b5d3950cf820b1ee1c4f9f"}}
```

{{< /tab >}}
{{< /tabs >}}

结构（重点是处理`utm_content`）：

| / | 说明 | 是否Firebase已自动统计 | 例子 |
| ---------- | --------- | ----------------- | ---------- |
| `utm_source` | 指流量来源；<br>字符串格式；| 是；<br>体现在BigQuery的**traffic_source.source** | (direct)<br>apps.facebook.com<br>google-play |
| `utm_medium` | 同上 | 是；<br>体现在BigQuery的**traffic_source.medium** | (none)<br>organic |
| `utm_campaign` | 同上 | / | / |
| `utm_content` | 一般主要用于解析来自Facebook Ads的广告；<br>json字符串格式；| **否**；<br>因此重点是这里 | 详见下方；<br>Facebook Ads需进一步解密；|

#### utm_content 格式说明

参考来自 Facebook 官方文档：[Understand Facebook App Ads Referral URLs](https://developers.facebook.com/docs/app-ads/install-referrer/)

{{< tabs 原始格式 decode后的格式 解密并decode最核心的data后的格式 >}}
{{< tab >}}

```plaintext
%7B%22source%22%3A%20%7B%22data%22%3A%20%223154158d7cfc829685fab52df9b47ba67b89947743514445d11ad23788bb6467fcf3775aa3c7e87e47db0bc38a6ddd4a0cd49b0100bc036ec10b1082714416132495ac4cc09953805ab282865f2d2620a0914496188f15c649424752fa8a6edd78b6c85f2dc1c1de175c29a3efaf47b14afda86826fe1adbfe170ed1759186cbee98944c539641f55e0f42937ae4c1a6f84d4b9335087306d9af8c3d7379ad56bcfe1e021b93da20595f3ba14500c3056508fc154dac3175db2f5f45756afc914f9d910cd867e23b1d430158690dbc53b9aa098bbb056f8152502dcdb64d6ec96eccd908895f34262ce5c5068fb64cdb4595d6eb44553acc1bd56b40789192de7cf78f0c951a0aab2ede8a9eae23b60f95e26ca14c9c84076ab73927c88bf5d496c5cf4fe642d5e550add78fa84796383cb1c71f062a39f5297fb8e4a4717d13f2d7a3c738d37303b5080bdcb08a%22%2C%20%22nonce%22%3A%20%22ee8501a143b5d3950cf820b1ee1c4f9f%22%7D%7D
```

{{< /tab >}}
{{< tab >}}

```json
{"source": {"data": "3154158d7cfc829685fab52df9b47ba67b89947743514445d11ad23788bb6467fcf3775aa3c7e87e47db0bc38a6ddd4a0cd49b0100bc036ec10b1082714416132495ac4cc09953805ab282865f2d2620a0914496188f15c649424752fa8a6edd78b6c85f2dc1c1de175c29a3efaf47b14afda86826fe1adbfe170ed1759186cbee98944c539641f55e0f42937ae4c1a6f84d4b9335087306d9af8c3d7379ad56bcfe1e021b93da20595f3ba14500c3056508fc154dac3175db2f5f45756afc914f9d910cd867e23b1d430158690dbc53b9aa098bbb056f8152502dcdb64d6ec96eccd908895f34262ce5c5068fb64cdb4595d6eb44553acc1bd56b40789192de7cf78f0c951a0aab2ede8a9eae23b60f95e26ca14c9c84076ab73927c88bf5d496c5cf4fe642d5e550add78fa84796383cb1c71f062a39f5297fb8e4a4717d13f2d7a3c738d37303b5080bdcb08a", "nonce": "ee8501a143b5d3950cf820b1ee1c4f9f"}}
```

{{< /tab >}}
{{< tab >}}

```json
{
    "ad_id":"{ad-id}",
    "adgroup_id":"{ad-group-id}",
    "adgroup_name":"{ad-group-name}",
    "campaign_id":"{campaign-id}",
    "campaign_name":"{campaign-name}",
    "campaign_group_id":"23851271281990526", // 目标就是获取这个
    "campaign_group_name":"{campaign-group-name}",
    "account_id":"act_484103070416836",
    "ad_objective_name":"APP_INSTALLS"
}
```

{{< /tab >}}
{{< /tabs >}}

#### 解析方法

1. 先从 referrerUrl 中获取 utm_content；
    注意：原始的 referrerUrl 和获取到的 utm_content，在进行下一步操作之前，都需要先decode；
2. 再解密 utm_content（最核心的一步）：
    官方方法：[Example Decryption with PHP](https://developers.facebook.com/docs/app-ads/install-referrer/#step-3--decrypt-your-data)
    具体方法如下：
    1. 加密方式：**AES256-GCM**；
    2. 解密对象/密文：`utm_content` -> `source` -> `data`；
    3. 解密共需以下3个信息：
         - **Facebook Decryption Key**：即密钥，来自Facebook开发者后台；
         - **data**：即解密对象/密文。
            重要说明：data中包含了`tag`，因此处理时需要先忽略/截断。其中，关于tag：
             - 对应上述例子：`7d13f2d7a3c738d37303b5080bdcb08a`；
             - 位置：后缀；
             - 长度：固定长度的32个16进制字符，即16个字节；
                - https://pycryptodome.readthedocs.io/en/latest/src/cipher/aes.html
                <img src='/images/posts/tag-gcm.png' alt='MODE_GCM length'>
                - https://developers.facebook.com/docs/app-ads/install-referrer/
                  <img src='/images/posts/tag-gcm-16bytes.png' alt='Tag length 16 bytes'>
         - **nonce**：随机数，无实际意义，解密需要；
    4. 最后，使用以上信息，解密；
      其中，解密后的明文见 ***utm_content 格式说明*** 中的 ***解密并decode最核心的`data`后的格式***；

### Step3. 处理解析结果

1. 先获取 campaign_group_id：解密后的明文 -> `campaign_group_id`；
2. 设置用户属性 `campaign_id`：
   - 触发场景：新用户首次启动时触发，且仅触发一次（越早越好）；
   - 方法：[Set user properties](https://firebase.google.com/docs/analytics/user-properties?platform=android)
      ```java
      // 正常获取时
      mFirebaseAnalytics.setUserProperty("campaign_id", campaign_group_id);

      // 异常时（无法获取或解析）
      mFirebaseAnalytics.setUserProperty("campaign_id", "unknown");
      ```

## 测试方法

使用本文中的 *referrerUrl 格式说明* 中的例子即可；

## 附

### Firebase User Property 配额限制

见 https://support.google.com/firebase/answer/9237506?hl=en

- 个数：<= 25；
- 命名长度：<= 24个字符；
- 取值长度：<= 36个字符；

### referrerUrl 格式参考

见 [Adjust Placeholders for Partners](https://partners.adjust.com/placeholders/#Referrer)

```plaintext
utm_source%3Dmy.apps.com%26utm_campaign%3Dmy_campaign%26utm_content%3D%7B%22key1%22%3A0%2C%22key2%22%3A1623237220%7D
```

<img src='/images/posts/referrer-example.png' alt='Referrer example'>

### GA4 Scopes

[[GA4] Scopes of traffic-source dimensions](https://support.google.com/analytics/answer/11080067?hl=en#zippy=%2Cin-this-article)