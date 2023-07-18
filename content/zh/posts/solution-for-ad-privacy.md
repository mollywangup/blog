---
title: "广告风控指南：隐私政策"
date: 2021-09-29T07:14:58Z
draft: false
description: COPPA/CCPA/GDPR/LGPD
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Monetization
- AdMob
- GP
categories:
- Policy
- SDK
---

## 背景信息

- 移动应用/游戏出海业务（不考虑Web）
- 广告变现角度（开发者角度）
- AdMob 作为聚合平台
- GP 包

## 结论

1. 隐私政策的核心思想：
   
   可以概括为：约束开发者**利用隐私数据进行<u>相关</u>盈利**的行为，更具体一点指**收集设备信息标识用户，用以对不同用户进行差异化的广告展示，即展示个性化广告**。

   因此，遵守隐私政策的核心是：
      - 政策允许才能收集设备标识信息；
        - 注意：用户仍有权在设备层级进行控制，使得开发者收集到的是一串零；
      - 政策不允许需要显示声明未收集设备标识信息；

   {{< alert theme="info" >}}
   💡 保护隐私和向用户展示**非个性化广告**是不冲突的，因为此时用户是匿名的；
   {{< /alert >}}

2. 如何确定该遵守哪些隐私政策：
   - 如果受众包含儿童，则必须遵守《儿童在线隐私保护法》(COPPA)；
   - 如果受众包含美国加州，则必须遵守《加利福尼亚消费者隐私法》(CCPA)；
   - 如果受众包含欧盟、英国，则必须遵守《欧盟通用数据保护条例》(GDPR)；
   - 如果受众包含巴西，则必须遵守《巴西通用数据保护法》(LGPD)；

   {{< notice info >}}
⚠️️ 是坑也是技巧：
**受众**和**年龄评级**是两个东西，是允许**年龄评级是全年龄段但受众是排除了儿童了的**。
   {{< /notice >}}

## 如何风控
  
总结起来：设置年龄tag、移除广告ID权限；

### Google长远规划

<a href="https://privacysandbox.com/intl/en_us/" target="_blank">Privacy Sandbox技术</a>

### 一劳永逸的方法

<a href="https://developers.google.com/android/reference/com/google/android/gms/ads/identifier/AdvertisingIdClient.Info#public-string-getid" target="_blank">Remove ADID权限</a>

亦可参考同行做法：https://jinyoung.dev/posts/android_ads_policy/

```Java
<uses-permission android:name="com.google.android.gms.permission.AD_ID"
 tools:node="remove"/>
```

### 逐个SDK声明

#### AdMob SDK

说明：COPPA 和 GDPR 的tag二选一即可；

<a href="https://developers.google.com/admob/android/targeting#child-directed_setting" target="_blank">Child-directed setting (COPPA)</a>
<a href="https://developers.google.com/admob/android/targeting#users_under_the_age_of_consent" target="_blank">Users under the age of consent (GDPR)</a>
<a href="https://developers.google.com/admob/android/ccpa" target="_blank">Restricted Data Processing (CCPA)</a>

```Java
// COPPA
.setTagForChildDirectedTreatment(RequestConfiguration.TAG_FOR_CHILD_DIRECTED_TREATMENT_TRUE)

// GDPR
.setTagForUnderAgeOfConsent(RequestConfiguration.TAG_FOR_UNDER_AGE_OF_CONSENT_TRUE)
```

#### Facebook SDK

<a href="https://developers.facebook.com/docs/app-events/getting-started-app-events-android" target="_blank">Disable Collection of Advertiser IDs</a>

```Java
<application>
  ...
  <meta-data android:name="com.facebook.sdk.AdvertiserIDCollectionEnabled"
           android:value="false"/>
  ...
</application>
```

#### Firebase SDK

<a href="https://firebase.google.com/docs/analytics/configure-data-collection?platform=android#disable_advertising_id_collection" target="_blank">Disable Advertising ID collection</a>

```Java
<meta-data android:name="google_analytics_adid_collection_enabled" android:value="false" />
```

#### AppLovin SDK

<a href="https://developers.google.com/admob/android/mediation/applovin#eu_consent_and_gdpr" target="_blank">setIsAgeRestrictedUser (COPPA)</a>
<a href="https://developers.google.com/admob/android/mediation/applovin#eu_consent_and_gdpr" target="_blank">setHasUserConsent (GDPR)</a>
<a href="https://developers.google.com/admob/android/mediation/applovin#ccpa" target="_blank">setDoNotSell (CCPA)</a>

```Java
AppLovinPrivacySettings.setIsAgeRestrictedUser(true, context);
AppLovinPrivacySettings.setHasUserConsent(true, context);
AppLovinPrivacySettings.setDoNotSell(true, context);
```

#### Unity Ads SDK

<a href="https://developers.google.com/admob/android/mediation/unity#eu_consent_and_gdpr" target="_blank">GDPR</a>
<a href="https://developers.google.com/admob/android/mediation/unity#ccpa" target="_blank">CCPA</a>

```Java
MetaData gdprMetaData = new MetaData(this);
gdprMetaData.set("gdpr.consent", true);
gdprMetaData.commit();

MetaData ccpaMetaData = new MetaData(this);
ccpaMetaData.set("privacy.consent", true);
ccpaMetaData.commit();
```

#### Vungle SDK

<a href="https://developers.google.com/admob/android/mediation/vungle#eu_consent_and_gdpr" target="_blank">GDPR</a>
<a href="https://developers.google.com/admob/android/mediation/vungle#ccpa" target="_blank">CCPA</a>

```Java
Vungle.updateConsentStatus(Vungle.Consent.OPTED_IN, "1.0.0");
Vungle.updateCCPAStatus(Vungle.Consent.OPTED_IN);
```

## 附：概念定义

### 常看常新

<a href="https://android-developers.googleblog.com/2022/11/keeping-google-play-safe.html" target="_blank">Keeping Google Play Safe with New Features and Programs</a>

### 隐私

隐私在不同政策（国家）法律/平台下的定义：
- <a href="https://support.google.com/admob/answer/7686480?hl=en" target="_blank">"Personally Identifiable Information" (PII)</a>
- <a href="https://www.google.com/policies/privacy/" target="_blank">Google Play Services</a>
- <a href="https://support.google.com/admob/answer/6128543/" target="_blank">AdMob</a>
- <a href="https://firebase.google.com/policies/analytics/" target="_blank">Google Analytics for Firebase</a>
- <a href="https://firebase.google.com/support/privacy/" target="_blank">Firebase Crashlytics</a>
- <a href="https://www.facebook.com/about/privacy/update/printable/" target="_blank">Facebook</a>
- <a href="https://unity.com/legal/privacy-policy" target="_blank">Unity Ads</a>
- <a href="https://www.applovin.com/privacy/" target="_blank">AppLovin</a>

### 个性化广告

1. 基于用户兴趣，来对用户进行个性化广告展示；
2. 使用device identifiers、cookies，用于个性化广告；
3. AdMob定义的：<a href="https://support.google.com/admob/answer/7676680?hl=en" target="_blank">Personalized ads</a>
4. IAB定义的：

### 非个性化广告 (NPA)

1. 基于当前的上下文信息，及粗略的地理位置估计，来对用户进行非个性化广告展示；
2. 会使用device identifiers、cookies，但是不能用于个性化广告，仅可用于频次控制、反作弊等；
3. AdMob定义的：<a href="https://support.google.com/admob/answer/7676680?hl=en" target="_blank">Non-personalized ads (NPA)</a>

### 谷歌儿童政策

> https://support.google.com/googleplay/android-developer/answer/11043825?hl=en 
> Apps that target both children and older audiences must not transmit AAID, SIM serial, build serial, BSSID, MAC, SSID, IMEI and/or IMSI from children or users of unknown age.

## 附：四大隐私政策

### COPPA

The Children’s Online Privacy Protection Act (COPPA)

针对受众群体中包含儿童/未成年用户的App（也称为Family Policy）

1. [Tag an ad request from an app for child-directed treatment](https://support.google.com/admob/answer/6219315?hl=en)
2. [Comply with Google Play’s Families Policy using AdMob](https://support.google.com/admob/answer/6223431?hl=en)
3. [Set a maximum ad content rating](https://support.google.com/admob/answer/10478094?hl=en)
4. [Complying with COPPA: Frequently Asked Questions](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions)

### GDPR
EU user consent policy (GDPR)

针对欧盟、英国、瑞士的用户

1. [Tools to help publishers comply with the GDPR](https://support.google.com/admob/answer/7666366?hl=en)
2. [EU user consent policy](https://www.google.com/about/company/user-consent-policy/)
3. [Helping publishers and advertisers with consent](https://www.cookiechoices.org/intl/en/)
4. [IAB CMP list](https://iabeurope.eu/cmp-list/)
5. [IAB Europe Transparency & Consent Framework Policies](https://iabeurope.eu/iab-europe-transparency-consent-framework-policies/?rd=1#Appendix_A_Purposes_and_Features_Definitions)

### CCPA
California Consumer Privacy Act (CCPA)

针对美国加利福尼亚洲的用户

[Helping publishers comply with the California Consumer Privacy Act (CCPA)](https://support.google.com/admob/answer/9561022?hl=en)

{{< alert theme="info" >}}
The California Privacy Rights Act (CPRA) is a data privacy law that amends and expands upon the CCPA. The law takes effect on January 1, 2023.
{{< /alert >}}

### LGPD
Lei Geral de Proteção de Dados (LGPD)

针对巴西的用户

[Helping users comply with the Lei Geral de Proteção de Dados (LGPD)](https://support.google.com/admob/answer/9930897?hl=en)