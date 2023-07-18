---
title: "数据抓取方法论"
date: 2021-01-18T06:08:30Z
draft: false
description: Facebook Ads/Google Ads/Apple Search Ads/Adjust/AdMob/GA4
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Facebook Ads
- Google Ads
- Apple Search Ads
- Adjust
- AdMob
- GA4
categories:
- API
- MMP
---

无论哪种广告平台，数据抓取整体可概括为以下三步：

1. 申请接口；（访问令牌/access_token/密钥）
2. 使用官方提供的 API 文档，通过接口请求；
3. 统一指标字典，并写入数据库；

## Facebook Ads

总结概括起来：一个 BM 对应一个访问令牌，访问令牌由该 BM 下管理员权限的 system_user 生成，只要授权给该 system_user 的广告账户，其数据都可以通过该访问令牌请求。

### 接口申请方法

1. 在 Facebook 开发者后台，创建一个 `Business` 类型的应用，见 [Create an app](https://developers.facebook.com/apps/creation/) ；
2. 在 BM 创建一个管理员权限的 `system_user`，见 [Create, Retrieve and Update a System User](https://developers.facebook.com/docs/marketing-api/system-users/create-retrieve-update)；
3. 将上述应用绑定至 BM，注意该 BM 需要作为该应用的所有者；
4. 在 BM 中，使用 system_user 生成访问令牌即 `access_token`，其中：
   - 点击`生成`按钮时，必须选择上述创建的那个应用； 
   - `勾选数据权限`时，必须至少包含以下两个权限：
     - ads_read
     - ads_management

{{< expand "Facebook Ads 接口举例" >}}

注意：xxx 都是非必填项；

```yaml
data_source: xxx # 为了标记数据源或者 Manager account  
business_id: xxx     
business_name: xxx  
app_id: your_app_id # 必填项；Facebook app ID      
app_name: xxx     
app_secret: your_app_secret # 必填项；Facebook app secret 
system_user_id:  xxx    
system_user_name: xxx 
system_user_access_token: your_access_token # 必填项；申请的最终的access_token
```

{{< /expand >}}

{{< notice warning >}}

1. 只有`授权给 system_user` 的广告账户，access_token 才能访问到对应的广告账户的数据；
2. 每个 BM 对应一个 access_token，有几个 BM 就需要申请几个 access_token；

{{< /notice >}}

### 请求方法

1. 请求每个 BM 下的`所有广告账户`；
2. 对于每个 BM 下的每个广告账户，请求 `campaign` 层级的原始数据；
3. 所有 BM 的原始数据写入同一张数据库表如 `sources_adplatform`;
4. 注意事项：
   - 由于广告平台归因窗口的存在，且一般是28天或30天，因此`每天需要跑last 30天`的数据；
   - 存在数据覆写，需注意`组合键的确定`以为了数据的不重不漏原则；

### 官方文档

- [Facebook] [Marketing API](https://developers.facebook.com/docs/marketing-apis)
- [GitHub] [Facebook Business SDK for Python](https://github.com/facebook/facebook-python-business-sdk)

## Google Ads

线下PDF版：<a href="https://mollywangup.com/pdf/" target="_blank">Google Ads API快速上手指南 [2019]</a>


## Apple Search Ads

### 接口申请方法

不是访问令牌的形式，是物理的文件形式的密钥，见官方保姆级教程 [Implementing OAuth for the Apple Search Ads API](https://developer.apple.com/documentation/apple_search_ads/implementing_oauth_for_the_apple_search_ads_api)

### 官方文档

- [Apple] [Apple Search Ads](https://developer.apple.com/documentation/apple_search_ads)

## Adjust

- 官方文档: https://help.adjust.com/zh/article/kpi-service



## AdMob

- 官方文档: https://developers.google.com/admob/api/v1/report-metrics-dimensions

## GA4

- 官方文档: https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema

## 附：指标字典参考

| 编号  | 字段 | 定义 | 数据类型 | 数据来源 | API字段 |
| --- | --- | --- | --- | --- | --- | 
| 1 | app_name | App标识 | varchar | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 |  |
| 2 | os_name | 设备类型，如android/ios | varchar  | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 |  |
| 3 | store_type | 应用商店，如google/itunes/huawei/apkpure等 | varchar(10) | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 |  |
| 4 | media_source | 流量来源，如facebook_ads, google_ads, organic等 | varchar | 通过请求时的access_token确定 | / |
| 5 | account_id | 广告账户ID | varchar | API | account_id |
| 6 | account_name | 广告账户名称 | varchar | API | account_name |
| 7 | campaign_id | 广告推广计划ID | varchar | API | 'level': 'campaign', campaign_id |
| 8 | campaign_name | 广告推广计划名称 | varchar | API | 'level': 'campaign', campaign_name |
| 9 | date | 日期 | date | API | 由date_start + date_stop + time_increment确定，理论上我们应该是分天请求 |
| 10 | country | 国家，ISO 3166标准 | varchar | API | 'breakdowns': ['country'] |
| 11 | impressions | 广告展示量 | int | API | impressions |
| 12 | clicks | 广告点击量 | int | API | clicks |
| 13 | install_adplatform | 广告安装量 | int | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，       |
| 14 | cost | 广告花费 | float | API | spend |
| 15 | purchase_value | 广告带来的收入 | float | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，       |
| 16 | purchase | 付费次数 | int | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，       |
| 17 | purchase_unique | 付费人数 | int | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，       |
| 18 | optimizer | 优化师编号 | varchar | 通过正则表达式确定，具体方法见find.py | / |
| 19 | data_source | 所属BM/MCC | varchar | 通过请求时的access_token确定 | / |
| 20 | currency | 货币单位，目前全部是USD | varchar | API | account_currency |
| 21 | is_organic | 布尔值 | varchar | 通过maps.yaml中的映射关系确定 |  |
| 22 | geographic | 国家的全称 | varchar | 临时的字段，为了配合google_ads的本地.csv性质的数据源 | / |
| 23 | attribution_setting | 归因设置，如：1d_click | varchar | API | attribution_setting  |