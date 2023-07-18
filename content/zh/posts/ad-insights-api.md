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

无论哪种广告平台，数据抓取整体可概括为以下两步：

1. 申请接口；（访问令牌/access_token）
2. 使用官方提供的 API 文档，通过接口访问即可。

## Facebook Ads

### 参考文档

- [Facebook] [Marketing API](https://developers.facebook.com/docs/marketing-apis)
- [Create, Retrieve and Update a System User](https://developers.facebook.com/docs/marketing-api/system-users/create-retrieve-update)
- [GitHub] [Facebook Business SDK for Python](https://github.com/facebook/facebook-python-business-sdk)

- 请求 BM 下的广告账户列表：[Ad Account](https://developers.facebook.com/docs/marketing-api/reference/ad-account)
- 请求一个广告账户的原始数据：[Campaign Insights](https://developers.facebook.com/docs/marketing-api/reference/ad-campaign-group/insights/)
- 创建 App：[Create an app](https://developers.facebook.com/apps/creation/)

### 接口申请方法

1. 创建应用：在 Facebook 开发者后台，创建一个 `Business` 类型的 app，见 [Create an app](https://developers.facebook.com/apps/creation/) ；
2. 创建 System User：在 BM 创建一个管理员权限的 `system_user`，见 [Create, Retrieve and Update a System User](https://developers.facebook.com/docs/marketing-api/system-users/create-retrieve-update)；
3. 应用绑定 BM：将上述应用绑定至 BM，注意该 BM 需要作为 owner；
4. 使用 System User 生成访问令牌：
   1. 在 BM 中生成，点击`生成`按钮时，需要选择上述创建的那个应用； 
   2. 为上述 access_token `勾选数据权限`，即可生成最终的`access_token`；
      - ads_read
      - ads_management
  
{{< alert theme="warning" >}}
1. 只有`授权给 system_user` 的广告账户，access_token 才能访问到对应的广告账户的数据。因此当有新广告账户绑定至该BM时，需要手动授权该广告账户给 system_user；
2. 每个BM对应一个 access_token，有几个BM就需要几个 access_token;
{{< /alert >}}

{{< expand "接口例子" >}}

```yaml
data_source: 'bm_xxx' # 非必填项；为了标记数据源或者manager_account  
business_id: 3064441080294895 # 非必填项     
business_name: Tentacles overseas projects # 非必填项；bm名称      
app_id: # 必填项；claim过来的那个app对应的facebook开发者后台的app_id      
app_name: Hala # 非必填项；claim过来的那个app的名称      
app_secret: # 必填项；claim过来的那个app对应的facebook开发者后台的app_secret 
system_user_id:  # 非必填项；可通过'/me?'请求获得      
system_user_name: system_user_tentacles # 非必填项；system_user名称
system_user_access_token: # 必填项；申请的最终的access_token
```

{{< /expand >}}

### 请求方法

以单个 BM 为例：

1. 请求该 BM 下的`所有广告账户`；
2. 对于每个广告账户，请求 `campaign` 层级的原始数据；
3. 所有 BM 的原始数据写入同一张数据库表 `sources_adplatform`;
4. 注意事项：
   - 由于广告平台归因窗口的存在，且一般是28天或30天，因此`每天需要跑last 30天`的数据；
   - 存在数据覆写，需注意`组合键的确定`以为了数据的不重不漏原则；

### 指标字典

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

## Google Ads


## Apple Search Ads

- 官方文档: https://developer.apple.com/documentation/apple_search_ads


## Adjust

- 官方文档: https://help.adjust.com/zh/article/kpi-service



## AdMob

- 官方文档: https://developers.google.com/admob/api/v1/report-metrics-dimensions

## GA4

- 官方文档: https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema