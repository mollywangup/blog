---
title: "数据抓取方法论"
date: 2021-01-18T06:08:30Z
draft: false
description: 
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
---

## Facebook Ads
 
- 官方文档: https://developers.facebook.com/docs/marketing-apis
- 目标: 通过Facebook Ads API接口，实现请求所有Facebook广告账户的数据；

### 申请API接口

#### 方法概述

1. 在Facebook开发者后台，创建一个`Business`类型的app，且无需填写额外的app信息「就是这么好用」;
2. 在BM创建一个admin `system_user`；
3. 在BM将上述创建的app `claim过来`，即该BM是app owner；
4. 在BM用上述的system_user去生成一个access_token，点击`生成`按钮后，需先选择一个app，这时选择上述claim过来的那个app；
5. 为上述access_token`勾选数据权限`，即可生成最终的`access_token`；
   - ads_read
   - ads_management
6. 只有`授权给system_user`的广告账户，access_token才能访问到对应的广告账户的数据。因此当有新广告账户绑定至该BM时，需要手动授权该广告账户给system_user；
7. 每个BM对应一个access_token，有几个BM就需要几个access_token;

#### 最终的接口demo

```yaml
businesses_manager_tentacles:
	-
		data_source: 'bm_tentacles' # 非必填项；为了标记数据源或者manager_account
		business_id: 3064441080294895 # 非必填项；bm id
		business_name: Tentacles overseas projects # 非必填项；bm名称
		app_id: # 必填项；claim过来的那个app对应的facebook开发者后台的app_id
		app_name: Hala # 非必填项；claim过来的那个app的名称
		app_secret: # 必填项；claim过来的那个app对应的facebook开发者后台的app_secret
		system_user_id:  # 非必填项；可通过'/me?'请求获得
		system_user_name: system_user_tentacles # 非必填项；system_user名称
		system_user_access_token: # 必填项；申请的最终的access_token
```

### 请求方法

#### 方法概述

1. 请求其中一个BM下的所有`广告账户的列表`；
2. 对广告账户列表，异步请求（`is_async=True`）每个广告账户下的`campaign`（广告推广计划）层级的原始数据；
3. 其他BM重复上述3步操作；
4. 所有BM的原始数据写入同一张数据库表`sources_adplatform`;
5. 注意事项：
   - 由于广告平台归因窗口的存在，且一般是28天或30天，因此`每天需要跑last 30天`的数据；
   - 存在数据覆写，需注意`组合键的确定`以为了数据的不重不漏原则；
6. 申请不到接口时的替代方案：
   - 为什么需要考虑这个方案的存在性？因为Google Ads现在就没有申请到接口，Facebook Ads虽然现在两个BM都申请到了但存在BM被封进而接口失效的风险；
   - 替代的方案是什么？人工导入格式化的.csv文件上传至指定接口。

### 指标字典

| 编号  | 字段 | 定义 | 数据类型 | 数据来源 | API字段 |
| --- | --- | --- | --- | --- | --- | 
| 1 | app_name | App标识，与数据库的订单表取齐，全部小写 | varchar(15) | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 | 具体方法见fb_campaign_promoted_object.py |
| 2 | os_name | 设备类型，如android/ios | varchar(7)  | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 | 具体方法见fb_campaign_promoted_object.py |
| 3 | store_type | 应用商店，如google/itunes/huawei/apkpure等，不与目前的init_market_code取齐 | varchar(10) | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 | 具体方法见fb_campaign_promoted_object.py |
| 4 | media_source | 流量来源，如facebook_ads, google_ads, organic等 | varchar(25) | 通过请求时的access_token确定 | / |
| 5 | account_id | 广告账户ID. facebook_ads如act_12345678901, google_ads如act_123-456-7890, apple_search_ads如orgid_2006298 | varchar(25) | API | account_id |
| 6 | account_name | 广告账户名称 | varchar(50) | API | account_name |
| 7 | campaign_id | 广告推广计划ID | varchar(20) | API | 'level': 'campaign', campaign_id |
| 8 | campaign_name | 广告推广计划名称 | varchar(255) | API | 'level': 'campaign', campaign_name |
| 9 | date | 日期 | date | API | 由date_start + date_stop + time_increment确定，理论上我们应该是分天请求 |
| 10 | country | 国家，ISO 3166标准 | varchar(7) | API | 'breakdowns': ['country'] |
| 11 | impressions | 广告展示量 | int | API | impressions |
| 12 | clicks | 广告点击量 | int | API | clicks |
| 13 | install_adplatform | 广告安装量 | int | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，具体方法见fb_campaign_insights.py |
| 14 | cost | 广告花费 | float | API | spend |
| 15 | purchase_value | 广告带来的收入 | float | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，具体方法见fb_campaign_insights.py |
| 16 | purchase | 广告带来的收入对应的购买的笔数（付费次数） | int | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，具体方法见fb_campaign_insights.py |
| 17 | purchase_unique | 广告带来的收入对应的购买的人数（付费人数） | int | API | actions, 这个字段返回一个json格式的，需要使用'filtering'，具体方法见fb_campaign_insights.py |
| 18 | optimizer | 优化师，如：wy_汪悦 | varchar(20) | 通过正则表达式确定，具体方法见find.py | / |
| 19 | data_source | 所属BM/MCC | varchar(25) | 通过请求时的access_token确定 | / |
| 20 | currency | 货币单位，目前全部是USD | varchar(3) | API | account_currency |
| 21 | is_organic | 只有两种取值：organic和non-organic | varchar(12) | 通过maps.yaml中的映射关系确定 |  |
| 22 | geographic | 国家的全称 | varchar(50) | 临时的字段，为了配合google_ads的本地.csv性质的数据源 | / |
| 23 | attribution_setting | 归因设置，如：1d_click | varchar(25) | API | attribution_setting  |

### 参考文档

1. 请求BM下的广告账户列表: https://developers.facebook.com/docs/marketing-api/reference/ad-account
2. 请求一个广告账户的原始数据: https://developers.facebook.com/docs/marketing-api/reference/ad-campaign-group/insights/
3. GitHub上的不同语言的代码库（翻到最下面）: https://developers.facebook.com/docs/business-sdk/getting-started

## Google Ads


## Apple Search Ads

- 官方文档: https://developer.apple.com/documentation/apple_search_ads


## Adjust

- 官方文档: https://help.adjust.com/zh/article/kpi-service



## AdMob

- 官方文档: https://developers.google.com/admob/api/v1/report-metrics-dimensions

## GA4

- 官方文档: https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema