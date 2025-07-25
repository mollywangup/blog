---
title: "广告数据抓取方法论"
date: 2021-01-18T06:08:30Z
draft: false
description: Facebook Ads/Google Ads/Apple Search Ads.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Facebook Ads
- Google Ads
- Apple Search Ads
categories:
- API
---

无论哪个广告平台，数据抓取整体可概括为以下两步：

1. 申请接口；（访问令牌/密钥文件）
2. 通过接口请求数据；

## Facebook Ads

每个 BM 对应一个访问令牌，访问令牌由该 BM 下管理员权限的 system_user 生成，只要授权给该 system_user 的广告账户，其数据都可以通过该访问令牌请求。

### 接口申请方法

1. 在 Facebook 开发者后台，创建一个 `Business` 类型的应用：
   - 方法见 [Create an app](https://developers.facebook.com/apps/creation/) ；
2. 在 BM 创建一个管理员权限的 `system_user`：
   - 方法见 [Create, Retrieve and Update a System User](https://developers.facebook.com/docs/marketing-api/system-users/create-retrieve-update)；
3. 将上述应用绑定至 BM，注意该 BM 需要作为该应用的所有者；
4. 在 BM 中，使用 system_user 生成 `access_token` 即访问令牌，其中：
   - 点击`生成`按钮时，必须选择上述创建的那个应用； 
   - `勾选数据权限`时，必须至少包含以下两个权限：
     - ads_read
     - ads_management

{{< expand "Facebook Ads 接口举例" >}}

```yaml
data_source:  
business_id:      
business_name:   
app_id: your_app_id # 必配置项；Facebook app ID      
app_name:      
app_secret: your_app_secret # 必配置项；Facebook app secret 
system_user_id:      
system_user_name:  
system_user_access_token: your_access_token # 必配置项；访问令牌
```

{{< /expand >}}

{{< notice warning >}}

1. 只有已授权给 system_user 的广告账户，访问令牌才能访问到对应广告账户的数据；
2. 每个 BM 对应一个访问令牌，有几个 BM 就需要申请几个访问令牌；

{{< /notice >}}

### 官方文档

- [Facebook] [Marketing API](https://developers.facebook.com/docs/marketing-apis)
- [GitHub] [Facebook Business SDK for Python](https://github.com/facebook/facebook-python-business-sdk)

## Google Ads

### 接口申请方法

见官方保姆级教程 <a href="https://mollywangup.com/pdf/Google%20Ads%20API%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B%E6%8C%87%E5%8D%97%20%5B2019%5D.pdf" target="_blank">Google Ads API快速上手指南 [2019]</a>

{{< expand "Google Ads 接口举例" >}}

```yaml
data_source: 
manager_account: 
developer_token: your_developer_token # 必配置项；
user_agent:
client_id: your_client_id # 必配置项；
client_secret: your_client_secret # 必配置项；
refresh_token: your_refresh_token # 必配置项；
```

{{< /expand >}}

### 官方文档

- [Google] [Google Ads API](https://developers.google.com/google-ads/api/docs/first-call/overview?hl=en)

## Apple Search Ads

### 接口申请方法

接口是物理的文件形式的密钥，见官方保姆级教程 [Implementing OAuth for the Apple Search Ads API](https://developer.apple.com/documentation/apple_search_ads/implementing_oauth_for_the_apple_search_ads_api)

### 官方文档

- [Apple] [Apple Search Ads API](https://developer.apple.com/documentation/apple_search_ads)

## 附：指标字典参考

| 编号  | 字段 | 定义 | 数据类型 | 数据来源 |
| --- | --- | --- | --- | --- |
| 1 | app_name | App标识 | varchar | 通过AdSet层级的AdPromotedObject对象的object_store_url或者application_id间接获得 |
| 2 | os_name | 设备类型，如android/ios | varchar  | 同上 |
| 3 | store_type | 应用商店 | varchar | 同上 |
| 4 | media_source | 流量来源 | varchar | 通过 Token 配置进行判断 |
| 5 | account_id | 广告账户ID | varchar | API |
| 6 | account_name | 广告账户名称 | varchar | API |
| 7 | campaign_id | 广告推广计划ID | varchar | API |
| 8 | campaign_name | 广告推广计划名称 | varchar | API |
| 9 | date | 日期 | date | API |
| 10 | country | 国家，ISO 3166标准 | varchar | API |
| 11 | impressions | 广告展示量 | int | API |
| 12 | clicks | 广告点击量 | int | API |
| 13 | install | 广告安装量 | int | API |
| 14 | cost | 广告花费 | float | API |
| 15 | purchase_value | 广告带来的收入 | float | API | 
| 16 | purchase | 付费次数 | int | API | 
| 17 | purchase_unique | 付费人数 | int | API | 
| 18 | optimizer | 优化师编号 | varchar | 通过正则 |
| 19 | data_source | 所属BM/MCC | varchar | 通过 Token 配置 |
| 20 | currency | 币种单位，需要统一为 USD | varchar | API |
| 21 | is_organic | 布尔值 | varchar | 通过映射配置 |
| 22 | attribution_setting | 归因窗口设置 | varchar | API |