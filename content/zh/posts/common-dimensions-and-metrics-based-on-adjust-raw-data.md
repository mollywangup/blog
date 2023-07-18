---
title: "基于 Adjust 原始数据的指标体系"
date: 2023-04-16T08:39:46Z
draft: false
description: newUser, DAU, ARPDAU, ARPU (New), DAV, eCPM, RR, LTV 等。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
- SQL
categories:
- MMP
---

本文基于 Adjust（原始数据） -> S3（云存储）-> Druid（数仓）-> Superset（可视化）。

几点说明：
- 共两个阶段会对已有字段（以下称为列）进行加工：
  - 写入数仓时：在 Adjust/S3 原有列的基础上；
  - 写入数仓后：在 Druid 原有列的基础上，也就是可视化查询时；
- 示例的 SQL 语句省略了除0的情况；
- Druid 不支持窗口函数；

## 统计原则

### 一个 ID，两个时间戳

- `adid`：用户唯一标识；
- `installed_at`：首次打开的时间戳；
- `created_at`：事件发生的时间戳，在数仓中为`__time`；（Druid 需要）

{{< notice info >}}
时区说明：时间戳类型全部为 UTC 时区；
{{< /notice >}}

### 统计次数

没有使用 `COUNT(*)`，是为了事件**去重**。

```sql
COUNT(DISTINCT __time)
```

### 统计人数

使用的是 Adjust 的设备标识 `adid`.

```sql
COUNT(DISTINCT adid)
```

### 统计频次

次数 / 人数。

```sql
COUNT(DISTINCT __time) / COUNT(DISTINCT adid)
```

## 新增计算列

{{< alert theme="warning" >}}
⚠ 注意：写入数仓前的批量任务中新增，因此是基于 **Adjust/S3** 的原始列。
{{< /alert >}}

### days_x

✍ **同期群分析，本质上是围绕 `created_at` 和 `installed_at` 之间相差的天数展开的**。

```sql
-- days_x
TIMESTAMPDIFF(DAY, MILLIS_TO_TIMESTAMP("{installed_at}" * 1000), MILLIS_TO_TIMESTAMP("{created_at}" * 1000)) AS "days_x"

-- hours_x
TIMESTAMPDIFF(HOUR, MILLIS_TO_TIMESTAMP("{installed_at}" * 1000), MILLIS_TO_TIMESTAMP("{created_at}" * 1000)) AS "hours_x"

-- minutes_x
TIMESTAMPDIFF(MINUTE, MILLIS_TO_TIMESTAMP("{installed_at}" * 1000), MILLIS_TO_TIMESTAMP("{created_at}" * 1000)) AS "minutes_x"
```

{{< alert theme="info" >}}
📌 关于次日，有两种可能的定义：

1. 严格间隔 24h 为次日；
2. 过了零点就是次日了;

<br>
⚠ Adjust 和这里计算 days_x 的方式，都属于第一种。
{{< /alert >}}

### event_name

取齐，便于分析。

```sql
CASE 
    WHEN "{activity_kind}" <> 'event' THEN "{activity_kind}"
    ELSE "{event_name}"
END AS "event_name"
```

### media_source

用于区分流量来源（归因）。
⚠ 需要按需修改：实际接入的流量源。

```sql
CASE
    WHEN "{fb_install_referrer_campaign_group_id}" IS NOT NULL THEN 'Facebook Ads'
    ELSE 'Organic'
END AS "media_source"
```

### revenue_kind

用于区分收入类型。
⚠ 需要按需修改：收入事件名称。

```sql
CASE
    WHEN "{activity_kind}" = 'ad_revenue' THEN 'Ad'
    WHEN "{event_name}" = 'purchase' THEN 'IAP'
    WHEN "{event_name}" = 'subscription' THEN 'Subscription'
    ELSE 'Unknown'
END AS "revenue_kind"
```

### revenue

用于统一计算所有类型的收入：广告、内购（一次性）、订阅。
⚠ 需要按需修改：实际接入的聚合平台、收入事件名称。

```sql
CASE
    WHEN "{event_name}" IN ('purchase', 'subscription') THEN "[price]"
    WHEN "{activity_kind}" = 'ad_revenue' AND "{ad_mediation_platform}" = 'applovin_max_sdk' THEN "{reporting_revenue}"
    ELSE 0
END AS "revenue"
```

### is_test_device

未严格在 sandbox 环境测试时，手动维护的内部测试设置列表，用于剥离出生成环境的数据。

```sql
CASE 
    WHEN "{gps_adid}" IN ('83640d56-411c-46b2-9e2b-10ca1ad1abe1', 'af7f3092-445d-4db0-b682-297b5bb5fdc4') THEN 't'
    WHEN REGEXP_LIKE("{device_name}", 'XXXMobi') THEN 't'
    ELSE 'f'
END AS "is_test_device"
```

## 基础指标

{{< alert theme="warning" >}}
⚠ 注意：写入数仓后计算的，因此是基于 **Druid** 的原始列。
{{< /alert >}}

### newUser

新增。

```sql
COUNT(DISTINCT CASE WHEN activity_kind = 'install' THEN adid END)
```

### DAU

关于活跃的定义：

- Adjust：与应用发生互动，见 [What is an active user?](https://www.adjust.com/glossary/active-user/)
- Firebase：用户在应用前台互动，并记录了 `user_engagement` 事件，见 [User activity over time](https://support.google.com/firebase/answer/6317517?hl=en#active-users&zippy=%2Cin-this-article)
- BigQuery：至少发生了一个事件，且该事件的参数 `engagement_time_msec` > 0，见 [N-day active users](https://support.google.com/analytics/answer/9037342?hl=en#ndayactives&zippy=%2Cin-this-article)
- 自行定义：至少发生了一次自定义的 `login` 事件；

```sql
COUNT(DISTINCT CASE WHEN event_name = 'login' THEN adid END)
```

### ARPDAU

活跃用户的 ARPU，其中活跃使用上述自定义的。

```sql
SUM(revenue) / DAU
```

### ARPU (New)

新用户的 ARPU.

```sql
SUM(revenue) / newUser
```

## 同期群指标

### RR

留存率。与上述活跃定义取齐，留存率计算公式：`Rx = Dx活跃 / D0活跃`。

```sql
-- 0D
COUNT(DISTINCT CASE WHEN event_name = 'login' AND days_x = 0 THEN adid END)

-- 1D
COUNT(DISTINCT CASE WHEN event_name = 'login' AND days_x = 1 THEN adid END)

-- 7D
COUNT(DISTINCT CASE WHEN event_name = 'login' AND days_x = 7 THEN adid END)

-- R1
1D / 0D

-- R7
7D / 0D
```

### LTV

给定周期内的总收入。

```sql
-- LT7
SUM(CASE WHEN days_x <= 6 THEN revenue END)

-- LT14
SUM(CASE WHEN days_x <= 13 THEN revenue END)
```

## 广告变现指标

### Imps

广告展示次数。

```sql
COUNT(DISTINCT CASE WHEN activity_kind = 'ad_revenue' THEN __time END)
```

### DAV

广告展示人数。

```sql
COUNT(DISTINCT CASE WHEN activity_kind = 'ad_revenue' THEN adid END)
```

### eCPM

```sql
SUM(CASE WHEN revenue_kind = 'Ad' THEN revenue END) / Imps * 1000
```

### Imps per DAU

活跃用户，平均广告展示次数。

```sql
Imps / DAU
```

### Imps per DAV

看到广告的用户，平均广告展示次数。

```sql
Imps / DAV
```

## 附：原始数据结构

原始数据示例，有助于理解数据结构。

{{< expand "Adjust/S3 原始数据举例（已脱敏）" >}}
```plaintext
{environment}	{activity_kind}	{created_at}	{installed_at}	{timezone}	{app_name}	{app_version_short}	{country}	{city}	{os_name}	{os_version}	{device_type}	{device_manufacturer}	{device_name}	{gps_adid}	{adid}	{android_id}	{language}	{tracker}	{tracker_name}	{is_organic}	{is_reattributed}	{reporting_currency}	{reporting_cost}	{reporting_revenue}	{event}	{event_name}	[level]	[id]	[price]	[subscription_type]	[transaction_id]	[error_code]	{ad_mediation_platform}	{ad_revenue_network}	[ad_format]	[ad_space]	[ad_network_name]	[ad_revenue]	[ad_error_code]	[is_vip]	[trial_state]	{fb_install_referrer_publisher_platform}	{fb_install_referrer_campaign_group_name}	{fb_install_referrer_campaign_group_id}	{fb_install_referrer_campaign_name}	{fb_install_referrer_campaign_id}	{fb_install_referrer_adgroup_name}	{fb_install_referrer_adgroup_id}	
production	event	1687957195	1687956440	UTC-0600	PACKAGE_NAME	1.1.1	us	Nampa	android	13	phone	Samsung	GalaxyA715G	21d01e91-1338-44ba-94c1-9a392d832d5b	df50cd2988ddf1f78b6116c22389c557		en	unattr	Unattributed	0	0				71cxx6	claim_rewarded_ad									Rewarded	1110004				f	never	unknown	CAMPAIGN_NAME	23855267775170400	ADSET_NAME	23855267775220400	AD_NAME	23855268033500400	
```
{{< /expand >}}

{{< expand "Druid 原始数据举例（已脱敏）" >}}
```json
{
    "__time":"2023-07-14T07:28:54.000Z",
    "environment":"production",
    "activity_kind":"event",
    "installed_at":1689319730000,
    "timezone":"UTC-0700",
    "app_name":"PACKAGE_NAME",
    "app_version_short":"1.1.2",
    "country":"us",
    "city":"Los Angeles",
    "os_name":"android",
    "os_version":"12",
    "device_type":"phone",
    "device_manufacturer":"Motorola",
    "device_name":"motogstylus5G(2022)",
    "gps_adid":"1af0adeb-5f20-4b13-a7c7-ae11cbf55947",
    "adid":"b82355d7d28c623bf918156c8f7486b1",
    "android_id":"",
    "language":"en",
    "tracker":"11n573m9",
    "tracker_name":"Organic",
    "is_organic":1,
    "is_reattributed":0,
    "reporting_currency":"",
    "reporting_cost":0,
    "reporting_revenue":0,
    "event":"mbvbnw",
    "event_name2":"login",
    "level":"",
    "id":"",
    "price":0,
    "subscription_type":"",
    "transaction_id":"",
    "error_code":"",
    "ad_mediation_platform":"",
    "ad_revenue_network":"",
    "ad_format":"Unknown",
    "ad_space":"Unknown",
    "ad_network_name":"Unknown",
    "ad_revenue":0,
    "ad_error_code":"",
    "is_vip":"f",
    "trial_state":"never",
    "fb_network_name":"instagram",
    "fb_campaign_name":"CAMPAIGN_NAME",
    "fb_campaign_id":"23855267775170453",
    "fb_adgroup_name":"ADSET_NAME",
    "fb_adgroup_id":"23855267775220453",
    "fb_creative_name":"AD_NAME",
    "fb_creative_id":"23855268033500453",
    "phase":"",
    "update_at":1689328800513,
    "days_x":0,
    "hours_x":0,
    "minutes_x":0,
    "event_name":"login",
    "revenue_kind":"Unknown",
    "media_source":"Facebook Ads",
    "revenue":0,
    "is_test_device":"f"
}
```
{{< /expand >}}
