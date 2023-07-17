---
title: "基于 BigQuery 原始数据的指标体系"
date: 2023-02-28T08:39:46Z
draft: false
description: newUser, DAU, ARPDAU, ARPU (New), DAV, eCPM, RR, LTV 等。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- BigQuery
- SQL
categories:
- 
---

本文基于 Firebase（原始数据）-> GCS（云存储）-> BigQuery（数仓）-> Looker Studio（可视化）。

几点说明：
- 共两个阶段会对已有字段（以下称为列）进行加工：
  - Looker Studio 连接 BigQuery 时：加在原有列的基础上；
  - Looker Studio 可视化查询时：在 Looker 已导入列的基础上；
- 示例的 SQL 语句省略了除0的情况；
- BigQuery 支持窗口函数；

## 统计原则

### 一个 ID，两个时间戳

- `user_pseudo_id`：用户唯一标识；
- `user_first_touch_timestamp`：首次打开的时间戳；
- `event_timestamp`：事件发生的时间戳；

### 统计次数

没有使用 `COUNT(*)`，是为了事件**去重**。

```sql
COUNT(DISTINCT event_timestamp)
```

### 统计人数

使用的是 Firebase/BigQuery 的匿名用户标识 `user_pseudo_id`.

```sql
COUNT(DISTINCT user_pseudo_id)
```

### 统计频次

次数 / 人数。

```sql
COUNT(DISTINCT event_timestamp) / COUNT(DISTINCT user_pseudo_id)
```

## 新增计算列

{{< alert theme="warning" >}}
⚠ 注意：写入数仓前的批量任务中新增，因此是基于 **** 的原始列。
{{< /alert >}}

### days_x

✍ **同期群分析，本质上是围绕 `event_timestamp` 和 `user_first_touch_timestamp` 之间相差的天数展开的**。

```sql
-- days_x
CAST(TIMESTAMP_DIFF(TIMESTAMP_MICROS(event_timestamp), TIMESTAMP_MICROS(user_first_touch_timestamp), DAY) AS INT64) AS days_x

-- hours_x
CAST(TIMESTAMP_DIFF(TIMESTAMP_MICROS(event_timestamp), TIMESTAMP_MICROS(user_first_touch_timestamp), HOUR) AS INT64) AS hours_x

-- minutes_x
CAST(TIMESTAMP_DIFF(TIMESTAMP_MICROS(event_timestamp), TIMESTAMP_MICROS(user_first_touch_timestamp), MINUTE) AS INT64) AS minutes_x
```

### revenue_kind

用于区分收入类型。

```sql
CASE event_name
    WHEN 'ad_play_ok' THEN 'AD'
    WHEN 'purchase_gold_ok' THEN 'IAP'
    ELSE 'unknown'
END AS revenue_kind
```

### media_source

用于区分流量来源（归因）。

```sql
CASE
    WHEN "{fb_install_referrer_campaign_group_id}" IS NOT NULL THEN 'Facebook Ads'
    ELSE 'Organic'
END AS "media_source"
```

### revenue

用于统一计算所有类型的收入：广告、内购（一次性）、订阅。

```sql
CASE
    WHEN "{event_name}" IN ('purchase', 'subscription') THEN "[price]"
    WHEN "{activity_kind}" = 'ad_revenue' AND "{ad_mediation_platform}" = 'applovin_max_sdk' THEN "{reporting_revenue}"
    ELSE 0
END AS "revenue"
```

## 基础指标

{{< alert theme="warning" >}}
⚠ 注意：写入数仓后计算的，因此是基于**数仓**的原始列。
{{< /alert >}}

### newUser

新增。

```sql
COUNT(DISTINCT CASE WHEN event_name = 'first_open' THEN user_pseudo_id END)
```

### DAU

关于活跃的定义：

- Adjust：与应用发生互动，见 [What is an active user?](https://www.adjust.com/glossary/active-user/)；
- Firebase：用户在应用前台互动，并记录了 `user_engagement` 事件，见 [User activity over time](https://support.google.com/firebase/answer/6317517?hl=en#active-users&zippy=%2Cin-this-article)；
- BigQuery：至少发生了一个事件，且该事件的参数 `engagement_time_msec` > 0，见 [N-day active users](https://support.google.com/analytics/answer/9037342?hl=en#ndayactives&zippy=%2Cin-this-article)

```sql
-- Firebase定义的活跃
COUNT(DISTINCT CASE WHEN event_name = 'user_engagement' THEN user_pseudo_id END)

-- 自定义的活跃
COUNT(DISTINCT CASE WHEN event_name = 'login' THEN user_pseudo_id END)
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

留存率。与上述活跃定义取齐，留存率计算方式：`Rx = Dx活跃 / D0活跃`。

```sql
-- 0D
COUNT(DISTINCT CASE WHEN event_name = 'user_engagement' AND days_x = 0 THEN user_pseudo_id END)

-- 1D
COUNT(DISTINCT CASE WHEN event_name = 'user_engagement' AND days_x = 1 THEN user_pseudo_id END)

-- 7D
COUNT(DISTINCT CASE WHEN event_name = 'user_engagement' AND days_x = 7 THEN user_pseudo_id END)

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
COUNT(DISTINCT CASE WHEN event_name = 'ad_play_ok' THEN event_timestamp END)
```

### DAV

广告展示人数。

```sql
COUNT(DISTINCT CASE WHEN event_name = 'ad_play_ok' THEN user_pseudo_id END)
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