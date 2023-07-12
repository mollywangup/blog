---
title: "将 Adjust 原始数据导出的两种方式"
date: 2023-04-04T16:01:08Z
draft: false
description: 共两种方式，实时回传和每小时 csv 同步到云存储。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
categories:
- SDK
---

## 背景信息

为什么需要 Adjust 的原始数据？主要原因有以下两点：

1. 原始数据具有更细的颗粒度，而 Adjust 看板不支持按照事件参数进行拆分维度查看；
2. 原始数据具有更广的分析维度，比如设备信息等；

因此，使用原始数据具有更自由更广阔的分析空间。

为什么是 Adjust 而不是 Firebase + BigQuery？

1. 时效性：
   - BigQuery：延迟1天半（也可以实时得加钱）；
   - Adjust：接近实时；
2. 费用成本：
   - BigQuery：相对高额的分析费用，可参考 BigQuery收费标准及桃花源花费 
   - Adjust：按非自然量的安装收费 + 广告收入额外收费；
3. 数据的使用价值：
   - BigQuery：侧重于用户行为分析；
   - Adjust：专业的归因供应商，打通推广+变现两侧；
4. 开发成本：
   - BigQuery：使用Firebase进行事件统计，需要单独处理打通推广侧（目前仅可实现Facebook Ads，未来其他推广平台都是潜在的坑）；
   - Adjust：自备服务器，自建数据库（但原始数据已经接近结构化了）；

## 如何导出

### 说明

1. 基于事件和事件参数导出；
2. 支持的事件：包含自动/手动统计：
  1. 自动统计的事件：除了Events，其余全部为自动统计事件；
  2. 手动统计的事件：Events；

[Recommended placeholders for callbacks](https://help.adjust.com/en/article/recommended-placeholders-callbacks)

3. 支持的事件参数：包含自动/手动两类：
  1. 自动统计的参数：对应叫做Placeholder，支持的列表见：https://partners.adjust.com/placeholders
  2. 手动统计的参数：对应叫做CallbackParameter，支持的上报方式见：
    - Adjust SDK方式上报（够用了）：https://help.adjust.com/en/article/event-tracking-android-sdk#callback-parameters
    - Adjust S2S方式上报：https://help.adjust.com/en/article/server-to-server-events#share-custom-data

### 方法一：CSV导出至云服务器

1. 每小时自动导出一次：https://help.adjust.com/en/article/csv-uploads
   <img src='/images/posts/csv-uploads.png' alt='csv-uploads'>

2. 需要提前设置接收的云服务器（二选一）：
  1. AWS S3：https://help.adjust.com/en/article/amazon-s3#set-up-in-aws-console
  2. Google Cloud Storage：https://help.adjust.com/en/article/google-cloud-storage#set-up-in-google-cloud-console
3. 需要提前设置导出的列格式：
  如针对ad revenue事件：https://help.adjust.com/en/article/csv-uploads#format-your-csv-definition
```plaintext
{app_name},{app_version_short},{os_name},{os_version},{country},{device_type},{gps_adid},{random_user_id},[user_id],{installed_at},{created_at},{is_organic},{network_name},{campaign_name},{adgroup_name},{ad_mediation_platform},{ad_revenue_network},{ad_impressions_count},{currency},{reporting_revenue}
```

### 方法二：实时回传

1. 实时回传：https://help.adjust.com/en/article/set-up-callbacks
   <img src='/images/posts/setup-callbacks.png' alt='setup-callbacks'>

2. 需要提前在自有服务器配置回传URL：
  [Callback structure](https://help.adjust.com/en/article/callback-structure)
  [Recommended placeholders for callbacks](https://help.adjust.com/en/article/recommended-placeholders-callbacks)
