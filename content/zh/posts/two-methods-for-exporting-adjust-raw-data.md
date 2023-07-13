---
title: "将 Adjust 原始数据导出的两种方式"
date: 2023-04-04T16:01:08Z
draft: false
description: 共两种方式，实时回传和每小时上传 CSV 至云储存。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
categories:
- SDK
- MMP
---

## 背景信息

### Why 原始数据？

1. 看板数据不支持按照**事件参数**进行更细颗粒度的分析，而原始数据可以；
2. 看板数据不支持自由的多维度交叉分析，而原始数据可以；
3. 看板数据无法分析默认统计到的一些玩家属性如设备信息等，而原始数据可以；<br><br>

因此，使用原始数据具有更自由更广阔的分析空间。

### Why Adjust NOT BigQuery？

1. 时效性：
   - BigQuery：延迟1天半（也可以实时但得加钱）；
   - Adjust：接近实时；
2. 费用成本：
   - BigQuery：相对高额的计算/分析费用，可参考 [How BigQuery pricing works](https://cloud.google.com/bigquery/#section-5)
   - Adjust：按非自然量的安装收费 + 其他附加服务费（广告收入、订阅）；
3. 数据的使用价值：
   - BigQuery：侧重于用户行为分析；
   - Adjust：专业的归因供应商，打通推广+变现两侧；
4. 开发成本：
   - BigQuery：使用 Firebase 进行事件统计，需要单独处理打通推广侧（目前仅可实现 Facebook Ads，未来其他推广平台都是潜在的坑）；
   - Adjust：自备服务器，自建数据库（但原始数据已经接近结构化了）；

## 如何导出

### 说明

1. 基于事件和事件参数导出；
2. 支持的事件：包含自动/手动统计：
	- 自动统计的事件：除了 **`Events`**，其余全部为自动统计事件；
	- 手动统计的事件：**`Events`**；<br>
	<img src='/images/posts/recommended-placeholders-for-callbacks.png' alt='recommended-placeholders-for-callbacks'><br>
3. 支持的事件参数：包含自动/手动两类：
	- 自动统计的参数：对应叫做`Placeholder`，支持的列表见 [Adjust Placeholders for Partners
](https://partners.adjust.com/placeholders)
	- 手动统计的参数：对应叫做`CallbackParameter`，支持的上报方式见：
		- Adjust SDK方式上报（够用了）：[Callback parameters](https://help.adjust.com/en/article/event-tracking-android-sdk#callback-parameters)
		- Adjust S2S方式上报：[Share custom data](https://help.adjust.com/en/article/server-to-server-events#share-custom-data)

### 方法一：CSV 至云储存

1. 设置每小时自动导出一次：[CSV uploads to cloud storage](https://help.adjust.com/en/article/csv-uploads)

2. 需要提前设置接收的云服务器（二选一）：
	- AWS S3：[Set up your project in the AWS Management Console](https://help.adjust.com/en/article/amazon-s3#set-up-in-aws-console)
	- Google Cloud Storage：[Set up your project in the Google Cloud Console](https://help.adjust.com/en/article/google-cloud-storage#set-up-in-google-cloud-console)

3. 需要提前设置导出的列格式：[Format your CSV definition](https://help.adjust.com/en/article/csv-uploads#format-your-csv-definition)
	- 共以下3种类型的参数作为列：
		- 常量：使用双引号，如`"my constant"`
		- 内置参数：使用花括号，如`{gps_adid}`
		- 自定义参数：使用中括号，如`[user_id]`
	- 例子：
```plaintext
"my constant",{gps_adid},[user_id],{installed_at},{event_name},[item_number],{reporting_revenue}
```

### 方法二：实时回传

1. 设置实时回传：[Set up callbacks](https://help.adjust.com/en/article/set-up-callbacks)<br>
	<img src='/images/posts/setup-callbacks.png' alt='setup-callbacks'><br>
2. 需要提前在自有服务器配置回传 URL：
	- [Callback structure](https://help.adjust.com/en/article/callback-structure)
	- [Recommended placeholders for callbacks](https://help.adjust.com/en/article/recommended-placeholders-callbacks)
