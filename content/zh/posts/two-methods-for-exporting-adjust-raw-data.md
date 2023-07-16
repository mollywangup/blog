---
title: "将 Adjust 原始数据导出的两种方法"
date: 2023-04-04T16:01:08Z
draft: false
description: 共两种方法，实时回传和每小时上传 CSV 至云储存。
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

由于 Adjust 看板数据具有较大的分析局限性，因此有必要使用[原始数据导出功能](https://help.adjust.com/en/article/raw-data-exports)，用以更细颗粒度、更自由的多维度交叉分析。

共两种方法，一种是导出至云存储，一种是实时回传给自己的服务器。

![How it works（图源 Adjust）](https://images.ctfassets.net/5s247im0esyq/5IzZDHUzGTvKFMe2IGnPCj/5b60d8ac5c97a05b2e71976c7be8b77f/02075bdf-e44b-4d3c-ac3b-31b736c20a56.png)

<!-- {{< expand "为什么不建议使用 BigQuery?" >}}

1. 时效性：
   - BigQuery：延迟1天半，实时额外收费见 [Data extraction pricing](https://cloud.google.com/bigquery/pricing#data_extraction_pricing)；
   - Adjust：接近实时；
2. 费用成本：
   - BigQuery：相对高额的计算/分析费用，可参考 [How BigQuery pricing works](https://cloud.google.com/bigquery/#section-5)
   - Adjust：按非自然量的安装收费 + 其他附加服务费（广告收入、订阅）；
3. 数据的使用价值：
   - BigQuery：侧重于用户行为分析；
   - Adjust：专业的归因供应商，打通推广+变现两侧；
4. 开发成本：
   - BigQuery：使用 Firebase 进行事件统计，需要单独处理打通推广侧（目前仅可实现 Facebook Ads，未来其他推广平台都是潜在的坑）；
   - Adjust：自备服务器，自建数据库（但原始数据已经接近结构化了）； -->

<!-- {{< /expand >}} -->

## 导出机制

基于事件（广义）及对应的事件参数导出。

### 支持的事件（广义）

Adjust 称为 `activity_kind`，但本质上属于**事件**。对应的是触发机制。常见需要导出的事件如下：

- Click
- Installs
- Events：需要手动创建`event_token`；
- Ad revenue：需要依赖聚合 SDK 获取，且额外收费；
- Subscriptions：需要手动添加额外的代码，且额外收费；
- Uninstall：需要依赖 FCM SDK 每天发送静默推送消息来监测是否已卸载；

### 支持的事件参数

对应的是数据颗粒度。按照是否需要手动设置，共分为以下两类：

- 内置参数：对应 `Placeholder`，支持的列表见 [Adjust Placeholders for Partners
](https://partners.adjust.com/placeholders)
- 自定义参数：对应 `CallbackParameter`，支持的上报方式见：
	- SDK 方式：[Callback parameters](https://help.adjust.com/en/article/event-tracking-android-sdk#callback-parameters)
	- S2S 方式：[Share custom data](https://help.adjust.com/en/article/server-to-server-events#share-custom-data)

## 方法一：CSV 至云储存

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
## 方法二：实时回传

1. 设置实时回传：[Set up callbacks](https://help.adjust.com/en/article/set-up-callbacks)<br>
	<img src='/images/posts/setup-callbacks.png' alt='setup-callbacks'><br>
2. 需要提前在自有服务器配置回传 URL：
	- [Callback structure](https://help.adjust.com/en/article/callback-structure)
	- [Recommended placeholders for callbacks](https://help.adjust.com/en/article/recommended-placeholders-callbacks)
