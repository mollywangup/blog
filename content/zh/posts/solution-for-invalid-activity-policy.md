---
title: "广告风控指南：无效流量"
date: 2021-10-18T09:43:43Z
draft: false
description: 使用测试广告，遵守广告格式植入指南，基于 CTR 的优化，使用最新的变现 SDK.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Invalid activity
- Monetization
- AdMob
---

无效流量，也常称作无效活动，以下统一口径为无效流量。

## 结论

1. 减少误点击（划重点：各广告平台权重最高的数据指标是 CTR ）：
   - 可通过使用测试广告单元 ID 及添加测试设备，来避免开发者的误点击；
   - 可通过遵守广告格式植入指南，来避免用户的误点击；
   - 可通过优化广告请求及展示逻辑，来进行频次控制；
2. 使用最新的变现 SDK：
   - 可避免因 SDK 自身 bug 引发的相关问题；

## 无效流量

1. 定义见 [Invalid traffic](https://support.google.com/admob/answer/3342054?hl=en)
	可理解为：广告点击表现异常时，会被AdMob（各个广告平台大同小异）判定为无效流量。且无效流量不会产生收益，已产生的收益也会被收回；
	> 所有可能会**虚增**广告客户费用或发布商收入的点击或展示都属于无效流量。这其中包括蓄意制造的欺诈性流量，也包括误点击。 
2. 被判定为无效流量的后果：
	- **轻则限制广告填充**，即在广告请求环节返回`ERROR_CODE_NO_FILL`；
{{< expand "ERROR_CODE_NO_FILL" >}}
The ad request was successful, but no ad was returned due to lack of ad inventory.
{{< /expand >}}
	- **重则封变现账户**；

## 如何风控

### 建议做法一：使用测试广告

结论：**添加测试设备就安全**；

| 是否使用测试广告单元ID&nbsp;&nbsp;&nbsp; | 是否添加测试设备&nbsp;&nbsp;&nbsp; | 安全性评估&nbsp;&nbsp;&nbsp; | 建议程度 |
| ---------- | --------- | ----------------- | ---------- |
| *yes* | yes | 最安全 | 强烈建议 |
| no | yes | 100%安全 | 建议 |
| yes | no | 不太安全 | 不建议 |
| no | no | 不安全 | 禁止 |

### 建议做法二：遵守广告格式植入指南

#### 原生广告（Native Advanced）

植入指南：[Native ads policies & guidelines](https://support.google.com/admob/answer/6329638?hl=en)

1. 需手动添加广告标识（Ad Attribution），即下图中的黄色badge：

	<img src='/images/posts/ad-attribution-badge.png' alt='Ad Attribution badge'>

2. 广告背景必须不可点击：
   
	> 广告背景必须是不可点击的（即不含可供点击的“空白区域”）。如果您将图片要素用作广告背景，则该图片必须是不可点击的。

	<img src='/images/posts/ad-background-unclickable.png' alt='Ad background unclickable'>

#### 插屏广告（Interstitial）

植入指南：[Interstitial ad guidance](https://support.google.com/admob/answer/6066980?hl=en)

1. 插屏广告展示前后必须是不同的页面，即如 A页面 -> 插屏广告 -> B页面：

	<img src='/images/posts/interstitial-y.png' alt='best practice for Interstitial ad'>

2. 当进行插屏广告的展示时，需要确保App暂停跑接下来的流程；

	> https://developers.google.com/admob/android/interstitial#some_best_practices
	> Remember to pause the action when displaying an interstitial ad. 
	> For example, when you make the call to display an interstitial ad, be sure to pause any audio output being produced by your app.

#### 开屏广告（App Open）

植入指南：[App open ad guidance](https://support.google.com/admob/answer/9341964?hl=en)

1. 合规做法：

	启动页 -> 开屏广告（展示在启动页动画上） -> 主界面

	<img src='/images/posts/app-open-y.png' alt='best practice for App Open ad'>

2. 违规做法：

	- 启动页 -> 主界面 -> 开屏广告（展示在主界面上）

	<img src='/images/posts/app-open-n1.png' alt='not approved for App Open ad1'>

	- 启动页 -> 开屏广告（展示在空白/未知内容上） -> 主界面
    
	<img src='/images/posts/app-open-n2.png' alt='not approved for App Open ad2'>

### 建议做法三：基于 CTR 的优化

#### 经验 CTR

1. AdMob: 5%左右；
2. Facebook Audience Network: 5%内是安全线，10%是会判违规；

#### 优化启动规则

目标：降低广告请求的频次，同时提升用户体验；

具体做法：
1. 设置固定的启动时长，如10s；
2. 设置启动时间间隔，如10s；

#### 优化开屏位置的广告展示 

目标：降低误点击，使广告展示没那么突兀；

具体做法：给用户3s至10s的缓冲时间；

#### 优化 App 内页面间切换的动画

目标：页面切换丝滑，广告载入载出也丝滑；

具体做法：
1. 设置页面间的左滑/右滑规则；
2. 适配 RTL；

#### 完整且准确的跟踪广告活动

目标：对于用户在应用中完整的广告活动，做到有较高程度的把控；

具体做法：在广告加载、广告展示、广告点击事件中，收集点击的唯一标识`ad_response_id`参数；

#### 全屏广告格式阅后即焚

目标：降低误点击，不重复的无效展示全屏广告；

具体做法：
1. 展示全屏广告后 -> 用户立即切到后台 -> 再次热启动回到前台 -> **该全屏广告不应该还在**；
2. 当用户回到前台后，按照**手动关闭全屏广告**的流程，展示相应流程的下一步的页面；

### 建议做法四：使用最新的变现 SDK

详见各个 SDK 文档的 release notes.

### 参考别人的经验

1. 来自呼伟：[Admob账户“无效流量”被限制，账户解封经历](https://mp.weixin.qq.com/s/GlQqIXEX2afZoDjgIUuX3w)
2. 来自 AdMob 官方案例分享：[AdMob 开发者成功解除无效流量限制的亲身案例分享](https://mp.weixin.qq.com/s/mKDoqlt4hwGLdfZKkxREgA)