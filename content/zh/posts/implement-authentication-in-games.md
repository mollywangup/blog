---
title: "接入各种登录方式"
date: 2022-03-02T15:03:42Z
draft: false
description: 可直接接入，可间接接入。常见登录方式有 Facebook/Google/Play Games.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Firebase
categories:
- Login
---

## 背景信息

### 三个目标

1. 提高用户标识的唯一性：
  {{< expand "建议做法" >}}
     1. 首次打开时：强行匿名注册，并生成唯一标识`user_id`；
     2. 首次非匿名登录时：取该登录方式对应的唯一标识，并与`user_id`进行关联；
     3. 每次登录游戏时：指点击`开始游戏`按钮时，需确保有`user_id`信息；
  {{< /expand >}}
2. 保存游戏进度：
  {{< expand "建议做法" >}}
     - **非匿名登录**：将与之关联的游戏进度上传至服务端；
       - 主要用途：清除全部数据、卸载重装、跨设备，非匿名登录后可直接下载游戏进度；
     - **匿名登录**：不用管；
  {{< /expand >}}
3. 社交：分享、互赠礼物、查看排行榜等；

### 不同登录方式的对比

1. Facebook 登录：强社交 + 跨设备系统 + 接入简单，最优先建议；
2. Play Games 登录：游戏 + 弱社交 + 谷歌全家桶 + 仅限安卓系统；
3. Google 登录：优先级最低，暂不考虑；

## 方式一：直接接入

### 接入 Facebook 登录方式

#### 官方文档
 
[Facebook]：
   - [Facebook SDK for Unity Reference](https://developers.facebook.com/docs/unity/reference/current)
   - [Facebook Login Best Practices](https://developers.facebook.com/docs/facebook-login/best-practices)
   - [User Experience Design](https://developers.facebook.com/docs/facebook-login/userexperience)
  
#### 流程概述

1. 登入（未登录 -> 登录）：
  点击登录按钮 -> 处理登录流程 ->（可选）登录成功/失败的交互反馈 -> 同步按钮状态；
1. 登出（登录 -> 未登录）：
  点击退出按钮 -> 处理退出流程 ->（可选）退出成功的交互反馈 -> 同步按钮状态；

#### 测试方法

需要测试以下三种可能的情形：

1. 设备上未安装Facebook，跳转至设备浏览器的Facebook web登录界面（m.facebook.com）；
2. 设备上已安装Facebook但未登录，跳转至Facebook app登录界面；
3. 设备上已安装Facebook且已登录，直接原地申请获取玩家权限；

### 接入 Play Games 登录方式

#### 官方文档

1. [Android] [Get started with the Google Play Games plugin for Unity](https://developer.android.com/games/pgs/unity/unity-start)
2. [GitHub] [Sign in](https://github.com/playgameservices/play-games-plugin-for-unity#sign-in)

#### 流程概述

{{< notice info >}}
仅需登入，无登出入口（因为方法已被官方删除）；
{{< /notice >}}

以下是登入流程：
1. 设备无谷歌服务框架的，**直接跳过**，不作任何处理；
2. 已创建 Play 游戏账号的，**系统自动登录**；
3. 未创建 Play 游戏账号的，**需要手动处理**（ManuallyAuthenticate）：
   - 可直接放弃集成；
   - 可强制引导登录，引导玩家以当前谷歌账号创建 Play 游戏账号；

#### 测试方法

需要测试以下三种可能的情形：

1. 设备上无谷歌服务框架，直接跳过，不作任何处理；
2. 设备上有谷歌服务框架，但未创建 Play 游戏账号，引导创建 Play 游戏账号；
   - 未登录 Google 账号，未创建 Play 游戏账号；
   - 已登录 Google 账号，未创建 Play 游戏账号；
3. 设备上有谷歌服务框架，且已创建 Play 游戏账号，直接自动登录；

{{< notice info >}}
补充说明：创建和删除 Play 游戏账号，直接在系统设置里操作，相对容易操作；
{{< /notice >}}

## 方式二：间接接入

使用 **Firebase Authentication** 接入。

### 关于 Firebase Authentication

1. 支持的登录方式

  结论：**支持第三方登录如 Facebook/Google/Play Games 等，也支持直接注册如邮件/电话/匿名**；

  <img src='/images/posts/firebase-authentication-sign-in-providers.png' alt='Firebase Authentication: sign-in providers'>

2. 收费标准

  结论：**月活5W以内免费，超过部分每个$0.0025-$0.0055**；

  1. Firebase全产品线：[Pricing plans](https://firebase.google.com/pricing)
  2. Firebase Authenticate：[No cost and Pay as you go](https://firebase.google.com/docs/auth#identity-platform-limits)

### 官方文档

1. [Firebase] Firebase Authentication in Unity：
   - [Get Started](https://firebase.google.com/docs/auth/unity/start)
   - [Anonymous Login](https://firebase.google.com/docs/auth/unity/anonymous-auth)
   - [Facebook Login](https://firebase.google.com/docs/auth/unity/facebook-login)
   - [Play Games Login](https://firebase.google.com/docs/auth/unity/play-games)
   - [Link Multiple Auth Providers to an Account](https://firebase.google.com/docs/auth/unity/account-linking)
2. [Facebook] Facebook SDK for Unity：
   - [Facebook Login Examples](https://developers.facebook.com/docs/unity/examples#login)
   - [Facebook Login Permissions](https://developers.facebook.com/docs/permissions/reference#login_permissions)
    关于权限范围：
     - 默认可向玩家申请：`email`和`public_profile`；
     - 如需核心的社交功能，则需要额外的权限：`user_friends`，具体申请路径是：
       - Facebook开发者后台 -> App Review；
3. [GitHub] Google Play Games plugin for Unity：
   - [Sign in](https://github.com/playgameservices/play-games-plugin-for-unity#sign-in)
   - [Add Achievements and Leaderboards](https://github.com/playgameservices/play-games-plugin-for-unity#add-achievements-and-leaderboards)