---
title: "使用 Firebase Authentication 接入不同登录方式"
date: 2023-03-28T15:03:42Z
draft: false
description: 
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
   - 建议做法：
     1. 首次打开时：强行匿名注册，并生成唯一标识user_id；
     2. 首次非匿名登录时：取该登录方式对应的唯一标识，并与user_id进行关联；
     3. 每次登录游戏时：指点击`开始游戏`按钮时，需确保有user_id信息；
2. 保存游戏进度：
   - 建议做法：
     - **非匿名登录**：将与之关联的游戏进度上传至服务端；
       - 主要用途：清除全部数据、卸载重装、跨设备，非匿名登录后可直接下载游戏进度；
     - **匿名登录**：不用管；
3. 社交：分享、互赠礼物、查看排行榜等；
4. 恢复购买（非目标）：
    因为恢复购买与是否登录无关，逻辑是通过Google Play的API，先获取本地设备已登录的Google Play Store账号，然后查询对应的历史购买记录；

### 不同登录方式的对比

1. Facebook登录：强社交 + 跨设备系统 + 接入简单，最优先建议；
2. Play Games登录：游戏 + 弱社交 + 谷歌全家桶 + 仅限安卓系统；
3. Google登录：优先级最低，暂不考虑；

## 关于 Firebase Authentication

### 支持的登录方式

结论：**支持第三方登录如 Facebook/Google/Play Games 等，也支持直接注册如邮件/电话/匿名**；

<img src='/images/posts/firebase-authentication-sign-in-providers.png' alt='Firebase Authentication: sign-in providers'>

### 收费标准

结论：**月活5W以内免费，超过部分每个$0.0025-$0.0055**；

1. Firebase全产品线：https://firebase.google.com/pricing
2. Firebase Authenticate：https://firebase.google.com/docs/auth#identity-platform-limits

## 具体接入

### 接入Facebook登录

官方文档

1. [Firebase] Firebase Authentication in Unity：
   - [Get Started](https://firebase.google.com/docs/auth/unity/start)
   - [Anonymous Login](https://firebase.google.com/docs/auth/unity/anonymous-auth)
   - [Facebook Login](https://firebase.google.com/docs/auth/unity/facebook-login)
   - [Link Multiple Auth Providers to an Account](https://firebase.google.com/docs/auth/unity/account-linking)
2. [Facebook] Facebook SDK for Unity：
   - [Facebook Login Examples](https://developers.facebook.com/docs/unity/examples#login)
   - [Facebook Login Permissions](https://developers.facebook.com/docs/permissions/reference#login_permissions)
    关于权限范围：
     - 默认可向玩家申请：`email`和`public_profile`；
     - 如需核心的社交功能，则需要额外的权限：`user_friends`，具体申请路径是：
       - Facebook开发者后台 -> App Review；

### 接入Play Games登录方式

官方文档

1. [Firebase] Firebase Authentication in Unity：
   - [Get Started](https://firebase.google.com/docs/auth/unity/start)
   - [Anonymous Login](https://firebase.google.com/docs/auth/unity/anonymous-auth)
   - [Play Games Login](https://firebase.google.com/docs/auth/unity/play-games)
   - [Link Multiple Auth Providers to an Account](https://firebase.google.com/docs/auth/unity/account-linking)
2. [GitHub] Google Play Games plugin for Unity：
   - [Sign in](https://github.com/playgameservices/play-games-plugin-for-unity#sign-in)
   - [Add Achievements and Leaderboards](https://github.com/playgameservices/play-games-plugin-for-unity#add-achievements-and-leaderboards)