---
title: "Session"
date: 2025-01-12T01:42:27Z
draft: false
description: Adjust 对 Session 的定义及统计规则。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Adjust
categories:
- MMP
---

## Session 定义

官方定义详见 <a href="https://www.adjust.com/zh/glossary/session/" target="_blank">会话定义</a>

> 应用会话是指用户在安装后与应用进行交互的行为。

## Session 统计规则

### 触发机制

每次应用启动时判断，如果本次 Session 已累计超过 30 分钟，则触发一个新 Session 事件，否则不触发。具体规则如下图：

<img src="https://a.storyblok.com/f/47007/2400x1260/d428ff5b36/240529_glossary_sessions_zh_v02.png/m/2880x0/filters:quality(80)" alt="Session 统计规则（图源 Adjust）">

### 数据存储

推荐占位符详见 <a href="https://help.adjust.com/zh/article/session-callbacks" target="_blank">会话回传</a>

| placeholders&nbsp;&nbsp;&nbsp; | 定义&nbsp;&nbsp;&nbsp; | 补充说明&nbsp;&nbsp;&nbsp; |
| ---------- | ---------- | ---------- |
| **last_session_time** | 上次会话时间戳 | 所有都生效 |
| **session_count** | 当前 Adjust SDK 记录的第 x 个会话 | 仅服务端事件不生效 |
| **lifetime_session_count** | 用户生命周期记录的第 x 个会话 | 所有都生效 | 
| **last_time_spent** | 上次会话时长（秒）| 仅 session 事件生效 |
| **time_spent** | 用户当前的会话时长（秒）| 仅客户端事件生效（不包含 session 事件） | 

{{< notice warning >}}
坑：**安装**和**再归因**，Dashboard 计 session，但原始数据未计。因此处理导出的原始数据时，需手动将其计入 session；
{{< /notice >}}

### 数据处理

#### 计算平均停留时长


#### 计算留存率


