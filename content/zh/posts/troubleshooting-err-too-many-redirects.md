---
title: "踩坑：ERR_TOO_MANY_REDIRECTS"
date: 2021-06-29T03:55:45Z
draft: false
description: 自建的网站，当添加 Cloudflare 后，忽然报错 ERR_TOO_MANY_REDIRECTS. 
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- Cloudflare
categories:
- Troubleshooting
---

## 问题描述

当打开自己的网站时，浏览器报错：`ERR_TOO_MANY_REDIRECTS`

## 问题定位

这个报错通常是由于重定向死循环导致，如下图: 

<img src='/images/posts/redirect_loop.png' alt='A Redirect Loop'>

## 解决思路

检查网站的重定向设置，无论是主动设置的还是被动设置的。

### 思路一：排除 Cloudflare 设置

如果有接 Cloudflare, 可以尝试将 `SSL/TLS encryption mode` 修改为 `Full (strict)`

> Cloudflare 官方参考文档：<a href="https://developers.cloudflare.com/ssl/troubleshooting/too-many-redirects/#err_too_many_redirects" target="_blank">ERR_TOO_MANY_REDIRECTS</a>

<img src='/images/posts/cloudflare_ssltls_encryption_mode.png' alt='SSL/TLS encryption mode'>
