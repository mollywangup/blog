---
title: "Fix ERR_TOO_MANY_REDIRECTS"
date: 2023-06-29T03:55:45Z
draft: false
description: 
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

当打开某个网址时，浏览器提示：`ERR_TOO_MANY_REDIRECTS`

## 解决方案

如果有接Cloudflare，可以尝试修改`SSL/TLS encryption mode`设置为`Full (strict)`

{{< alert theme="info" >}}
[Cloudflare - Available encryption modes](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/#available-encryption-modes)
{{< /alert >}}

> <a href="https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/#available-encryption-modes" target="_blank">Cloudflare - Available encryption modes</a>

<img src='/static/images/posts/cloudflare_ssltls_encryption_mode.png' alt='SSL/TLS encryption mode'>