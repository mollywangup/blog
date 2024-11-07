---
title: "搭建属于你自己的 VPN 服务器"
date: 2021-09-15T16:29:59Z
draft: false
description: 共包含三种主流的连接协议：IKEv2/OpenVPN/WireGuard.
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- VPN
categories:
- Tool
---

共包含三种主流的连接协议：IKEv2/OpenVPN/WireGuard. 
无论使用哪种连接协议，核心都是 **在服务端生成配置 -> 在客户端导入配置**；

## 准备工作

- 一个服务器；（本文以 Ubuntu/Debian 为例）
- 一点点 Linux 知识；

## IKEv2

### Server-side

> [IPsec VPN Server Auto Setup Scripts](https://github.com/hwdsl2/setup-ipsec-vpn "hwdsl2 IKEv2")

（建议）先更新软件包和系统：

```shell
sudo apt-get update && sudo apt-get dist-upgrade
```

安装 IKEv2 VPN：（👉 密码纠结症指路 <a href="https://mollywangup.com/posts/generate-a-secure-password/" target="_blank">快速生成一个安全的随机密码</a>）

```shell
wget https://get.vpnsetup.net -O vpn.sh
sudo VPN_IPSEC_PSK='your_ipsec_pre_shared_key' \
VPN_USER='your_vpn_username' \
VPN_PASSWORD='your_vpn_password' \
sh vpn.sh
```

获取客户端配置文件：

```shell
sudo ikev2.sh --listclients
sudo ikev2.sh --addclient [client name]
sudo ikev2.sh --exportclient [client name]
```

{{< expand "例子：导出名称为 vpnclient 的客户端配置" >}}

##### 导出

```shell
sudo ikev2.sh --exportclient vpnclient
```

##### 结果如下

```plaintext
================================================

IKEv2 client "vpnclient" exported!

VPN server address: xxx.xx.xx.xx

Client configuration is available at:
/path/vpnclient.p12 (for Windows & Linux)
/path/vpnclient.sswan (for Android)
/path/vpnclient.mobileconfig (for iOS & macOS)

Next steps: Configure IKEv2 clients. See:
https://vpnsetup.net/clients

================================================
```

{{< /expand >}}

### Client-side

根据不同客户端类型，导出对应后缀的配置文件，然后傻瓜式操作即可。
注意：一般是在系统层级的网络中配置。

## OpenVPN

### Server-side

> [openvpn-install](https://github.com/Nyr/openvpn-install#openvpn-install)

安装 OpenVPN VPN：

```shell
wget https://git.io/vpn -O openvpn-install.sh && bash openvpn-install.sh
```

### Client-side

根据不同客户端类型，下载对应的[客户端应用程序](https://openvpn.net/client/)，然后傻瓜式操作即可。

## WireGuard

### Server-side

> [wireguard-install](https://github.com/Nyr/wireguard-install#installation)

安装 WireGuard VPN：

```shell
wget https://git.io/wireguard -O wireguard-install.sh && bash wireguard-install.sh
```

### Client-side

根据不同客户端类型，下载对应的[客户端应用程序](https://www.wireguard.com/install/)，然后傻瓜式操作即可。

{{< expand "macOS 无法从 App Store 下载时的替代方案" >}}

##### [由此下载](https://github.com/aequitas/macos-menubar-wireguard/releases)

安装 CLI tool for WireGuard：

```shell
brew install wireguard-tools
```

配置 wg0.conf 文件：
```shell
cd /etc/wireguard
vim wg0.conf
```
{{< /expand >}}