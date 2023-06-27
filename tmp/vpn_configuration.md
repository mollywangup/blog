- [IKEv2](#ikev2)
- [OpenVPN](#openvpn)
- [WireGuard](#wireguard)


**Note**: There is two steps to do, Configure VPN clients, then Export VPN configuration for clients.

# IKEv2
[IPsec VPN Server Auto Setup Scripts](https://github.com/hwdsl2/setup-ipsec-vpn "hwdsl2 IKEv2")

**Step1**: update your server with `sudo apt-get update && sudo apt-get dist-upgrade` (Ubuntu/Debian) or `sudo yum update `and reboot. This is optional, but recommended.

**Step2**: install the VPN.
```shell
wget https://get.vpnsetup.net -O vpn.sh
sudo VPN_IPSEC_PSK='your_ipsec_pre_shared_key' \
VPN_USER='your_vpn_username' \
VPN_PASSWORD='your_vpn_password' \
sh vpn.sh
```

**Step3**: manage IKEv2 clients.
```shell
sudo ikev2.sh --listclients
sudo ikev2.sh --addclient [client name]
sudo ikev2.sh --exportclient [client name]
```

# [OpenVPN](https://openvpn.net/)

**Step1**: Server
```shell
wget https://git.io/vpn -O openvpn-install.sh && bash openvpn-install.sh
```

**Step2**: Client
- macOS/iOS/Android/Windows
- iOS
- Android
- Windows




# [WireGuard](https://www.wireguard.com/)
[wireguard-install](https://github.com/Nyr/wireguard-install "Nyr WireGuard")

**Step1**: Server 
```shell
wget https://git.io/wireguard -O wireguard-install.sh && bash wireguard-install.sh
```

**Step2**: Client
- macOS

**option1**: Download from App Store
See https://www.wireguard.com/install/

**option2**: Can not Download from App Store
See https://github.com/aequitas/macos-menubar-wireguard/releases
First, install CLI tool for WireGuard:
```shell
brew install wireguard-tools
```
Second, copy the configuration to the `wg0.conf`:
```shell
cd /etc/wireguard
vim wg0.conf
```



Error
`Error: Permission denied @ apply2files - /usr/local/lib/docker/cli-plugins`
> https://stackoverflow.com/questions/72784094/homebrew-error-permission-denied-apply2files-usr-local-lib-docker-cli-pl



```shell
scp [server username]@[VPN server address]:[Client configuration path] [Local path]
```

[^1]: [openvpn-install](https://github.com/Nyr/openvpn-install)


