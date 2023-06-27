- [方法一：adb command line](#方法一adb-command-line)
  - [安装](#安装)
    - [macOS](#macos)
    - [Windows](#windows)
    - [验证安装](#验证安装)
  - [电脑到手机](#电脑到手机)
  - [手机到电脑](#手机到电脑)
- [方法二：Android File Transfer](#方法二android-file-transfer)
- [方法三：Python的http.server](#方法三python的httpserver)


## 方法一：adb command line
### 安装

#### macOS
```shell
brew install android-platform-tools
```
#### Windows
下载adb：https://blog.csdn.net/x2584179909/article/details/108319973
添加到系统环境变量：https://www.cnblogs.com/handsomefa9527/articles/13030403.html

#### 验证安装
```shell
adb --version
```

### 电脑到手机
```shell
# 查看手机文件目录, 一般download文件夹在
adb shell
ls
cd /sdcard/Download

# adb push <本地路径> <远程路径>
adb push /Users/liwang/GoogleDrive/xxx.apk /sdcard/Download
```

### 手机到电脑
```shell
# 一般截图文件在
cd /sdcard/DCIM/Screenshots

# adb pull <远程路径> <本地路径>
adb pull /sdcard/DCIM/Screenshots/xxx.jpg /Users/liwang
```

## 方法二：Android File Transfer
https://www.android.com/filetransfer/

## 方法三：Python的http.server
```shell
cd /Users/liwang/GoogleDrive/python/fengche/api

source venv/venv-data/bin/activate
python -m http.server
python -m http.server 8000 

# 只要连接再同一WiFi下，无论是电脑或者手机，都可以实现访问对应目录的文件
http://[::]:8000/
```