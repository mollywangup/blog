- [安装MongoDB](#安装mongodb)
  - [](#)
  - [安装MongoDB Compass](#安装mongodb-compass)
- [查询](#查询)
  - [创建及查看数据库](#创建及查看数据库)
  - [创建及查看集合](#创建及查看集合)
  - [创建新用户](#创建新用户)



```shell
pip install google-cloud-bigquery

```





# 安装MongoDB
## 
[Install MongoDB Community Edition on macOS](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/#install-mongodb-community-edition-on-macos)

在 macOS 上创建本地 MongoDB 数据库可以按照以下步骤进行：

1. 打开终端，并执行以下命令安装 Homebrew 包管理器：

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. 安装完成后，执行以下命令安装 MongoDB：

```
brew tap mongodb/brew
brew update
brew install mongodb-community@6.0
```

对应的默认：
configure file: `/usr/local/etc/mongod.conf`
log directory: `/usr/local/var/log/mongodb`
data directory: `/usr/local/var/mongodb`


3. 启动/关闭/查看 MongoDB 服务：

```
# as a macOS service
brew services start mongodb-community@6.0
brew services stop mongodb-community@6.0
brew services list

# manually as a background process
mongod --config /usr/local/etc/mongod.conf
ps aux | grep -v grep | grep mongod
```



4. 在控制台中连接到 MongoDB 实例：

```
mongosh
```
开启/关闭 MongoDB 的性能监测服务：
```
db.enableFreeMonitoring()
db.disableFreeMonitoring()
{
  state: 'enabled',
  message: 'To see your monitoring data, navigate to the unique URL below. Anyone you share the URL with will also be able to view this page. You can disable monitoring at any time by running db.disableFreeMonitoring().',
  url: 'https://cloud.mongodb.com/freemonitoring/cluster/Z6HD6MJL3EVWYYCBJT6OU4MUFG5OKXD4',
  userReminder: '',
  ok: 1
}
```

## 安装MongoDB Compass
[MongoDB Compass Download](https://www.mongodb.com/try/download/compass)


# 查询

## 创建及查看数据库
1. 有则切换无则直接创建；
2. 此时数据库为空，默认存储在内存中，只有有数据了才会持久化到硬盘中；

```shell
use fengche
db
```

## 创建及查看集合
```shell
db.createCollection('test')
show collections
```

## 创建新用户
```shell
db.createUser({ user: looker,
                pwd: `some_password_here`,
                roles: [ "readWrite" ]
              })
```
