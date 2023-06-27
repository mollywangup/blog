- [基础语法](#基础语法)
  - [新建或修改密码](#新建或修改密码)
    - [首次登录并设置密码](#首次登录并设置密码)
    - [没密码时改密码](#没密码时改密码)
    - [修改密码](#修改密码)
  - [登录](#登录)
    - [数据库信息](#数据库信息)
    - [数据库登录](#数据库登录)
      - [普通登录](#普通登录)
      - [SSH登录](#ssh登录)
      - [Python连接](#python连接)
        - [直接连接](#直接连接)
        - [SSH连接](#ssh连接)
  - [新增用户及修改权限](#新增用户及修改权限)
    - [新增用户](#新增用户)
    - [查看权限](#查看权限)
    - [修改权限](#修改权限)
    - [查看用户](#查看用户)
  - [数据库操作](#数据库操作)
    - [创建数据库](#创建数据库)
    - [删除数据库](#删除数据库)
    - [显示数据库](#显示数据库)
    - [选择数据库](#选择数据库)
  - [数据表操作](#数据表操作)
    - [创建数据表](#创建数据表)
      - [添加外键](#添加外键)
      - [添加唯一约束](#添加唯一约束)
    - [删除数据表](#删除数据表)
    - [显示数据表](#显示数据表)
    - [修改数据表](#修改数据表)
        - [DELETE语句](#delete语句)
      - [INSERT语句](#insert语句)
      - [ALTER语句](#alter语句)
      - [UPDATE语句](#update语句)
    - [查询数据表](#查询数据表)
      - [](#)
- [实操常用](#实操常用)
  - [Linux配置](#linux配置)
  - [数据库的备份与恢复](#数据库的备份与恢复)
    - [备份](#备份)
    - [恢复](#恢复)
  - [报错解决](#报错解决)
    - [sudo: netstat: command not found](#sudo-netstat-command-not-found)
  - [重启MySQL服务](#重启mysql服务)
  - [查看Host和Port](#查看host和port)


# 基础语法
## 新建或修改密码
### 首次登录并设置密码
```shell
# 首次登录
mysql -uroot
```

```mysql
# 设置密码
ALTER USER root@localhost IDENTIFIED WITH mysql_native_password BY 'YOUR_PASSWORD';
```

### 没密码时改密码
```shell
mysqladmin -u root -p password YOUR_PASSWORD
```

### 修改密码
```mysql
ALTER USER 'YOUR_USERNAME'@'YOUR_HOSTNAME' IDENTIFIED WITH mysql_native_password BY 'YOUR_PASSWORD';
```

## 登录
### 数据库信息
```shell
Username=YOUR_USERNAME
Password=YOUR_PASSWORD
Host=YOUR_HOSTNAME
Port=3306
Database=YOUR_DATABASE_NAME
```

### 数据库登录
#### 普通登录
```shell
mysql -u root -p
mysql -h YOUR_HOSTNAME -P YOUR_PORT -u YOUR_USERNAME -p
mysql -h YOUR_HOSTNAME -u YOUR_USERNAME -p
```

#### SSH登录
```shell
# 本质上和普通登录一样，只是需要先连接ssh远程
ssh YOUR_SSH_USERNAME@YOUR_SSH_HOSTNAME
mysql -h YOUR_HOSTNAME -u YOUR_USERNAME -p
```

#### Python连接
##### 直接连接
```python
from sqlalchemy import create_engine

engine = create_engine(
  "mysql+pymysql://{}:{}@{}:{}/{}".format(
    'YOUR_USERNAME', 
    'YOUR_PASSWORD', 
    'YOUR_HOSTNAME', 
    'YOUR_PORT',
    'YOUR_DATABASE_NAME',
  )
)

query = '''SELECT COUNT(*) FROM YOUR_TABLE_NAME;'''

with engine.begin() as connection:
    connection.execute(query)
```

##### SSH连接
```python
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import paramiko

private_key = paramiko.RSAKey.from_private_key_file(filename='YOUR_KEY_PATH', password='YOUR_KEY_PASSWORD')

with SSHTunnelForwarder(
  ssh_address_or_host=('YOUR_SSH_HOSTNAME', 22),
  ssh_private_key=private_key,
  ssh_username='YOUR_SSH_USERNAME',
  remote_bind_address=('YOUR_HOSTNAME', 3306),
  local_bind_address=('127.0.0.1', 13306)
) as server:
  engine = create_engine(
    "mysql+pymysql://{}:{}@{}:{}/{}".format(
      'YOUR_USERNAME', 
      'YOUR_PASSWORD', 
      'YOUR_HOSTNAME', 
      'YOUR_PORT',
      'YOUR_DATABASE_NAME',
    )
  )
    
  query = '''SELECT COUNT(*) FROM YOUR_TABLE_NAME;'''

  with engine.begin() as connection:
      connection.execute(query)
```

## 新增用户及修改权限
### 新增用户
其中`'YOUR_NEW_USERNAME'@'%'`表示可连远程
```mysql
CREATE USER 'YOUR_NEW_USERNAME'@'%' IDENTIFIED WITH mysql_native_password BY 'YOUR_PASSWORD';
```

### 查看权限
```mysql
SHOW GRANTS FOR root@localhost;
SHOW GRANTS FOR admin;
```

### 修改权限
```mysql
# 授权所有权限
GRANT ALL PRIVILEGES ON *.* TO 'YOUR_NEW_USERNAME'@'%';
FLUSH PRIVILEGES;

# 移除所有权限
REVOKE ALL PRIVILEGES ON *.* FROM 'YOUR_NEW_USERNAME'@'%';
FLUSH PRIVILEGES;

# 授权指定数据库
GRANT ALL PRIVILEGES ON YOUR_DATABASE_NAME.* TO 'YOUR_NEW_USERNAME'@'%';
FLUSH PRIVILEGES;

# 授权指定数据表
GRANT ALL PRIVILEGES ON YOUR_DATABASE_NAME.YOUR_TABLE_NAME TO 'YOUR_NEW_USERNAME'@'%';
FLUSH PRIVILEGES;
```

### 查看用户
```mysql
USE mysql;
SELECT host, user, authentication_string, plugin FROM user;
```


## 数据库操作
### 创建数据库
```mysql
CREATE DATABASE YOUR_DATABASE_NAME;
```

### 删除数据库
```mysql
DROP DATABASE YOUR_DATABASE_NAME;
```

### 显示数据库
```mysql
SHOW DATABASES;
```

### 选择数据库
```mysql
USE YOUR_DATABASE_NAME;
```

## 数据表操作
### 创建数据表

```mysql
# 复制数据库表
CREATE TABLE `YOUR_TABLE_NAME_COPY` AS (SELECT * FROM `YOUR_TABLE_NAME`);

# table例子: sources_adplatform
CREATE TABLE IF NOT EXISTS `sources_adplatform` (
   `id` INT UNSIGNED AUTO_INCREMENT,
   `app_name` VARCHAR(16) NOT NULL,
   `os_name` VARCHAR(8) NOT NULL,
   `media_source` VARCHAR(32) NOT NULL,
   `account_id` VARCHAR(32) NOT NULL,
   `account_name` VARCHAR(64) NOT NULL,
   `campaign_id` VARCHAR(32) NOT NULL,
   `campaign_name` VARCHAR(255) NOT NULL,
   `date` DATE NOT NULL,
   `country` VARCHAR(8) NOT NULL,
   `impressions` INT DEFAULT 0,
   `clicks` INT DEFAULT 0,
   `install_adplatform` INT DEFAULT 0,
   `event_network_accept` INT DEFAULT 0,
   `cost` FLOAT DEFAULT 0.00,
   `purchase_value` FLOAT DEFAULT 0.00,
   `purchase` INT DEFAULT 0,
   `purchase_unique` INT DEFAULT 0,
   `optimizer` VARCHAR(32) NOT NULL,
   `data_source` VARCHAR(32) NOT NULL,
   `currency` VARCHAR(3) NOT NULL,
   `is_organic` VARCHAR(16) NOT NULL,
   `attribution_setting` VARCHAR(32) NOT NULL DEFAULT 'ignore_no_use',
   PRIMARY KEY ( `id` ),
   UNIQUE KEY ( `account_id`, `campaign_id`, `date`, `country`, `data_source`, `attribution_setting`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
```

#### 添加外键
```mysql
CONSTRAINT <索引名> FOREIGN KEY (<列名>)  REFERENCES <主表名>(<列名>)
CONSTRAINT fk_ FOREIGN KEY () REFERENCES ()
```

#### 添加唯一约束
```mysql
UNIQUE KEY `resource_name` (`resource_name`,`resource_type`)
```

### 删除数据表
```mysql
DROP TABLE YOUR_TABLE_NAME;
```

### 显示数据表
```mysql
SHOW TABLES;
```

### 修改数据表
##### DELETE语句
```mysql
DELETE FROM `YOUR_TABLE_NAME` WHERE `date` BETWEEN '2021-02-01' AND '2021-02-01';
```

#### INSERT语句
```mysql
INSERT INTO `YOUR_TABLE_NAME` (`email`, `password`) VALUES (%s, %s);
```

#### ALTER语句
```mysql
# 新增字段
ALTER TABLE `YOUR_TABLE_NAME` ADD `geographic` VARCHAR(50) NOT NULL DEFAULT 'ignore_no_use';

# 修改字段定义
ALTER TABLE `YOUR_TABLE_NAME` MODIFY `optimizer` VARCHAR(20);

# 删除UNIQUE KEY后并新增
SHOW KEYS FROM `YOUR_TABLE_NAME`;
ALTER TABLE `YOUR_TABLE_NAME` DROP INDEX `app_name`;
ALTER TABLE `YOUR_TABLE_NAME` ADD UNIQUE KEY (`app_name`, `os_name`, `store_type`,  `channel_id`, `channel`, `login_type_code`,  `login_type`, `date_paying`, `country`,  `timezone`);

# 删除字段
ALTER TABLE `YOUR_TABLE_NAME` DROP days_after_register;

# 新增并来源于已有字段的处理
ALTER TABLE `YOUR_TABLE_NAME` ADD `event_time_hour` INT;
UPDATE `YOUR_TABLE_NAME` SET event_time_hour = (SELECT HOUR(`Event Time`));

ALTER TABLE `YOUR_TABLE_NAME` ADD `minute_x` INT NOT NULL DEFAULT -777;
UPDATE `YOUR_TABLE_NAME` SET minute_x = (SELECT TIMESTAMPDIFF(MINUTE, first_open_time, event_timestamp));

# 日期处理
ALTER TABLE `YOUR_TABLE_NAME` ADD `update_time_utc` DATETIME;
UPDATE `YOUR_TABLE_NAME` SET `update_time_utc` = (SELECT DATE_ADD(update_time, INTERVAL -8 hour));

# REGEXP_SUBSTR、REGEXP_REPLACE
ALTER TABLE `YOUR_TABLE_NAME` ADD `pid_int` VARCHAR(64);
UPDATE `YOUR_TABLE_NAME` SET `pid_int` = (SELECT REGEXP_SUBSTR(`ClickUrl`,'pid=.*_int'));

ALTER TABLE `YOUR_TABLE_NAME` ADD `pid_int` VARCHAR(64);
UPDATE `YOUR_TABLE_NAME` SET `pid_int` = (SELECT offer.pid_int FROM offer WHERE callback.offer_id=offer.ID);
```

#### UPDATE语句
```mysql
UPDATE sources_adplatform 
SET os_name = 'android' 
WHERE media_source = 'google_ads'
AND app_name = 'kkchat'
AND optimizer = 'wy_汪悦'
AND os_name = 'unknown'
AND store_type = 'unknown'
AND campaign_name IN ('KKlive_GCC/en_wy_1109-yhn-TP_vidonly', 'KKlive_GCC/en_wy_1109-yhn-in_vidonly', 'KKlive_GLB_wy_1104_vidonly', 'KKlive_GLB_wy_1106-yhn_vidonly', 'KKlive_GLB_wy_1126-yhn_vidonly');
```

### 查询数据表
#### 


# 实操常用

## Linux配置

```shell
cd /etc/mysql

# 错的
cat my.cnf

# 正确的
cd /etc/mysql/mysql.conf.d
cat mysqld.cnf
```

## 数据库的备份与恢复
### 备份
```mysql
mysqldump -uroot -pjywlbj --log-error=./chushou_vchat_dump.err -B chushou_vchat > ./chushou_vchat.sql

mysqldump -uYOUR_USERNAME -pYOUR_PASSWORD --log-error=./YOUR_DATABASE_NAME_dump.err -B YOUR_DATABASE_NAME > ./YOUR_DATABASE_NAME.sql
```
### 恢复
```mysql
# 如果是.zip格式需先解压，解压后后缀为.sql
# 恢复整个数据库
mysql -uYOUR_USERNAME -pYOUR_PASSWORD YOUR_DATABASE_NAME < YOUR_BACKUP_SQL_FILE_PATH
```

## 报错解决
### sudo: netstat: command not found

解决方案：https://www.Linuxandubuntu.com/home/fixed-bash-netstat-command-not-found-error

```shell
# Ubuntu
sudo apt install net-tools
```

## 重启MySQL服务

```shell
service mysql restart
```

## 查看Host和Port
```mysql
# 查看Host
SELECT SUBSTRING_INDEX(host, ':', 1) AS ip, COUNT(*) FROM information_schema.processlist GROUP BY ip;

# 查看Port
SHOW VARIABLES WHERE Variable_name = 'port';
```