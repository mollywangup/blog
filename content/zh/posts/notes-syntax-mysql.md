---
title: "MySQL 基本语法"
date: 2021-01-28T06:48:47Z
draft: false
description: CURD, 用户及权限管理，常用数据库和数据表操作等。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- MySQL
- SQL
- OLTP
categories:
- DB
---

✍ 本文作为学习笔记。

## 初始化

### 首次仅登录

```shell
mysql -uroot
```

### 首次登录并设置密码

```shell
mysqladmin -u root -p password <password>
```

### 设置/修改密码

```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY <password>;
```

### 数据库连接

#### 普通登录

```shell
mysql -u root  -p
mysql -h <host> -P <port> -u <username> -p 
```

#### SSH登录

和普通登录一个道理，只是需要提前登录SSH;

```shell
ssh <sshuser>@<sshhost>
mysql -h <host> -u <username> -p
```

#### URI连接

```
mysql://<username>:<password>@<host>:<port>/<database_name>
```

## 权限管理

### 新增用户

以下以用户`wangli`为例：

```sql
CREATE USER 'wangli'@'%' IDENTIFIED WITH mysql_native_password BY '12345678';
```

### 查看权限

```sql
SHOW GRANTS FOR root@localhost;
SHOW GRANTS FOR admin;
```

### 修改权限

```sql
-- 授权所有权限
GRANT ALL PRIVILEGES ON *.* TO 'wangli'@'%';
FLUSH PRIVILEGES;

-- 移除所有权限
REVOKE ALL PRIVILEGES ON *.* FROM 'wangli'@'%';
FLUSH PRIVILEGES;

-- 授权指定数据库
GRANT ALL PRIVILEGES ON <database_name>.* TO 'admin'@'%';
FLUSH PRIVILEGES;
```

## 数据库操作

### 创建数据库

```sql
CREATE DATABASE <database_name>;
```

### 删除数据库
```sql
DROP DATABASE <database_name>;
```

### 显示数据库
```sql
SHOW DATABASES;
```

### 切换数据库
```sql
USE <database_name>;
```

## 数据表操作

### 创建数据表

```sql
-- 复制已有数据库表
CREATE TABLE `new_table` AS (SELECT * FROM `old_table`);

-- 直接创建数据库表 
CREATE TABLE IF NOT EXISTS `mytable`(
   `id` INT UNSIGNED AUTO_INCREMENT,
   `account_id` VARCHAR(25) NOT NULL COMMENT '广告账户ID',
   `account_name` VARCHAR(50),
   `media_source` VARCHAR(25) NOT NULL,
   `data_source` VARCHAR(25) NOT NULL,
   `created_time` TIMESTAMP NOT NULL,
   `account_status` VARCHAR(10),
   `disable_reason` VARCHAR(30),
   `currency` VARCHAR(3),
   `spend_cap` FLOAT,
   `amount_spent` FLOAT,
   `amount_remain` FLOAT,
   `updated_time` TIMESTAMP NOT NULL,
   `timezone`  VARCHAR(33),
   PRIMARY KEY ( `id`),
   UNIQUE KEY ( `account_id` )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
```

### 删除数据表

```sql
DROP TABLE <table_name>;
```

### 显示数据表

```sql
SHOW TABLES;
```

### 修改数据表

#### DELETE语句

```sql
DELETE FROM <table_name> WHERE `date` BETWEEN '2021-02-01' AND '2021-02-01';
```

#### INSERT语句

```sql
INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)
```

#### ALTER语句

```sql
-- 新增字段
ALTER TABLE <table_name> ADD `new_int_col` INT DEFAULT 0;

-- 修改字段定义
ALTER TABLE <table_name> MODIFY `optimizer` VARCHAR(20);

-- 删除UNIQUE KEY后并新增
SHOW KEYS FROM <table_name>;
ALTER TABLE <table_name> DROP INDEX `app_name`;
ALTER TABLE <table_name> ADD UNIQUE KEY (`app_name`, `os_name`, `store_type`,  `channel_id`, `channel`, `login_type_code`,  `login_type`, `date_paying`, `country`,  `timezone`);

-- 删除字段
ALTER TABLE <table_name> DROP `days_after_register`;

-- 新增并来源于已有字段的处理
ALTER TABLE <table_name> ADD `event_time_hour` INT;

-- 日期处理
ALTER TABLE <table_name> ADD `update_time_utc` DATETIME;
```

#### UPDATE语句

```sql
UPDATE <table_name> SET event_time_hour = (SELECT HOUR(`Event Time`));
UPDATE <table_name> SET minute_x = (SELECT TIMESTAMPDIFF(MINUTE, first_open_time, event_timestamp));
UPDATE <table_name> SET `update_time_utc` = (SELECT DATE_ADD(update_time, INTERVAL -8 hour));
```

## 常用查询

### 整体常用

```sql
-- Select columns                    
SELECT
    col_1,
    col_2,
     ... ,
    col_n

-- Source of data                    
FROM table t

-- Gather info from other sources    optional
JOIN other_table ot
  ON (t.key = ot.key)

-- Conditions                        optional
WHERE some_condition(s)

-- Aggregating                       optional
GROUP BY col_group_list

-- Restricting aggregated values     optional
HAVING some_condition(s)

-- Sorting values                    optional
ORDER BY col_order_list

-- Limiting number of rows           optional
LIMIT some_value
```

### JOIN

```sql
FROM table_1 t1
type_of_join table_2 t2
  ON (t2.key = t1.key)
```

| Type of join | Illustration | 
| ---------- | --------- | 
| INNER JOIN | <img src='https://www.mit.edu/~amidi/teaching/data-science-tools/illustrations/join-sql/003.png?f1ac039e0897d82dd87ddb134d3acca2'> |
| LEFT JOIN | <img src='https://www.mit.edu/~amidi/teaching/data-science-tools/illustrations/join-sql/002.png?59960a43a2bcff0bb51fe2daf608602e'> |
| RIGHT JOIN | <img src='https://www.mit.edu/~amidi/teaching/data-science-tools/illustrations/join-sql/003.png?f1ac039e0897d82dd87ddb134d3acca2'> |
| FULL JOIN | <img src='https://www.mit.edu/~amidi/teaching/data-science-tools/illustrations/join-sql/004.png?5a9a038972fdd9cf0d3beccf03f02db9'> |

## 其他

### 备份数据库

#### 备份

```sql
mysqldump -uroot -p<password> --log-error=/path/xxx.err -B <database_name> > /path/xxx.sql
```

#### 恢复

```sql
-- 如果是.zip格式需先解压，解压后后缀为.sql
-- 恢复整个数据库
mysql -uroot -p<password> <database_name> < /path/xxx.sql

```

### 报错解决

#### sudo: netstat: command not found

> [[Fixed] Bash: Netstat: Command Not Found](https://www.linuxandubuntu.com/home/fixed-bash-netstat-command-not-found-error)

```shell
# Ubuntu
sudo apt install net-tools
```

### Linux MySQL 配置路径

```shell
cat /etc/mysql/mysql.conf.d/mysqld.cnf
```

### 服务重启

```shell
service mysql restart
```

### 其他查看

#### 查看Host
```sql
SELECT SUBSTRING_INDEX(host,':',1) AS ip , COUNT(*) FROM information_schema.processlist GROUP BY ip;
```

#### 查看Port

```sql
SHOW VARIABLES WHERE Variable_name = 'port';
```

#### 查看用户

```sql
USE mysql;
SELECT host, user, authentication_string, plugin FROM user;
```