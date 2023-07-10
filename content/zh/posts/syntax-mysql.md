---
title: "MySQL 基本语法"
date: 2021-01-28T06:48:47Z
draft: false
description: MySQL syntax
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- MySQL
categories:
- DB
---

## 初始化

### 首次登录并设置密码

```shell
# 首次登录
mysql -uroot
```

```sql
-- 设置密码
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY <Password>;
```

### 设置密码

```shell
mysqladmin -u root -p password <Password>
```

### 修改密码

```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY <Password>;
```

## 数据库登录

### 普通登录

```shell
mysql -u root  -p
mysql -h {Host} -P {Port} -u {UserName} -p 
```

### SSH登录

```shell
ssh ec2-user@ec2-18-196-30-101.eu-central-1.compute.amazonaws.com
mysql -h tc-ff-prod-rds-slave.cn2n4ksfkf1t.eu-central-1.rds.amazonaws.com -u wangli -p
```

### Python连接

```python
```

## 用户及权限管理

### 新增用户

`其中'wangli'@'%'表示可连远程`
```mysql
CREATE USER 'wangli'@'%' IDENTIFIED WITH mysql_native_password BY 'jywlbj';
```

### 查看权限

```mysql
SHOW GRANTS FOR root@localhost;
SHOW GRANTS FOR admin;
```

### 修改权限

```sql
# 授权所有权限
GRANT ALL PRIVILEGES ON *.* TO 'wangli'@'%';
FLUSH PRIVILEGES;

# 移除所有权限
REVOKE ALL PRIVILEGES ON *.* FROM 'wangli'@'%';
FLUSH PRIVILEGES;

# 授权指定数据库
GRANT ALL PRIVILEGES ON fengche.* TO 'admin'@'%';
FLUSH PRIVILEGES;
```

### 查看用户

```sql
USE mysql;
SELECT host, user, authentication_string, plugin FROM user;
```

## 数据库操作

### 创建数据库

```sql
create database chushou_vchat_tmp;
```

### 删除数据库
```sql
drop database chushou_vchat_tmp;
```

### 显示数据库
```sql
show databases;
```

### 切换数据库
```sql
use chushou_vchat_tmp;
```

## 数据表操作

### 创建数据表

```sql
# 复制已有数据库表
CREATE TABLE `evest_postback_install` AS (SELECT * from `evest_postback_install3`);

# 直接创建数据库表: account_id_config
CREATE TABLE IF NOT EXISTS `account_id_config `(
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
drop table sources_adjust_tmp;
```

### 显示数据表

```sql
show tables;
```

### 修改数据表

#### DELETE语句

```sql
DELETE FROM `sources_adplatform_temp` WHERE `date` BETWEEN '2021-02-01' AND '2021-02-01';
```

#### INSERT语句

```sql
INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)
```

#### ALTER语句

```sql
# 新增字段
ALTER TABLE `sources_adjust` ADD  `revenue_events_adjust` INT DEFAULT 0;

# 修改字段定义
ALTER TABLE `sources_adplatform_temp` MODIFY `optimizer` VARCHAR(20);

# 删除UNIQUE KEY后并新增
SHOW KEYS FROM `internal_revenue`;
ALTER TABLE `internal_revenue` DROP INDEX `app_name`;
ALTER TABLE `internal_revenue` ADD UNIQUE KEY (`app_name`, `os_name`, `store_type`,  `channel_id`, `channel`, `login_type_code`,  `login_type`, `date_paying`, `country`,  `timezone`);

# 删除字段
ALTER TABLE internal_revenue_cohort DROP days_after_register;

# 新增并来源于已有字段的处理
ALTER TABLE `evest_postback_install3` ADD `event_time_hour` INT;
UPDATE `evest_postback_install3` SET event_time_hour = (SELECT HOUR(`Event Time`));

ALTER TABLE `bq_full` ADD `minute_x` INT NOT NULL DEFAULT -777;
UPDATE `bq_full` SET minute_x = (SELECT TIMESTAMPDIFF(MINUTE, first_open_time, event_timestamp));

# 日期处理
ALTER TABLE `callback` ADD `update_time_utc` DATETIME;
UPDATE `callback` SET `update_time_utc` = (SELECT DATE_ADD(update_time, INTERVAL -8 hour));
```

#### UPDATE语句

```sql
UPDATE `bq_full` SET minute_x = (SELECT TIMESTAMPDIFF(MINUTE, first_open_time, event_timestamp));
UPDATE `offer` SET `pid_int` = (SELECT REGEXP_SUBSTR(`ClickUrl`,'pid=.*_int'));
UPDATE `offer` SET `pid_int` = (SELECT REGEXP_REPLACE(`pid_int`, 'pid=', ''));
UPDATE `callback` SET `offer_id` = (SELECT sendtask.OfferID FROM sendtask WHERE callback.task_id=sendtask.ID);
UPDATE `offer` SET `af_prt` = (SELECT REGEXP_SUBSTR(`ClickUrl`, '\\?af_prt=[\w\sa-zA-Z0-9_]*&|&af_prt=[\w\sa-zA-Z0-9_]*&|&af_prt=.*'));
UPDATE `callback` SET `app_id` = (SELECT offer.app_id FROM offer WHERE callback.offer_id=offer.ID);
```

## 其他操作

#### 查看Host
```sql
SELECT SUBSTRING_INDEX(host,':',1) AS ip , COUNT(*) FROM information_schema.processlist GROUP BY ip;
```

#### 查看Port

```sql
SHOW VARIABLES WHERE Variable_name = 'port';
```