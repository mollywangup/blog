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
CREATE TABLE IF NOT EXISTS `table_name`(
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

#### DELETE

```sql
DELETE FROM <table_name> WHERE conditions;
```

#### INSERT

```sql
INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)
```

#### ALTER

```sql
-- 新增字段
ALTER TABLE <table_name> ADD `new_int_col` INT DEFAULT 0;

-- 修改字段定义
ALTER TABLE <table_name> MODIFY `old_str_col` VARCHAR(20);

-- 删除 UNIQUE KEY 后并新增
SHOW KEYS FROM <table_name>;
ALTER TABLE <table_name> DROP INDEX `app_name`;
ALTER TABLE <table_name> ADD UNIQUE KEY (`app_name`, `os_name`, `store_type`,  `channel_id`,`login_type_code`, `date_paying`, `country`,  `timezone`);

-- 删除字段
ALTER TABLE <table_name> DROP `days_x`;

-- 新增并来源于已有字段的处理
ALTER TABLE <table_name> ADD `event_time_hour` INT;

-- 日期处理
ALTER TABLE <table_name> ADD `update_time_utc` DATETIME;
```

#### UPDATE

```sql
UPDATE <table_name> SET event_time_hour = (SELECT HOUR(`Event Time`));
UPDATE <table_name> SET minute_x = (SELECT TIMESTAMPDIFF(MINUTE, first_open_time, event_timestamp));
UPDATE <table_name> SET `update_time_utc` = (SELECT DATE_ADD(update_time, INTERVAL -8 hour));
```

## 查询操作

### 查询结构

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

### 表连接

#### JOIN

```sql
-- 方式一                             explicit join
FROM table_1 t1
type_of_join table_2 t2
  ON t2.key = t1.key
  -- USING (key)

-- 方式二                             implicit join
FROM table_1 t1, table_2 t2
WHERE t2.key = t1.key 
```

其中，常见 `type_of_join` 如下：
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL JOIN
- CROSS JOIN（笛卡尔连接，交叉连接）

#### UNION

要求合并的两张表，至少有一个相同列，且自动去重。

```sql
SELECT
  col_1
FROM table_1
UNION
SELECT
  col_1
FROM table_2;
```

### 聚合

```sql
-- 普通聚合
SELECT
  col_1,
  agg_function(col_2)
FROM table
GROUP BY col_1;

-- 包含小计和总计的聚合                   ROLLUP
SELECT
  col_1,
  col_2,
  agg_function(col_3)
FROM table
GROUP BY col_1, col_2 WITH ROLLUP;
-- GROUP BY ROLLUP(col_1, col_2);
```

{{< alert theme="warning" >}}
⚠️ 注意：MySQL 中无法使用以下 `GROUPING SETS`：

```sql
GROUP BY
  GROUPING SETS (
    (col_1, col_2),
    (col_1),
    (col_2),
    ()
  )
```
{{< /alert >}}

### 窗口函数

```sql
some_window_function() OVER (
  PARTITION BY some_col
  ORDER BY another_col
)
```

其中，常见窗口函数如下：

| 函数&nbsp;&nbsp;&nbsp;&nbsp; | 用途&nbsp;&nbsp;&nbsp;&nbsp; | 
| --------------- | --------------- |
| ROW_NUMBER() | 排序，如 `1, 2, 3, 4` |
| RANK() | 排序，如 `1, 2, 2, 4` |
| DENSE_RANK() | 排序，如 `1, 2, 2, 3` |
| FIRST_VALUE(col) | 取第一个值 |
| LAST_VALUE(col) | 取最后一个值 |
| NTH_VALUE(col, n) | 取第 n 个值 |
| LAG(col, n) | 取前第 n 个值 |
| LEAD(col, n) | 取后第 n 个值 |
| NTILE(n) | 分成 n 组 |

### WITH AS

```sql
WITH cte_1 AS (
  SELECT ...
),
cte_2 AS (
  SELECT ...
)

SELECT ...
FROM ...

-- 例子
WITH cte AS (
  SELECT 
  salary,
  RANK() OVER (ORDER BY salary DESC) AS rk
  FROM Employee
)

SELECT * FROM cte;
```

## 函数

以下为高频常用函数，完整列表见 **[MySQL Functions](https://www.w3schools.com/sql/sql_ref_mysql.asp)**

### 数值函数

1. 保留小数：
   - `ROUND(number, decimals)`：四舍五入
   - `TRUNCATE(number, decimals)`：直接截取
   - `CEILING(number)`：向上取整，即 MIN({>=number})
   - `FLOOR(number)`：向下取整，即 MAX({<=number})

2. 统计计算：
   - `MIN(expression)`：求最小值
   - `MAX(expression)`：求最大值
   - `AVG(expression)`：求平均值
   - `SUM(expression)`：求和
   - `COUNT(expression)`：求次数

3. 数学计算：
   - `MOD(x, y)`：求余
     - `x MOD y`
     - `x % y`
   - `SQRT(number)`：求平方根
   - `POWER(x, y)`：求 x 的 y 幂次方

测试例子：

```sql
SELECT ROUND(3.1456, 2), TRUNCATE(3.1456, 2), CEILING(3.1456), FLOOR(3.1456);

SELECT MOD(3, 2), SQRT(16), POWER(8, 2);
```

### 字符串函数

```sql

```

## 数据库备份

### 备份

```shell
mysqldump -uroot -p<password> --log-error=/path/xxx.err -B <database_name> > /path/xxx.sql
```

### 恢复

```shell
# 如果是.zip格式需先解压，解压后后缀为.sql
# 恢复整个数据库
mysql -uroot -p<password> <database_name> < /path/xxx.sql
```

## 其他

### 查看Host
```sql
SELECT SUBSTRING_INDEX(host,':',1) AS ip , COUNT(*) FROM information_schema.processlist GROUP BY ip;
```

### 查看Port

```sql
SHOW VARIABLES WHERE Variable_name = 'port';
```

### 查看用户

```sql
USE mysql;
SELECT host, user, authentication_string, plugin FROM user;
```