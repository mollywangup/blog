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

以下为常用函数，完整列表见 **[MySQL Functions](https://www.w3schools.com/sql/sql_ref_mysql.asp)**

### 数字函数

#### 常用函数

1. 保留小数：
   - `ROUND(number, decimals)`：四舍五入
   - `TRUNCATE(number, decimals)`：直接截取
   - `CEILING(number)`：向上取整，即 MIN({>=number})
   - `FLOOR(number)`：向下取整，即 MAX({<=number})

2. 数学运算：
   - `MOD(x, y)`：求余
     - or `x MOD y`
     - or `x % y`
   - `SQRT(number)`：求平方根
   - `POWER(x, y)`：求 x 的 y 幂次方

#### 测试例子

```sql
SELECT ROUND(3.1456, 2), TRUNCATE(3.1456, 2), CEILING(3.1456), FLOOR(3.1456);
SELECT MOD(3, 2), SQRT(16), POWER(8, 2);
```

### 字符串函数

#### 常用函数

- 高频使用：
  - `LENGTH(str)`：求长度
  - `UPPER(str)`：转大写
  - `LOWER(str)`：转小写
  - `REPLACE(str, from_str, to_str)`：替换
  - `CONCAT(str1, str2, ...)`：拼接

- 左右处理：
  - `LTRIM(str)`：删左/头部空格
  - `RTRIM(str)`：删右/尾部空格
  - `TRIM(str)`：删左右空格
  - `LPAD(str, len, padstr)`：左填充，以达到指定长度
  - `RPAD(str, len, padstr)`：右填充，以达到指定长度

- 字符串提取：
  - `LEFT(str, len)`：自左边取
  - `RIGHT(str, len)`：自右边取
  - `MID(str, pos, len)`：自指定位置取
    - or `SUBSTR(str, pos, len)`
    - or `SUBSTRING(str, pos, len)`

- 其他
  - `LOCATE(substr, str)`：子字符串第一次出现的位置。不区分大小写，未找到时返回0
    - or `POSITION(substr IN str)`
  - `REPEAT(str, count)`：重复字符串指定次数
  - `REVERSE(str)`：反转字符串

#### 测试例子

```sql
SELECT CONCAT('first_name', ' ', 'last_name');

SELECT LPAD('molly', 10, '_'), RPAD('molly', 10, '_');

SELECT LOCATE('com', 'google.com'), POSITION("COM" IN 'google.com');
SELECT REPEAT('MySQL', 3);
```

### 日期函数

#### 常用函数

1. 获取当下日期时间
   - `NOW()`：返回当前日期和时间
   - `CURDATE()`：返回当前日期
     - or `CURRENT_DATE()`
   - `CURTIME()`：返回当前时间
     - or `CURRENT_TIME()`

2. 提取年月日时分秒
   - `EXTRACT(unit FROM date)`：通用的提取函数
     - unit 可取值 YEAR/MONTH/DAY 等，详见 [Temporal Intervals
](https://dev.mysql.com/doc/refman/8.0/en/expressions.html#temporal-intervals)
      {{< alert theme="warning" >}}
⚠️ 建议使用 EXTRACT() 函数，因为属于标准 SQL 语言，适配性更高。
      {{< /alert >}}
   - `YEAR(date)`：年份
   - `QUARTER(date)`：季度
   - `MONTH(date)`：月份
   - `DAY(date)`：该月份的天数
   - `HOUR(datetime)`：小时数
   - `MINUTE(datetime)`：分钟数
   - `SECOND(datetime)`：秒数
   - `MONTHNAME(date)`：字符串格式的月份，如 August
   - `DAYNAME(date)`：字符串格式的星期数，如 Thursday

3. 格式化：
   - `DATE_FORMAT(date, format)`：format 取值详见 [MySQL 8.0 Reference Manual](https://dev.mysql.com/doc/refman/8.0/en/date-and-time-functions.html#function_date-format)
   - `CONVERT_TZ(dt, from_tz, to_tz)`：转时区

4. 日期运算（不同 DBMS 相差较大）
   - `DATE_ADD(date, INTERVAL expr unit)`：增加时间间隔。unit 同 EXTRACT() 函数
     - or `DATE_SUB(date,INTERVAL -expr unit)`
   - `DATEDIFF(date1, date2)`：计算相差天数，注意是 *date1 - date2*

#### 测试例子

```sql
SELECT NOW(), CURDATE(), CURRENT_DATE(), CURTIME(), CURRENT_TIME();

SELECT NOW(), YEAR(NOW()), QUARTER(NOW()), MONTH(NOW()), DAY(NOW()), HOUR(NOW()), MINUTE(NOW()), SECOND(NOW()), MONTHNAME(NOW()), DAYNAME(NOW());

SELECT NOW(), EXTRACT(DAY FROM NOW());

SELECT DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s'), DATE_FORMAT(NOW(), '%W %M %d %Y %l:%i:%s %p'), DATE_FORMAT(NOW(), '%a %b %d %Y %l:%i:%s %p'), DATE_FORMAT(NOW(), '%r'), DATE_FORMAT(NOW(), '%T');

SELECT DATE_ADD(NOW(), INTERVAL -1 DAY);
SELECT DATEDIFF('2017-01-01', '2016-12-24');

SELECT TO_DAYS('2017-06-20'), TO_DAYS('2017-06-20 09:34:00'), FROM_DAYS(736865);
```

### 聚合函数

#### 常用函数

- `MAX(expr)`：求最大值
- `MIN(expr)`：求最小值
- `AVG(expr)`：求平均值
- `SUM(expr)`：求和
- `COUNT(expr)`：求次数

### 窗口函数

窗口函数基于**分区和排序后的查询结果的行数据**进行计算。语法如下：

```sql
some_window_function OVER (
  PARTITION BY some_col
  ORDER BY another_col
)

SELECT
  val,
  ROW_NUMBER() OVER w AS 'row_number',
  RANK()       OVER w AS 'rank',
  DENSE_RANK() OVER w AS 'dense_rank'
FROM numbers
WINDOW w AS (ORDER BY val);
```

#### 常用函数

1. 聚合函数：上述 聚合函数 中的都适用；

2. 排序函数：
   - 排名：
     - `ROW_NUMBER()`：返回排名，如 1, 2, 3, 4, ...
     - `RANK()`：返回排名，如 1, 2, 2, 4, ...
     - `DENSE_RANK()`：返回排名，如 1, 2, 2, 3, ...
     - `NTILE(n)`：分成 n 组，返回组别
   - 排名分布：
     - `PERCENT_RANK()`：返回排名的百分比
       - 计算公式：*(rank - 1) / (rows - 1)*
        <!-- <img src='https://www.sqlshack.com/wp-content/uploads/2019/08/sql-percentile-function.png' alt='n = 11'> -->
     - `CUME_DIST()`：返回值累计分布的百分比，如 top 10%
       - 计算公式：*小于或大于等于当前值的行数 / rows*

3. 值函数/偏移函数：
   - `FIRST_VALUE(col)`：取第一行值
   - `LAST_VALUE(col)`：取最后一行值
   - `NTH_VALUE(col, n)`：取第 n 行值
   - `LAG(col, n, defaut)`：取向**前**偏移 n 行的值，若不存在则取 defaut
   - `LEAD(col, n, defaut)`：取向**后**偏移 n 行的值，若不存在则取 defaut

#### 宝藏参考

- MySQL 官方手册：[Window Functions](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html)
- [How to use Window functions in SQL Server](https://www.sqlshack.com/use-window-functions-sql-server/)
- [Overview of SQL RANK functions](https://www.sqlshack.com/overview-of-sql-rank-functions/)
- [Calculate SQL Percentile using the PERCENT_RANK function in SQL Server](https://www.sqlshack.com/calculate-sql-percentile-using-the-sql-server-percent_rank-function/)

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