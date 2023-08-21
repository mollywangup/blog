---
title: "MySQL 基本语法"
date: 2021-01-28T06:48:47Z
draft: false
description: 用户及权限管理，常用数据库和数据表操作，窗口函数，表连接等。
hideToc: false
enableToc: true
enableTocContent: false
tocPosition: inner
tags:
- MySQL
- SQL
categories:
- DB
---

✍ 本文作为学习笔记。

## 安装及配置

### 安装

```shell
brew install mysql
```

### 首次登录

#### 方式一：先登录后设置密码

```shell
mysql -u root
```
```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY <password>;
```

#### 方式二：直接登录并设置密码

```shell
mysqladmin -u root -p password <password>
```

### 连接数据库

#### 命令行方式

```shell
ssh <sshuser>@<sshhost>                   # optional
mysql -h <host> -P <port> -u <username> -p
```

#### URI 方式

```plaintext
mysql://<username>:<password>@<host>:<port>/<database_name>
```

### 权限管理

以下以用户`wangli`为例。

#### 新增用户

```sql
CREATE USER 'wangli'@'%' IDENTIFIED WITH mysql_native_password BY '12345678';
```

#### 查看权限

```sql
SHOW GRANTS FOR root@localhost;
SHOW GRANTS FOR admin;
```

#### 修改权限

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

## 数据库

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

### 备份与恢复

#### 备份

```shell
mysqldump -uroot -p<password> --log-error=/path/xxx.err -B <database_name> > /path/xxx.sql
```

#### 恢复

```shell
# 如果是.zip格式需先解压，解压后后缀为.sql
# 恢复整个数据库
mysql -uroot -p<password> <database_name> < /path/xxx.sql
```

## 数据表

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

## 查询

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

共包含 `JOIN`（左右连接）和 `UNION`（上下连接）两种连接方式。

#### JOIN

可显示连接，也可隐式连接。前者更灵活。

```sql
-- 方式一                             explicit join
FROM table_1 t1
type_of_join table_2 t2
  ON t2.key = t1.key

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

<br>{{< alert theme="info" >}}
💡 显示连接中，当连接的两张表的**所有 key** 完全一致时，使用 `USING` 相较于 `ON` 更为简洁。即以下两个表达式具有相同的作用：
- `ON t2.key1 = t1.key1 AND t2.key2 = t1.key2`
- `USING (key1, key2)`
{{< /alert >}}

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

### 分组聚合

#### GROUP BY

```sql
SELECT
  col_1,
  agg_function(col_2)
FROM table
GROUP BY col_1;
```

#### ROLLUP

`ROLLUP` 用于在分组聚合时，包含小计和总计。

```sql
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

### 高级查询

#### 查看 Host
```sql
SELECT SUBSTRING_INDEX(host,':',1) AS ip , COUNT(*) FROM information_schema.processlist GROUP BY ip;
```

#### 查看 Port

```sql
SHOW VARIABLES WHERE Variable_name = 'port';
```

#### 查看用户

```sql
USE mysql;
SELECT host, user, authentication_string, plugin FROM user;
```

## 函数

以下为常用函数，完整列表见 **[MySQL 8.0 Reference Manual](https://dev.mysql.com/doc/refman/8.0/en/built-in-function-reference.html)**

### 数值函数

官方手册见 [Numeric Functions and Operators](https://dev.mysql.com/doc/refman/8.0/en/numeric-functions.html)

保留小数：
- `ROUND(x, decimals)`：四舍五入
- `TRUNCATE(x, decimals)`：直接截取
- `CEILING(x)`：向上取整，即 MIN({>=number})
- `FLOOR(x)`：向下取整，即 MAX({<=number})

<br>数学运算：
- `MOD(x, y)`：求余
  - or `x MOD y`
  - or `x % y`
- `SQRT(x)`：求平方根
- `POWER(x, y)`：求 x 的 y 幂次方

{{< expand "练习一下">}}

```sql
SELECT ROUND(3.1456, 2), TRUNCATE(3.1456, 2), CEILING(3.1456), FLOOR(3.1456);
SELECT MOD(3, 2), SQRT(16), POWER(8, 2);
```

{{< /expand >}}

### 字符串函数

官方手册见 [String Functions and Operators](https://dev.mysql.com/doc/refman/8.0/en/string-functions.html)

常用：
- `LENGTH(str)`：求长度
- `UPPER(str)`：转大写
- `LOWER(str)`：转小写
- `REPLACE(str, from_str, to_str)`：替换
- `CONCAT(str1, str2, ...)`：拼接

<br>子串提取：
- `LEFT(str, len)`：自左边取
- `RIGHT(str, len)`：自右边取
- `MID(str, pos, len)`：自指定位置取
  - or `SUBSTR(str, pos, len)`
  - or `SUBSTRING(str, pos, len)`

<br>左右处理：
  - `LTRIM(str)`：删左/头部空格
  - `RTRIM(str)`：删右/尾部空格
  - `TRIM(str)`：删左右空格
  - `LPAD(str, len, padstr)`：左填充，以达到指定长度
  - `RPAD(str, len, padstr)`：右填充，以达到指定长度

<br>其他：
- `LOCATE(substr, str)`：子串第一次出现的位置。不区分大小写
  - or `POSITION(substr IN str)`
- `REVERSE(str)`：反转字符串

{{< expand "练习一下">}}

```sql
SELECT CONCAT('first_name', ' ', 'last_name');

SELECT LPAD('molly', 10, '_'), RPAD('molly', 10, '_');

SELECT LOCATE('com', 'google.com'), POSITION("COM" IN 'google.com');
```

{{< /expand >}}

### 日期函数

官方手册见 [Date and Time Functions](https://dev.mysql.com/doc/refman/8.0/en/date-and-time-functions.html)

获取当前日期时间：
- `NOW()`：返回当前日期和时间
- `CURDATE()`：返回当前日期
  - or `CURRENT_DATE()`
- `CURTIME()`：返回当前时间
  - or `CURRENT_TIME()`

<br>提取年月日时分秒：
- `EXTRACT(unit FROM date)`：通用的提取函数（建议）。详见 [unit](https://dev.mysql.com/doc/refman/8.0/en/expressions.html#temporal-intervals)
- `YEAR(date)`：年份
- `QUARTER(date)`：季度
- `MONTH(date)`：月份
- `DAY(date)`：该月份的天数
- `HOUR(time)`：小时数
- `MINUTE(time)`：分钟数
- `SECOND(time)`：秒数
- `MONTHNAME(date)`：字符串格式的月份，如 August
- `DAYNAME(date)`：字符串格式的星期数，如 Thursday

<br>格式化：
- `DATE_FORMAT(date, format)`：详见 [format](https://dev.mysql.com/doc/refman/8.0/en/date-and-time-functions.html#function_date-format)
- `CONVERT_TZ(dt, from_tz, to_tz)`：转时区

<br>日期运算：
- `DATE_ADD(date, INTERVAL expr unit)`：unit 同 EXTRACT() 函数
  - or `DATE_SUB(date,INTERVAL -expr unit)`
- `DATEDIFF(date1, date2)`：计算相差天数，注意是 *date1 - date2*
{{< alert theme="warning" >}}
⚠️ 注意，这里不同 DBMS 相差较大
{{< /alert >}}

{{< expand "练习一下">}}

```sql
SELECT NOW(), CURDATE(), CURRENT_DATE(), CURTIME(), CURRENT_TIME();

SELECT NOW(), EXTRACT(YEAR FROM NOW()), YEAR(NOW()), QUARTER(NOW()), MONTH(NOW()), DAY(NOW()), HOUR(NOW()), MINUTE(NOW()), SECOND(NOW()), MONTHNAME(NOW()), DAYNAME(NOW());

SELECT DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s'), DATE_FORMAT(NOW(), '%W %M %d %Y %l:%i:%s %p'), DATE_FORMAT(NOW(), '%a %b %d %Y %l:%i:%s %p'), DATE_FORMAT(NOW(), '%r'), DATE_FORMAT(NOW(), '%T');

SELECT DATE_ADD(NOW(), INTERVAL 1 DAY), DATE_SUB(NOW(), INTERVAL -1 DAY);
SELECT DATEDIFF('2017-01-01', '2016-12-24');
```
{{< /expand >}}

### 聚合函数

官方手册见 [Aggregate Functions](https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html)

- `MAX(expr)`：求最大值
- `MIN(expr)`：求最小值
- `AVG(expr)`：求平均值
- `SUM(expr)`：求和
- `COUNT(expr)`：求次数

### 窗口函数

官方手册见 [Window Functions](https://dev.mysql.com/doc/refman/8.0/en/window-function-descriptions.html)

窗口函数基于**分区和排序后的查询结果的每行数据**进行计算。语法如下：

```sql
window_function OVER (
  PARTITION BY some_col
  ORDER BY another_col
)

-- window_name
SELECT
  val,
  ROW_NUMBER() OVER w AS 'row_number',
  RANK()       OVER w AS 'rank',
  DENSE_RANK() OVER w AS 'dense_rank'
FROM numbers
WINDOW w AS (ORDER BY val);

SELECT
  DISTINCT year, country,
  FIRST_VALUE(year) OVER (w ORDER BY year ASC) AS first,
  FIRST_VALUE(year) OVER (w ORDER BY year DESC) AS last
FROM sales
WINDOW w AS (PARTITION BY country);
```

<br>窗口函数可分为以下三类：

1. 聚合函数：上述 聚合函数 中的都适用；

2. 排序函数
   - `ROW_NUMBER()`：返回排名，如 1, 2, 3, 4, ...
   - `RANK()`：返回排名，如 1, 2, 2, 4, ...
   - `DENSE_RANK()`：返回排名，如 1, 2, 2, 3, ...
   - `NTILE(n)`：分成 n 组，返回组别
   - `PERCENT_RANK()`：返回排名的百分比
     - 计算公式：*(rank - 1) / (rows - 1)*
   - `CUME_DIST()`：返回值累计分布的百分比，如 top 10%
     - 计算公式：*rows(小于或大于等于当前值) / rows*

3. 值函数/偏移函数
   - `FIRST_VALUE(col)`：取第一行值
   - `LAST_VALUE(col)`：取最后一行值
   - `NTH_VALUE(col, n)`：取第 n 行值
   - `LAG(col, n, defaut)`：取向**前**偏移 n 行的值，若不存在则取 defaut
   - `LEAD(col, n, defaut)`：取向**后**偏移 n 行的值，若不存在则取 defaut

<br>【宝藏】带图理解：
- [How to use Window functions in SQL Server](https://www.sqlshack.com/use-window-functions-sql-server/)
- [Overview of SQL RANK functions](https://www.sqlshack.com/overview-of-sql-rank-functions/)
- [Calculate SQL Percentile using the PERCENT_RANK function in SQL Server](https://www.sqlshack.com/calculate-sql-percentile-using-the-sql-server-percent_rank-function/)

### 控制流函数

官方手册见 [Flow Control Functions](https://dev.mysql.com/doc/refman/8.0/en/flow-control-functions.html#function_nullif) 和 [Comparison Functions and Operators](https://dev.mysql.com/doc/refman/8.0/en/comparison-operators.html#function_coalesce)

说明：CASE 属于运算符且支持多条件，其余为函数。

- `CASE WHEN condition THEN expr1 ELSE expr2 END`
- `CASE value WHEN compare_value THEN expr1 ELSE expr2 END`
- `IF(condition, expr1, expr2)`：如果条件为真，则返回 expr1，否则返回 expr2
- `IFNULL(expr1, expr2)`：如果 expr1 不为 null 则返回 expr1，否则返回 expr2
- `NULLIF(expr1, expr2)`：如果相等，则返回 null，否则返回 expr1
- `COALESCE(expr1, expr2, ...)`：返回第一个不为 null 的值，若都为 null 则返回 null
  {{< alert theme="warning" >}}
👏 `COALESCE()` 很巧妙很好用，以下两个表达式具有相同的作用：
- COALESCE(expr1, expr2, expr3) 
- IFNULL(expr1, IFNULL(expr2, IFNULL(expr3, NULL)))
  {{< /alert >}}

{{< expand "使用 CASE 解释三个异常值处理函数 IFNULL()/NULLIF()/COALESCE()" >}}

```sql
-- IFNULL(expr1, expr2)
CASE 
  WHEN expr1 IS NOT NULL THEN expr1 
  ELSE expr2
END

-- NULLIF(expr1, expr2)
CASE 
  WHEN expr1 = expr2 THEN NULL
  ELSE expr1 
END 

-- COALESCE(expr1, expr2, expr3)
CASE 
  WHEN expr1 IS NOT NULL THEN expr1 
  WHEN expr2 IS NOT NULL THEN expr2
  WHEN expr3 IS NOT NULL THEN expr3
  ELSE NULL
END
```

{{< /expand >}}

{{< expand "练习一下">}}

```sql
SELECT IFNULL(1/0, 'yes'), IFNULL(1/1, 'yes'), IFNULL(NULL, NULL);

SELECT COALESCE(NULL, 1), COALESCE(NULL, NULL, NULL), COALESCE(NULL, NULL, NULL, 'Unknown');
SELECT COALESCE(1/0, 2/0, 3/1), IFNULL(1/0, IFNULL(2/0, IFNULL(3/1, NULL)));
```

{{< /expand >}}

### 其他函数

- `CAST(expr AS type)`：值类型转换，详见 [type](https://dev.mysql.com/doc/refman/8.0/en/cast-functions.html#function_cast)，如 CHAR/SIGNED/FLOAT/DOUBLE/DATE/DATETIME

{{< expand "练习一下">}}

```sql
SELECT CAST(3.1415 AS SIGNED);
```

{{< /expand >}}