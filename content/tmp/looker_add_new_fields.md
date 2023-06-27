

```sql
-- FORMAT_DATETIME('%F', First Touch Time)
-- PARSE_DATE('%Y-%m-%d', First Open Date)

-- Event Time HKT
DATETIME_ADD(Event Time, INTERVAL 8 HOUR)

-- First Touch Time HKT
DATETIME_ADD(First Touch Time, INTERVAL 8 HOUR)


-- Event Date HKT
PARSE_DATE('%Y-%m-%d', FORMAT_DATETIME('%F', Event Time HKT))

-- First Open Date
PARSE_DATE('%Y-%m-%d', FORMAT_DATETIME('%F', First Touch Time HKT))


-- Days X
DATETIME_DIFF(Event Time HKT, First Touch Time HKT, DAY)

-- tmp_first_touch_time_diff_hour
DATETIME_DIFF(First Touch Time HKT, First Touch Time, HOUR)


-- Time Zone
CASE
	WHEN Time Zone Offset = 28800
THEN 'HKT'
ELSE 'not_HKT'
END

-- Media Source
CASE 
	WHEN Acquired Source IN ('apps.facebook.com')
THEN 'facebook_ads'
ELSE 'organic'
END

```

- `FORMAT_DATETIME`函数的`format_string`格式见 [Format elements](https://cloud.google.com/bigquery/docs/reference/standard-sql/format-elements#format_elements_date_time)
- `DATETIME_DIFF`是前者减后者
- 常用函数见 [Function list](https://support.google.com/looker-studio/table/6379764?hl=en)


## 坑
参考 [3+ reasons your GA4 data doesn’t match](https://analyticscanvas.com/3-reasons-your-ga4-data-doesnt-match/)
- `event_date`: 创建时设置的时区；
- `event_timestamp`: 默认UTC时区；


- looker里的`Event Time`是`DATETIME`格式而不是`TIMESTAMP`格式；
- `Days X`不同于`Days Since First Touch`，前者是按照`天数`相减，后者是按照`小时/分钟`相减；