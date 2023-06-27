- [logEvent](#logevent)
  - [ad\_response\_id](#ad_response_id)
- [setUserProperty](#setuserproperty)


# logEvent
[使用Firebase SDK进行事件埋点](https://fengche.feishu.cn/docx/doxcnnBi25zvgq5ZzhASxQTBIaf)




## ad_response_id
没有就随机生成自定义的，格式：
  - 长度：
    - 18到26个字符（包含18和26）
  - 可包含的字符：
    - `[a-zA-Z0-9]`：即字母数字，必须包含；
    - `-`：即中划线，非必须包含；
    - `_`：即下划线，非必须包含；
  - 例子：
    ```
    CK-I-JGL2fsCFZC_0QQdZQEE1w
    CNTrw7ra2fsCFWbYcwEdeoUF_w
    CKKWm4bI2fsCFaUH5godnHQAKA
    -EOJY-PMAo6X-wan3by4BQ
    ```


# setUserProperty
[【归因】为新用户设置用户属性（持续追加）](https://fengche.feishu.cn/docx/V2BTdF8sQonh8txoDXCccCZlnAb)
