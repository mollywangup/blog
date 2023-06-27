- [注意事项](#注意事项)
- [政策相关](#政策相关)
- [方法概述](#方法概述)


# 注意事项
1. 需注意尽量集成最新版SDK，可参考 [Release notes for SDKs or services](https://gist.github.com/mollywangup/13f49f70d25eafa077e57ffed6776316)
2. `facebook_app_id`是facebook叫法，`facebook_developer_id`是接口叫法，指的是同一个东西，详见下文；


# 政策相关
- 受众包含儿童：需遵守儿童政策，详情及应对措施详见 []()；
- 受众不包含儿童：跳过这一步；


# 方法概述
- 官方文档
  - GitHub: 
    - [Facebook SDK for Android](https://github.com/facebook/facebook-android-sdk)
  - Facebook SDK: 
    - [Getting Started with the Facebook SDK for Android](https://developers.facebook.com/docs/android/getting-started)
    - [Add App Events](https://developers.facebook.com/docs/app-events/getting-started-app-events-android#step-3--integrate-the-facebook-sdk-in-your-android-app)

- 核心用途：基础设施类集成，核心是初始化后的自动统计事件功能（`Automatically Logged Events`）；
- 流程概述
  - 先从服务端接口获取参数配置：
    - `facebook_app_id`形如`1047692719395097`；
    - `facebook_client_token`形如`9cf04d4ff0e7661a828f571ac4735ab9`；
    - optional: `fb_login_protocol_scheme`: 仅当App中包含Facebook登录功能时才需要；

    ```json
    {
        "facebook_app_id": "",
        "facebook_client_token": "",
        "fb_login_protocol_scheme": ""
    }
    ```
  - 再直接按照上述官方文档中的 _**Step 3: Integrate the Facebook SDK in Your Android App**_ 操作即可，其中：
    - 若服务端返回非空值（即已配置），则客户端正常做初始化处理；
    - 若服务端返回空值（即未配置），则客户端不做初始化处理或直接忽略该SDK；

- 测试方法
  - 面向开发：[Enabling Debug Logs](https://developers.facebook.com/docs/app-events/getting-started-app-events-android#enabling-debug-logs)
    ```java
    FacebookSdk.setIsDebugEnabled(true);
    FacebookSdk.addLoggingBehavior(LoggingBehavior.APP_EVENTS);
    ```
  - 面向运营：
    - option 1: [App Ads Helper](https://developers.facebook.com/tools/app-ads-helper/)
    - option 2: [Events Manager](https://business.facebook.com/events_manager2/list/app/790833925449113/overview?act=518122528886487)