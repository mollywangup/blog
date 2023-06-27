- [注意事项](#注意事项)
- [政策相关](#政策相关)
- [理解Firebase](#理解firebase)
- [第一步：集成Firebase SDK](#第一步集成firebase-sdk)
- [第二步：集成Analytics SDK](#第二步集成analytics-sdk)
- [第三步：集成Firebase Crashlytics SDK](#第三步集成firebase-crashlytics-sdk)
- [第四步：集成Remote Config SDK](#第四步集成remote-config-sdk)
- [第五步：集成Performance Monitoring SDK](#第五步集成performance-monitoring-sdk)


# 注意事项
1. 需注意尽量集成最新版SDK，可参考 [Release notes for SDKs or services](https://gist.github.com/mollywangup/13f49f70d25eafa077e57ffed6776316) 
2. 需要考虑隔离问题，包括开发环境和测试机等；
3. 测试机必须安装谷歌框架服务（Google Play Service）；


# 政策相关
- 受众包含儿童：需遵守儿童政策，详情及应对措施详见 []()；
- 受众不包含儿童：跳过这一步；


# 理解Firebase
Firebase 项目实际上只是一个启用了额外的 Firebase 特定配置和服务的 Google Cloud 项目；

<img src="https://user-images.githubusercontent.com/46241961/212270083-1c2afc0b-d77b-4d24-adb0-f9b38b5d0f2e.png" title="firebase-projects-hierarchy_projects-apps-resources" width=50%>


# 第一步：集成Firebase SDK
- 官方文档：https://firebase.google.com/docs/android/setup
- 核心用途：基础设施类集成；
- 流程概述：详见文档；
  - 创建项目并导出`google-services.json`配置文件；
- 测试方法：此阶段无需测试，直接在应用阶段测试；


# 第二步：集成Analytics SDK
- 官方文档：https://firebase.google.com/docs/analytics/get-started?platform=android
  Google Analytics，一般也称作Google Analytics for Firebase（GA4F）

  <img src="https://user-images.githubusercontent.com/46241961/212273126-e0852f28-9025-49e3-b131-b7f2f3accbf7.png" title="relationship-between-GA-and-firebase" width=80%>

- 核心用途：收集用户行为数据，具体包括事件和用户属性，用于产品数据分析及后续对接Firebase其他产品、Google Ads等
- 流程概述：详见文档；
- 测试方法：此阶段无需测试，直接在应用阶段测试；


# 第三步：集成Firebase Crashlytics SDK
- 官方文档：https://firebase.google.com/docs/crashlytics/get-started?platform=android
- 核心用途：收集崩溃、ANR等的报告，用于后续提升产品稳定性；
- 流程概述：详见文档；
- 测试方法
  - 强制造成一次测试崩溃以完成设置；
  - 面向开发：https://firebase.google.com/docs/crashlytics/test-implementation?platform=android


# 第四步：集成Remote Config SDK
- 官方文档：
  - https://firebase.google.com/docs/remote-config/get-started?hl=zh-cn&platform=android

    ![modify-remote-config-programmatically](https://user-images.githubusercontent.com/46241961/212282087-2da25f26-15e4-4fc4-9f81-95a0a05ed1b9.png)

  - https://firebase.google.com/docs/remote-config/propagate-updates-realtime#android
    
    <img src="https://user-images.githubusercontent.com/46241961/212281329-ddb73b6d-68ea-44ba-9759-be17b5da47d0.png" title="propagate-remote-config-updates-in-real-time" width=85%>

- 核心用途：无需客户端发版/发包，即可实现云端/服务端控制提前定义好的参数；
- 流程概述：详见文档；
- 测试方法：此阶段无需测试，直接在应用阶段测试；


# 第五步：集成Performance Monitoring SDK
- 官方文档：https://firebase.google.com/docs/perf-mon/get-started-android
  
  ![firebase-performance](https://user-images.githubusercontent.com/46241961/212284225-7bfab2f5-f85e-4f9c-8463-3c9b05ea89c7.png)

- 核心用途：收集启动时长，监控网络请求等产品性能数据，用于后续提升产品性能、优化开屏位置的广告请求等；
- 流程概述：详见文档；
- 测试方法：
  - 面向开发：https://firebase.google.com/docs/perf-mon/troubleshooting?platform=android
  - 常见问题：https://stackoverflow.com/questions/tagged/firebase-performance
