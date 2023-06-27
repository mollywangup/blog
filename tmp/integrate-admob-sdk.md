- [注意事项](#注意事项)
- [适用版本（阶段性工作）](#适用版本阶段性工作)
- [政策相关](#政策相关)
- [前期：集成Google Mobile Ads SDK](#前期集成google-mobile-ads-sdk)


# 注意事项
1. 需注意尽量集成最新版SDK，可参考 [Release notes for SDKs or services](https://gist.github.com/mollywangup/13f49f70d25eafa077e57ffed6776316)
2. 需要考虑隔离问题，包括开发环境和测试机等；
3. 关于后续使用广告单元ID：
   - **必须遵守**：日常开发必须使用测试广告单元ID，只有打release包时才需要切换为生产环境；
   - 原因：政策原因，AdMob是出于保护广告生态系统的各个角色的出发点，落地到开发者身上，就是日常开发不能点击正式广告，如果点击了，就限制广告变现；
4. 测试机必须安装谷歌框架服务（Google Play Service）；
5. `AdMob SDK`和`Google Mobile Ads SDK`指的是同一个东西，日常沟通使用前者，开发文档一般使用后者；


# 适用版本（阶段性工作）
1. 前期：仅作为流量源（Network/Ad source）接入；
2. 后期：作为聚合平台/中介（Mediation）接入，需要添加其他流量源的adapter之类的；


# 政策相关
- 受众包含儿童：需遵守儿童政策，详情及应对措施详见 []()；
- 受众不包含儿童：跳过这一步；


# 前期：集成Google Mobile Ads SDK
- 官方文档：https://developers.google.com/admob/android/quick-start
- 核心用途：基础设施类集成；
- 流程概述：详见文档；
  - 先获取`AdMob App ID`，来自`google-services.json`，因此需要先更新此配置文件；
  - 关于声明`com.google.android.gms.permission.AD_ID`权限：
    - **需注意上述儿童政策**；
    - 如果SDK版本 >= 20.4.0，则直接跳过，因为已自动声明；
    - 如果SDK版本 <= 20.3.0，则需要手动声明；
  - 高级做法
    - 目标：优化初始化和广告加载（Beta 版）
    - 官方文档：https://developers.google.com/admob/android/optimize-initialization
    - 需要满足：SDK版本 >= 21.0.0
    - 代码层面：默认值时`false`，修改为`true`即可；
- 测试方法：此阶段无需测试，直接在应用阶段测试；

