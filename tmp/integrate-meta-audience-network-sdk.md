

# 官方文档
## interstitial
https://developers.facebook.com/docs/audience-network/setting-up/ad-setup/android/interstitial

## native_advanced
https://developers.facebook.com/docs/audience-network/setting-up/ad-setup/android/native

注意事项：
  - 原生广告的元素区别于AdMob的，直接按照FAN的官方定义即可；
  
    <img src="https://user-images.githubusercontent.com/46241961/212447931-d460a4a5-77d8-408a-a417-69d1312cd814.png" title="meta-audience-network-native-ads-template-medium-size" width=50%>

    View #1: Ad Icon
    View #2: Ad Title
    View #3: Sponsored Label
    View #4: AdOptionsView
    View #5: MediaView
    View #6: Social Context
    View #7: Ad Body
    View #8: Call to Action button


## native_banner
https://developers.facebook.com/docs/audience-network/setting-up/ad-setup/android/native-banner

注意事项：
  - 原生横幅广告的元素，直接按照FAN的官方定义即可；

    <img src="https://user-images.githubusercontent.com/46241961/212447902-cf91628d-54b1-46ef-a51b-7375ced0c2c7.png" title="meta-audience-network-native-banner-ads-template" width=50%>

    View #1: AdOptionsView
    View #2: Sponsored Label
    View #3: Ad Icon
    View #4: Ad Title
    View #5: Social Context
    View #6: Call-to-Action button


# 政策
## COPPA
政策：https://developers.facebook.com/docs/audience-network/optimization/best-practices/coppa
政策应对：
  - 需要设置`setMixedAudience`即将`setIsChildDirected`设置为`false`，详见：https://developers.facebook.com/docs/reference/android/current/class/AdSettings/#setMixedAudience
## CCPA
政策：https://developers.facebook.com/docs/audience-network/optimization/best-practices/ccpa
政策应对：
  - 需要设置`setDataProcessingOptions`即启用`Limited Data Use (LDU)`，详见上述政策链接：
    ```java
    AdSettings.setDataProcessingOptions(new String[] {"LDU"}, 1, 1000);
    ```


# 测试
## 仅添加测试设备
已添加为测试设备（注意：没有添加Test Users，因此只能使用默认素材类型或者由我短暂手动控制素材类型）：
https://developers.facebook.com/docs/audience-network/setting-up/testing/platform

## 手动测试广告素材类型（暂时不加）
https://developers.facebook.com/docs/audience-network/setting-up/test/test-device
因为需要先添加Test Users