- [Step1: Get Started with Android](#step1-get-started-with-android)
- [Step2: IS Mediation for FAN](#step2-is-mediation-for-fan)
- [Step3: 政策 (COPPA/CCPA/GDPR)](#step3-政策-coppaccpagdpr)
- [Step4: Impression level revenue integration](#step4-impression-level-revenue-integration)


# Step1: Get Started with Android
https://developers.is.com/ironsource-mobile/android/android-sdk/

# Step2: IS Mediation for FAN
https://developers.is.com/ironsource-mobile/android/facebook-mediation-guide/


> Note: Your app won’t be able to receive Audience Network ads until payout information has been added and it was sent for review.

# Step3: 政策 (COPPA/CCPA/GDPR)
https://developers.is.com/ironsource-mobile/android/regulation-advanced-settings/


# Step4: Impression level revenue integration
https://developers.is.com/ironsource-mobile/android/ad-revenue-measurement-integration/
```java
/** 
Invoked when the ad was displayed successfully and the impression data was recorded 
**/ 
@Override 
public void onImpressionSuccess(ImpressionData impressionData) { 
 // The onImpressionSuccess will be reported when the rewarded video and interstitial ad is opened. 
 // For banners, the impression is reported on load success.  Log.d(TAG, "onImpressionSuccess" + impressionData); 
        if (impressionData != null) 
        { 
                 Bundle bundle = new Bundle(); 
                 bundle.putString(FirebaseAnalytics.Param.AD_PLATFORM, "ironSource");  
                 bundle.putString(FirebaseAnalytics.Param.AD_SOURCE,impressionData.adNetwork());  
                 bundle.putString(FirebaseAnalytics.Param.AD_FORMAT, impressionData.getAdUnit());          
                 bundle.putString(FirebaseAnalytics.Param.AD_UNIT_NAME, impressionData.getInstanceName());
                 bundle.putString(FirebaseAnalytics.Param.CURRENCY, "USD");  
                 bundle.putDouble(FirebaseAnalytics.Param.VALUE, impressionData.getRevenue());  
                 
                 mFirebaseAnalytics.logEvent(FirebaseAnalytics.Event.AD_IMPRESSION, bundle);  
        
        } 
}} 
}
```
