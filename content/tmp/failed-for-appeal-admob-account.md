# How do users get to your site, mobile app and/or YouTube channel? How do you promote your content? *
93% of the new users come from apps.facebook.com as Facebook is the main channel that we promote our mobile app, and the rest are organic users. (Data source: Firebase)

# Have you or your site, mobile app and/or YouTube channel ever violated the AdSense or Ad Manager programme policies or Terms & Conditions? If yes, how? *
We always strictly enable test ads and comply with the policies & guidelines for each ad format that we have integrated. 
Our admob account status has experienced Limited ad serving twice. The first time starts on 2022-10-23 and ends on 2022-12-01, the second time starts on 2022-12-10.

# What was the reason for invalid activity on your site, mobile app and/or YouTube channel? Please provide detailed information about all of the specific reasons that you believe are relevant in your case. *
1. For the fisrt time the ad serving recover, we noticed the abnormal ad CTR especially on app open ads, we're trying to solve it on the next app update. But the account was deactivated before we did this.
2. Between 2022-12-05 and 2022-12-10, the new users and ad requests had fluctuations as we adjusted the promotion strategy, in detail are budgets and country.

By analyzing the raw data from BigQuery, we came to the following conclusions:
1. The most likely reason for the invalid activity might be due to poor traffic quality although we are victims. We found that the higher CLICK_CONNECT event triggers (when a user starts a VPN connection as it is the core function of our app) the more normal ad CTR is. This is the key point we should improve.
2. We believe that our ad integration is legal but room for optimization. Such as the show rate, frequency controlling to balance user experience and ad revenue, and smoother user experience on UI level to avoid ads showing accidentally.


# What changes will you implement to help improve ad traffic quality on your site, mobile app and/or YouTube channel? *
As a publisher, we are also victims. It should be in a win-win situation between AdMob and us in the long run.
1. First of all, optimize the traffic quality. (We should always do this.)
2. Optimized ad frequency by analyzing user in-app behaviors to balance user experience and ad revenue, and using Firebase Remote Config to control it in real-time. (Has taken effect in version1.2.0)
3. To improve the ad show rate on two existing ad placements, we decreased the ad requests and optimized the appropriate ad loading time. (Has taken effect in version1.2.0)
4. Optimized app UI interaction for smoother user experience to avoid ads showing accidentally, such as adapting to RTL device system. (Has taken effect in version1.2.0)
5. Optimized ad click events which we manually logged in order to monitor and collect abnormal traffic. (Has taken effect in version1.2.0)

# Please include any data from your site, mobile app and/or YouTube channel traffic logs or reports that indicate suspicious IP addresses, referrers or requests which could explain invalid activity. *
We consider the following users to be low quality, as they opened the app (`FIRST_OPEN`) and removed (`APP_REMOVE`) it without doing anything (`CLICK_CONNECT`) except clicking on ads (`AD_CLICK_GOAL`). 
Due to space limitations, the following is an example, and which means the ad CTR is 2/3.
Format: {user_pseudo_id};{ad_response_id of `AD_SHOW_SUCCESS`};{ad_response_id of `AD_CLICK_GOAL`}
7bef1ae81260cee5c2753f20106227d3;ws6TY5b9EoyRmLAPwp6eoAI,CJOYke3h7fsCFXPh5godnfUMVQ,bPCTY6DYHdbKmLAPlIK3mAE;ws6TY5b9EoyRmLAPwp6eoAI,bPCTY6DYHdbKmLAPlIK3mAE
Note:
1. All the data comes from BigQuery.
2. FIRST_OPEN, APP_REMOVE, SCREEN_VIEW are auto-logged events by Firebase.
3. AD_LOAD_SUCCESS, AD_SHOW_SUCCESS, AD_CLICK_GOAL are manually logged along with param "ad_response_id" for measuring and monitoring ad performance. 
4. "user_pseudo_id" is the id of the user defined by Firebase, and "ad_response_id" is the id of the ad defined by AdMob.