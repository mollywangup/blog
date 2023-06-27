App state | Notification | Data | Both
Foreground | onMessageReceived | onMessageReceived | onMessageReceived
Background | System tray | onMessageReceived | Notification: system trayData: in extras of the intent.
When app is in background onMessageReceived() is not called. This is how the behavior is.
When app is in background,notification will be handled by system tray and data will be handled by extras in intent in main launcher.



Local-Notification does not fire is app is closed (at least on Android): https://github.com/ionic-team/capacitor-plugins/issues/766
Push Notifications when app is closed: https://stackoverflow.com/questions/24313539/push-notifications-when-app-is-closed

> Your advice about the battery settings is CRUCIAL.


Local/Push Notifications（说明）: https://docs.coronalabs.com/guide/events/appNotification/index.html#localpush-notifications
notifications.scheduleNotification(): https://docs.coronalabs.com/plugin/notifications-v2/scheduleNotification.html
> Unlike iOS, local notifications on Android are managed by the application and not by the operating system. This means that all scheduled notifications and status bar notifications will be cleared when the application process terminates. However, pressing the Back key to exit the application window will not terminate the application process — this only destroys the application window which runs your project's Lua scripts and it will receive the "applicationExit" system event just before being destroyed. Thus, the application process will continue to run in the background — this is standard Android application behavior which allows its notifications to remain available. If the application process gets fully terminated, Corona will automatically restore all pending notifications when the application restarts.

