from django.urls import path
from . import views

urlpatterns = [
    # Main Navigation
    path('', views.dashboard_view, name='dashboard'),
    path('image/', views.image_detect_view, name='image_detect'),
    path('video/', views.video_detect_view, name='video_detect'),
    path('webcam/', views.webcam_view, name='webcam'),
    path('benchmarks/', views.benchmarks_view, name='benchmarks'),
    path('report/', views.forensic_report_view, name='forensic_report'),
    path('api/detect-frame/', views.api_detect_frame, name='api_detect_frame'),

    # Authentication & Email OTP
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('google-auth/', views.google_login_view, name='google_login'),
    path('google-oauth/', views.google_oauth_redirect_view, name='google_oauth_redirect'),
    path('google-callback/', views.google_oauth_callback_view, name='google_oauth_callback'),
    path('verify-otp/', views.verify_otp_view, name='verify_otp'),
    path('resend-otp/', views.resend_otp_view, name='resend_otp'),
    path('logout/', views.logout_view, name='logout'),

    # Professional Admin Portal
    path('admin-portal/', views.admin_portal_view, name='admin_portal'),
    path('admin-portal/delete/<int:record_id>/', views.api_admin_delete_record, name='admin_delete_record'),

    # User Profile & Advanced Settings
    path('settings/', views.profile_settings_view, name='profile_settings'),
    path('settings/regenerate-key/', views.api_regenerate_api_key, name='regenerate_api_key'),
    path('settings/export-history/', views.api_export_scan_history, name='export_scan_history'),
]
