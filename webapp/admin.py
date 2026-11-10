from django.contrib import admin
from .models import DetectionRecord, EmailOTP


@admin.register(DetectionRecord)
class DetectionRecordAdmin(admin.ModelAdmin):
    list_display = ('media_name', 'media_type', 'classification', 'confidence', 'risk_score', 'risk_level', 'user', 'created_at')
    list_filter = ('media_type', 'classification', 'risk_level', 'created_at')
    search_fields = ('media_name', 'user__username', 'likely_technique', 'summary_notes')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)


@admin.register(EmailOTP)
class EmailOTPAdmin(admin.ModelAdmin):
    list_display = ('user', 'otp_code', 'is_used', 'created_at', 'expires_at')
    list_filter = ('is_used', 'created_at')
    search_fields = ('user__username', 'user__email', 'otp_code')
    readonly_fields = ('created_at',)
