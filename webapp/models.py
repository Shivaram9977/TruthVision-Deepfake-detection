import random
import datetime
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class DetectionRecord(models.Model):
    MEDIA_TYPE_CHOICES = [
        ('IMAGE', 'Image'),
        ('VIDEO', 'Video'),
        ('WEBCAM', 'Live Webcam'),
    ]

    CLASSIFICATION_CHOICES = [
        ('REAL', 'Authentic / Real'),
        ('DEEPFAKE', 'Deepfake Manipulated'),
        ('SUSPICIOUS', 'Suspicious / Inconclusive'),
    ]

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='detections')
    media_name = models.CharField(max_length=255)
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES, default='IMAGE')
    classification = models.CharField(max_length=20, choices=CLASSIFICATION_CHOICES, default='REAL')
    confidence = models.FloatField(default=0.0)
    deepfake_probability = models.FloatField(default=0.0)
    authenticity_score = models.FloatField(default=100.0)
    risk_score = models.FloatField(default=0.0)
    risk_level = models.CharField(max_length=20, default='LOW')
    likely_technique = models.CharField(max_length=100, default='None')
    suspicious_zones = models.TextField(blank=True, default='')
    boundary_artifact_ratio = models.FloatField(default=0.0)
    frames_analyzed = models.IntegerField(default=1)
    flagged_frames_count = models.IntegerField(default=0)
    media_url = models.CharField(max_length=500, blank=True, default='')
    face_crop_url = models.CharField(max_length=500, blank=True, default='')
    gradcam_url = models.CharField(max_length=500, blank=True, default='')
    summary_notes = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.media_name} - {self.classification} ({self.risk_score:.1f}%)"


class EmailOTP(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='email_otps')
    otp_code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def is_valid(self):
        return not self.is_used and timezone.now() <= self.expires_at

    @classmethod
    def generate_otp_for_user(cls, user, validity_minutes=10):
        # Invalidate existing unused OTPs
        cls.objects.filter(user=user, is_used=False).update(is_used=True)
        otp = f"{random.randint(100000, 999999)}"
        expires = timezone.now() + datetime.timedelta(minutes=validity_minutes)
        return cls.objects.create(
            user=user,
            otp_code=otp,
            expires_at=expires,
            is_used=False
        )

    def __str__(self):
        return f"OTP for {self.user.username}: {self.otp_code} (Used: {self.is_used})"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    organization = models.CharField(max_length=150, blank=True, default='Forensics Lab')
    bio = models.TextField(blank=True, default='AIML Deep Learning & Media Forensics Analyst')
    
    # AI Engine Preferences
    default_model = models.CharField(max_length=50, default='EfficientNet-B0')
    default_fps = models.FloatField(default=2.0)
    risk_threshold = models.IntegerField(default=50)
    gradcam_colormap = models.CharField(max_length=30, default='JET')
    
    # Security Preferences
    enable_2fa_email = models.BooleanField(default=True)
    api_key = models.CharField(max_length=64, blank=True, default='')
    webhook_url = models.CharField(max_length=300, blank=True, default='')

    def __str__(self):
        return f"Profile of {self.user.username}"

    @classmethod
    def get_for_user(cls, user):
        profile, created = cls.objects.get_or_create(
            user=user,
            defaults={
                'api_key': f"tv_live_{user.id}_{random.randint(10000000, 99999999)}"
            }
        )
        if not profile.api_key:
            profile.api_key = f"tv_live_{user.id}_{random.randint(10000000, 99999999)}"
            profile.save()
        return profile

