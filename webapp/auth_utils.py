import os
import logging
from django.core.mail import send_mail, EmailMultiAlternatives
from django.conf import settings
from .models import EmailOTP

logger = logging.getLogger(__name__)


def send_otp_email(user, otp_obj):
    """
    Sends a high-contrast, formatted OTP email directly to the user's email address.
    """
    subject = f"TruthVision Security Verification Code: {otp_obj.otp_code}"
    
    html_message = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f1115; color: #ffffff; padding: 24px; margin: 0; }}
            .container {{ max-width: 520px; margin: 0 auto; background-color: #1c1f26; border: 1px solid #2e333d; border-radius: 12px; padding: 32px; text-align: center; }}
            .logo {{ font-size: 22px; font-weight: 800; color: #ffffff; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }}
            .logo span {{ color: #f97316; }}
            .otp-box {{ background-color: #121418; border: 2px solid #f97316; color: #ffffff; font-size: 34px; font-weight: 800; letter-spacing: 8px; padding: 16px; margin: 24px 0; border-radius: 8px; font-family: Consolas, Monaco, monospace; }}
            .info {{ color: #d1d5db; font-size: 15px; line-height: 1.6; margin-bottom: 12px; }}
            .footer {{ margin-top: 32px; font-size: 12px; color: #9ca3af; border-top: 1px solid #2e333d; padding-top: 16px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">Truth<span>Vision</span> AI</div>
            <p class="info">Hello <strong>{user.username}</strong>,</p>
            <p class="info">Your one-time authentication code for accessing TruthVision is:</p>
            <div class="otp-box">{otp_obj.otp_code}</div>
            <p class="info">This code is valid for <strong>10 minutes</strong>. Do not share this OTP with anyone.</p>
            <div class="footer">
                TruthVision Synthetic Media Detection &bull; Automated Security Protocol<br>
                If you did not request this login attempt, please secure your account immediately.
            </div>
        </div>
    </body>
    </html>
    """
    
    plain_message = f"TruthVision Security Verification Code: {otp_obj.otp_code}\n\nValid for 10 minutes. Do not share this OTP with anyone."
    from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', 'TruthVision Security <security@truthvision.ai>')
    
    try:
        msg = EmailMultiAlternatives(
            subject=subject,
            body=plain_message,
            from_email=from_email,
            to=[user.email]
        )
        msg.attach_alternative(html_message, "text/html")
        msg.send(fail_silently=False)
        logger.info(f"OTP successfully sent to {user.email}")
        return True
    except Exception as e:
        logger.warning(f"Notice sending email to {user.email}: {e}")
        return False
