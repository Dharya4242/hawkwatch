"""
backend/alerts.py — Twilio SMS alerts for high-severity incidents.

Silently skips if Twilio credentials are not configured in .env.
Called from upload and stream routes after each incident is processed.
"""

import os

from dotenv import load_dotenv

load_dotenv()

_SEVERITY_THRESHOLD: str = os.getenv("SEVERITY_ALERT_THRESHOLD", "CRITICAL")


def send_sms_alert(incident) -> bool:
    """
    Send a Twilio SMS for incidents at or above the configured severity threshold.

    Args:
        incident: Any object with .severity, .category, .video_source,
                  .timestamp, and .recommended_action attributes.

    Returns:
        True if the SMS was sent, False if skipped or failed.
    """
    if incident.severity != _SEVERITY_THRESHOLD:
        return False

    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_num = os.getenv("TWILIO_PHONE_NUMBER", "")
    to_num = os.getenv("ALERT_PHONE_NUMBER", "")

    if not all([sid, token, from_num, to_num]):
        print(f"  [Alert] Twilio not configured — skipping SMS for {incident.severity} incident")
        return False

    try:
        from twilio.rest import Client

        body = (
            f"HAWKWATCH {incident.severity} ALERT\n"
            f"Category : {incident.category}\n"
            f"Source   : {incident.video_source}\n"
            f"Time     : {incident.timestamp}\n"
            f"Action   : {incident.recommended_action}"
        )
        Client(sid, token).messages.create(body=body, from_=from_num, to=to_num)
        print(f"  [Alert] SMS sent to {to_num}")
        return True

    except Exception as e:
        print(f"  [Alert] ERROR sending SMS: {e}")
        return False
