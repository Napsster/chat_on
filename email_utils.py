"""SMTP email sending — dormant until SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD
are configured. Works with Gmail, Office365, or any standard SMTP relay."""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Tuple

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USERNAME)
SMTP_USE_SSL = os.environ.get("SMTP_USE_SSL", "false").lower() == "true"


def send_email(to: str, subject: str, body: str) -> Tuple[bool, str]:
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD):
        return False, "SMTP not configured (SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD missing)"
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_FROM
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            server.starttls()
        with server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to], msg.as_string())
        return True, "Sent"
    except Exception as e:
        return False, str(e)
