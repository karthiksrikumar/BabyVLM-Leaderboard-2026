"""Send BabyVLM notification emails.

Three delivery backends, chosen automatically:

1. **dry-run** — if ``BABYVLM_EMAIL_DRYRUN=1``, emails are printed, not sent
   (used by the tests and for local dev).
2. **SMTP** — if ``BABYVLM_SMTP_HOST`` is set, send via that server
   (``BABYVLM_SMTP_PORT``, ``BABYVLM_SMTP_USER``, ``BABYVLM_SMTP_PASS``,
   ``BABYVLM_SMTP_STARTTLS`` optional). Use this for a Gmail app-password sender.
3. **sendmail** — otherwise pipe the message to the local MTA
   (``/usr/sbin/sendmail``). This works out of the box on BU's SCC.
"""
import os
import smtplib
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

from runner.config import EMAIL_FROM, RECIPIENTS


def _build_message(subject: str, body_text: str, body_html: str | None, recipients: list[str]) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))
    return msg


def send_email(subject: str, body_text: str, body_html: str | None = None, recipients: list[str] | None = None) -> bool:
    """Send an email to the recipients (default: the BabyVLM organizer list).

    Returns True on apparent success, False otherwise. Never raises — a failed
    notification must not abort an evaluation run.
    """
    recipients = recipients or RECIPIENTS
    msg = _build_message(subject, body_text, body_html, recipients)

    # 1. dry-run
    if os.environ.get("BABYVLM_EMAIL_DRYRUN") == "1":
        print("=" * 70)
        print(f"[DRY-RUN EMAIL] To: {', '.join(recipients)}")
        print(f"Subject: {subject}")
        print("-" * 70)
        print(body_text)
        print("=" * 70)
        return True

    # 2. SMTP
    smtp_host = os.environ.get("BABYVLM_SMTP_HOST")
    if smtp_host:
        try:
            port = int(os.environ.get("BABYVLM_SMTP_PORT", "587"))
            with smtplib.SMTP(smtp_host, port, timeout=30) as server:
                if os.environ.get("BABYVLM_SMTP_STARTTLS", "1") == "1":
                    server.starttls()
                user = os.environ.get("BABYVLM_SMTP_USER")
                pw = os.environ.get("BABYVLM_SMTP_PASS")
                if user and pw:
                    server.login(user, pw)
                server.sendmail(EMAIL_FROM, recipients, msg.as_string())
            print(f"[email] sent via SMTP to {', '.join(recipients)}")
            return True
        except Exception as e:
            print(f"[email] SMTP send failed ({e}); falling back to sendmail")

    # 3. local sendmail
    sendmail = "/usr/sbin/sendmail"
    if not os.path.exists(sendmail):
        sendmail = "sendmail"
    try:
        proc = subprocess.run(
            [sendmail, "-t", "-oi"],
            input=msg.as_string().encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode == 0:
            print(f"[email] sent via sendmail to {', '.join(recipients)}")
            return True
        print(f"[email] sendmail exit {proc.returncode}: {proc.stderr.decode(errors='ignore')[:200]}")
        return False
    except Exception as e:
        print(f"[email] sendmail failed ({e}). Set BABYVLM_SMTP_* to use an external SMTP server.")
        return False
