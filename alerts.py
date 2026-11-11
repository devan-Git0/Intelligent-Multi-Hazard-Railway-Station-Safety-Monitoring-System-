"""
STEP 7 -- Alert Dispatcher
Sends each new Incident out through whichever channels are enabled in
config.py: console, spoken (pyttsx3), Slack webhook, email (SMTP).

Network channels run on a small background thread pool so a slow Slack
or SMTP call can never stall the video loop. Channels with no credentials
configured are skipped silently. Logging to the database is handled in
main.py, not here.

Run this file directly to test dispatch with no real credentials.
"""

import smtplib
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage

import requests

import config
from datatypes import Incident
from risk_fusion import LEVEL_ORDER


def _format(incident: Incident) -> str:
    ts = datetime.fromtimestamp(incident.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return (f"[{incident.severity_level}] {incident.hazard_type.replace('_', ' ')} "
            f"(score {incident.severity_score:.0f}) at {ts} -- {incident.details}")


class AlertDispatcher:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alerts")
        self._tts = None
        if config.ENABLE_VOICE_ALERTS:
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
            except Exception as exc:  # no audio device / driver on this machine
                print(f"[alerts] voice alerts disabled: {exc}")

    def dispatch(self, incident: Incident) -> None:
        if LEVEL_ORDER[incident.severity_level] < LEVEL_ORDER[config.MIN_ALERT_LEVEL]:
            return
        message = _format(incident)

        if config.ENABLE_CONSOLE_ALERTS:
            print(f"\n*** ALERT {message}")
        if self._tts is not None and LEVEL_ORDER[incident.severity_level] >= LEVEL_ORDER["HIGH"]:
            self._speak(incident)
        if config.SLACK_WEBHOOK_URL:
            self._pool.submit(self._send_slack, message)
        if (config.EMAIL_ENABLED and config.EMAIL_TO
                and LEVEL_ORDER[incident.severity_level] >= LEVEL_ORDER[config.EMAIL_MIN_LEVEL]):
            self._pool.submit(self._send_email, incident, message)

    # -- channels -----------------------------------------------------------
    def _speak(self, incident: Incident) -> None:
        try:  # runs on the caller's thread; pyttsx3 isn't safe to share across threads
            self._tts.say(f"Warning. {incident.hazard_type.replace('_', ' ')} detected.")
            self._tts.runAndWait()
        except Exception as exc:
            print(f"[alerts] voice failed: {exc}")

    @staticmethod
    def _send_slack(message: str) -> None:
        try:
            r = requests.post(config.SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
            r.raise_for_status()
        except Exception as exc:
            print(f"[alerts] Slack failed: {exc}")

    @staticmethod
    def _send_email(incident: Incident, message: str) -> None:
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[Railway Safety] {incident.severity_level}: {incident.hazard_type}"
            msg["From"] = config.EMAIL_FROM or config.EMAIL_USERNAME
            msg["To"] = ", ".join(config.EMAIL_TO)
            msg.set_content(message)
            with smtplib.SMTP(config.EMAIL_SMTP_HOST, config.EMAIL_SMTP_PORT, timeout=10) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(config.EMAIL_USERNAME, config.EMAIL_PASSWORD)
                s.send_message(msg)
        except Exception as exc:
            print(f"[alerts] email failed: {exc}")

    def close(self) -> None:
        self._pool.shutdown(wait=True)


if __name__ == "__main__":
    d = AlertDispatcher()
    d.dispatch(Incident("track_crossing", 1, 95.0, "CRITICAL", "person in danger zone", 1_700_000_000.0))
    d.dispatch(Incident("water_accumulation", -1, 30.0, "LOW", "should NOT print (below MIN_ALERT_LEVEL)", 1_700_000_000.0))
    d.close()
    print("\nAlert dispatcher OK (Slack/email skipped: no credentials configured).")
