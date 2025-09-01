# scripts/discord_alerts.py
import os
from datetime import datetime
from typing import Optional
from discordwebhook import Discord

def _get_webhook_url(explicit_url: Optional[str] = None) -> str:
    """
    Resolve the webhook URL in this order:
    1) Explicit function argument
    2) Environment variable DISCORD_WEBHOOK_URL
    """
    url = explicit_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        raise ValueError(
            "No Discord webhook URL provided. "
            "Set DISCORD_WEBHOOK_URL in your environment or pass webhook_url=..."
        )
    return url

def send_discord_message(content: str, webhook_url: Optional[str] = None) -> bool:
    """
    Post a plain text message to Discord using discordwebhook.

    Returns True if sent without raising, else False.
    """
    try:
        url = _get_webhook_url(webhook_url)
        discord = Discord(url=url)
        discord.post(content=content)
        return True
    except Exception:
        return False

def send_transactions_reminder(
    league_name: str,
    deadline_str_local: str,
    notes: str = "",
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Convenience wrapper to send a standardized transactions reminder.
    """
    lines = [
        f"**{league_name} — Transactions Reminder**",
        f"Deadline = {deadline_str_local}",
    ]
    if notes:
        lines.append(f"Notes = {notes}")

    content = "\n".join(lines)
    return send_discord_message(content=content, webhook_url=webhook_url)
