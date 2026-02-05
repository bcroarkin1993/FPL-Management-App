# scripts/waiver_alerts.py  (GitHub Actions-friendly; no config.py imports including from utils.py)
#
# Supports both Draft and Classic FPL alerts:
#   - Draft: 25.5h before kickoff (waiver/transaction deadline)
#   - Classic: 1.5h before kickoff (transfer deadline)

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

TZ = ZoneInfo("America/New_York")

# Default offsets (hours before kickoff)
DRAFT_OFFSET_HOURS = 25.5
CLASSIC_OFFSET_HOURS = 1.5


def _get_current_gameweek() -> int:
    """Fetch current/next GW from the official Draft endpoint."""
    r = requests.get("https://draft.premierleague.com/api/game", timeout=20)
    r.raise_for_status()
    data = r.json()
    return int(data["next_event"] if data.get("current_event_finished") else data["current_event"])


def _fixtures_for_event(gw: int):
    """Canonical fixtures endpoint; query by GW via params to avoid caching issues."""
    r = requests.get("https://fantasy.premierleague.com/api/fixtures/", params={"event": int(gw)}, timeout=20)
    r.raise_for_status()
    js = r.json()
    return js if isinstance(js, list) else []


def _earliest_kickoff_et(gw: int) -> datetime:
    """Earliest kickoff for a GW in ET."""
    times = []
    for fx in _fixtures_for_event(gw):
        k = fx.get("kickoff_time")
        if not k:
            continue
        dt_utc = datetime.fromisoformat(k.replace("Z", "+00:00"))
        times.append(dt_utc.astimezone(TZ))
    if not times:
        raise RuntimeError(f"No kickoff times found for GW {gw}")
    return min(times)


def get_next_transaction_deadline(offset_hours: float = 25.5, gw: int = None):
    """Returns (deadline_et, kickoff_et, gw). Deadline = earliest kickoff - offset."""
    if gw is None:
        gw = _get_current_gameweek()
    kickoff_et = _earliest_kickoff_et(gw)
    return kickoff_et - timedelta(hours=float(offset_hours)), kickoff_et, gw


def _check_and_send_alert(
    webhook: str,
    mention: str,
    deadline_et: datetime,
    gw: int,
    alert_type: str,
    now_et: datetime,
) -> bool:
    """
    Check if we're in an alert window and send notification if so.

    Args:
        webhook: Discord webhook URL
        mention: Mention string (user/role pings)
        deadline_et: The deadline datetime
        gw: Gameweek number
        alert_type: "Draft" or "Classic"
        now_et: Current time in ET

    Returns:
        True if an alert was sent, False otherwise
    """
    hours_left = (deadline_et - now_et).total_seconds() / 3600

    # Log timing info
    print(f"[waiver_alerts:{alert_type}] Deadline: {deadline_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts:{alert_type}] Hours until deadline: {hours_left:.2f}")

    # Skip if deadline has passed
    if hours_left < 0:
        print(f"[waiver_alerts:{alert_type}] Deadline has passed, skipping")
        return False

    # Fire at ~24h / 6h / 1h (tolerance ±30 min to accommodate GitHub Actions
    # scheduling delays and fractional offset hours)
    for target in (24, 6, 1):
        if abs(hours_left - target) <= 30/60:
            ts = deadline_et.strftime("%a %b %d • %I:%M %p %Z")

            if alert_type == "Draft":
                emoji = "🔔"
                desc = "Draft transactions"
            else:
                emoji = "⏰"
                desc = "Classic transfers"

            msg = f"{mention}{emoji} FPL **{alert_type}** deadline: {desc} for **GW {gw}** are due in ~**{target}h** (deadline **{ts}**)."
            requests.post(webhook, json={"content": msg}, timeout=10)
            print(f"[waiver_alerts:{alert_type}] Sent {target}h reminder")
            return True

    print(f"[waiver_alerts:{alert_type}] Outside target windows")
    return False


def main():
    # ---- Secrets / env (all provided via GitHub Actions) ----
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "")

    # Draft settings (disabled by default - opt-in via secrets)
    draft_enabled = os.getenv("FPL_DRAFT_ALERTS_ENABLED", "false").lower() in ("true", "1", "yes")
    draft_offset = float(os.getenv("FPL_DEADLINE_OFFSET_HOURS", str(DRAFT_OFFSET_HOURS)))

    # Classic settings (disabled by default - opt-in via secrets)
    classic_enabled = os.getenv("FPL_CLASSIC_ALERTS_ENABLED", "false").lower() in ("true", "1", "yes")
    classic_offset = float(os.getenv("FPL_CLASSIC_DEADLINE_OFFSET_HOURS", str(CLASSIC_OFFSET_HOURS)))

    # Gameweek override
    gw_env = os.getenv("FPL_CURRENT_GAMEWEEK")
    gw = int(gw_env) if gw_env and gw_env.isdigit() else None

    # Optional mentions
    mention_user = os.getenv("DISCORD_MENTION_USER_ID")   # e.g., "123456789012345678"
    mention_role = os.getenv("DISCORD_MENTION_ROLE_ID")   # e.g., "987654321098765432"
    mention = ""
    if mention_user:
        mention += f"<@{mention_user}> "
    if mention_role:
        mention += f"<@&{mention_role}> "

    if not webhook:
        print("[waiver_alerts] Missing DISCORD_WEBHOOK_URL")
        return

    if not draft_enabled and not classic_enabled:
        print("[waiver_alerts] Both Draft and Classic alerts are disabled")
        return

    # Resolve gameweek and kickoff time once (shared between both alert types)
    if gw is None:
        gw = _get_current_gameweek()
    kickoff_et = _earliest_kickoff_et(gw)
    now_et = datetime.now(TZ)

    print(f"[waiver_alerts] GW={gw}")
    print(f"[waiver_alerts] Now: {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Kickoff: {kickoff_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Draft alerts: {'enabled' if draft_enabled else 'disabled'} (offset={draft_offset}h)")
    print(f"[waiver_alerts] Classic alerts: {'enabled' if classic_enabled else 'disabled'} (offset={classic_offset}h)")

    alerts_sent = 0

    # Check Draft deadline
    if draft_enabled:
        draft_deadline = kickoff_et - timedelta(hours=draft_offset)
        if _check_and_send_alert(webhook, mention, draft_deadline, gw, "Draft", now_et):
            alerts_sent += 1

    # Check Classic deadline
    if classic_enabled:
        classic_deadline = kickoff_et - timedelta(hours=classic_offset)
        if _check_and_send_alert(webhook, mention, classic_deadline, gw, "Classic", now_et):
            alerts_sent += 1

    if alerts_sent == 0:
        print("[waiver_alerts] No alerts sent this run")
    else:
        print(f"[waiver_alerts] Sent {alerts_sent} alert(s)")


if __name__ == "__main__":
    main()