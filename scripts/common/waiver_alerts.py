# scripts/waiver_alerts.py  (GitHub Actions-friendly; no config.py imports including from utils.py)

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

TZ = ZoneInfo("America/New_York")

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

def main():
    # ---- Secrets / env (all provided via GitHub Actions) ----
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
    offset_h = float(os.getenv("FPL_DEADLINE_OFFSET_HOURS", "25.5"))
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
        print("Missing DISCORD_WEBHOOK_URL")
        return

    # Compute window
    deadline_et, kickoff_et, gw = get_next_transaction_deadline(offset_hours=offset_h, gw=gw)
    now_et = datetime.now(TZ)
    hours_left = (deadline_et - now_et).total_seconds() / 3600

    # Log timing info for debugging
    print(f"[waiver_alerts] GW={gw}, offset={offset_h}h")
    print(f"[waiver_alerts] Now: {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Kickoff: {kickoff_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Deadline: {deadline_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Hours until deadline: {hours_left:.2f}")

    # Fire at ~24h / 6h / 1h (tolerance ±20 min)
    sent = False
    for target in (24, 6, 1):
        if abs(hours_left - target) <= 20/60:
            ts = deadline_et.strftime("%a %b %d • %I:%M %p %Z")
            msg = f"{mention}🔔 FPL Draft transactions for **GW {gw}** are due in ~**{target}h** (deadline **{ts}**)."
            requests.post(webhook, json={"content": msg}, timeout=10)
            print(f"[waiver_alerts] Sent {target}h reminder")
            sent = True
            break

    if not sent:
        print(f"[waiver_alerts] Outside target windows (hours_left={hours_left:.2f})")

if __name__ == "__main__":
    main()