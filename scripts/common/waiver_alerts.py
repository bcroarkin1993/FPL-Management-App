# scripts/common/waiver_alerts.py  (GitHub Actions-friendly; no config.py imports including from utils.py)
#
# Supports both Draft and Classic FPL alerts:
#   - Draft: 25.5h before kickoff (waiver/transaction deadline)
#   - Classic: 1.5h before kickoff (transfer deadline)
# Also supports data source alerts:
#   - Rotowire: notifies when GW rankings article is published
#   - FFP: notifies when Fantasy Football Pundit updates for current GW

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

from scripts.common.alert_config import load_settings, update_alert_state

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
    alert_windows: list = None,
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
        alert_windows: List of hours-before-deadline to fire alerts (e.g. [24, 6, 1])

    Returns:
        True if an alert was sent, False otherwise
    """
    if alert_windows is None:
        alert_windows = [24, 6, 1]

    hours_left = (deadline_et - now_et).total_seconds() / 3600

    # Log timing info
    print(f"[waiver_alerts:{alert_type}] Deadline: {deadline_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts:{alert_type}] Hours until deadline: {hours_left:.2f}")
    print(f"[waiver_alerts:{alert_type}] Alert windows: {sorted(alert_windows, reverse=True)}")

    # Skip if deadline has passed
    if hours_left < 0:
        print(f"[waiver_alerts:{alert_type}] Deadline has passed, skipping")
        return False

    # Tolerance ±30 min to accommodate GitHub Actions scheduling delays
    for target in sorted(alert_windows, reverse=True):
        if abs(hours_left - target) <= 30/60:
            ts = deadline_et.strftime("%a %b %d • %I:%M %p %Z")

            if alert_type == "Draft":
                emoji = "\U0001f514"
                desc = "Draft transactions"
            else:
                emoji = "\u23f0"
                desc = "Classic transfers"

            msg = f"{mention}{emoji} FPL **{alert_type}** deadline: {desc} for **GW {gw}** are due in ~**{target}h** (deadline **{ts}**)."
            requests.post(webhook, json={"content": msg}, timeout=10)
            print(f"[waiver_alerts:{alert_type}] Sent {target}h reminder")
            return True

    print(f"[waiver_alerts:{alert_type}] Outside target windows")
    return False


def _check_data_source_alerts(webhook: str, mention: str, gw: int, settings: dict, kickoff_et: datetime) -> int:
    """
    Check if Rotowire/FFP data is available for the current GW and send alerts.

    Only sends alerts before the GW has started (before earliest kickoff).
    Each source is alerted at most once per GW via persistent state.

    Returns the number of alerts sent.
    """
    from scripts.common.data_source_checks import (
        is_rotowire_available_for_gw,
        is_ffp_available_for_gw,
    )

    now_et = datetime.now(TZ)

    # Never send data source alerts after the GW has started
    if now_et >= kickoff_et:
        print(f"[waiver_alerts:DataSource] GW {gw} has already started, skipping data source alerts")
        return 0

    ds_settings = settings.get("data_source_alerts", {})
    # Re-read state from disk to avoid stale in-memory values
    from scripts.common.alert_config import load_settings as _reload
    state = _reload().get("alert_state", {})
    alerts_sent = 0

    # Rotowire check
    if ds_settings.get("rotowire", {}).get("enabled", False):
        last_gw = state.get("last_rotowire_alert_gw", 0)
        if last_gw < gw:
            print(f"[waiver_alerts:Rotowire] Checking for GW {gw} data (last alert: GW {last_gw})")
            if is_rotowire_available_for_gw(gw):
                msg = f"{mention}\U0001f4ca **Rotowire** GW {gw} player rankings are now available!"
                requests.post(webhook, json={"content": msg}, timeout=10)
                update_alert_state("rotowire", gw)
                print(f"[waiver_alerts:Rotowire] Sent GW {gw} data alert")
                alerts_sent += 1
            else:
                print(f"[waiver_alerts:Rotowire] GW {gw} data not yet available")
        else:
            print(f"[waiver_alerts:Rotowire] Already alerted for GW {gw}")

    # FFP check
    if ds_settings.get("ffp", {}).get("enabled", False):
        last_gw = state.get("last_ffp_alert_gw", 0)
        if last_gw < gw:
            print(f"[waiver_alerts:FFP] Checking for GW {gw} data (last alert: GW {last_gw})")
            if is_ffp_available_for_gw(gw):
                msg = f"{mention}\U0001f4ca **Fantasy Football Pundit** GW {gw} projections are now available!"
                requests.post(webhook, json={"content": msg}, timeout=10)
                update_alert_state("ffp", gw)
                print(f"[waiver_alerts:FFP] Sent GW {gw} data alert")
                alerts_sent += 1
            else:
                print(f"[waiver_alerts:FFP] GW {gw} data not yet available")
        else:
            print(f"[waiver_alerts:FFP] Already alerted for GW {gw}")

    return alerts_sent


def main():
    # ---- Secrets / env (all provided via GitHub Actions) ----
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "")

    if not webhook:
        print("[waiver_alerts] Missing DISCORD_WEBHOOK_URL")
        return

    # Load JSON config (with defaults for missing keys)
    settings = load_settings()

    # Resolve settings: JSON config first, env var fallback
    dl = settings.get("deadline_alerts", {})
    draft_cfg = dl.get("draft", {})
    classic_cfg = dl.get("classic", {})

    draft_enabled = draft_cfg.get("enabled", False) or os.getenv("FPL_DRAFT_ALERTS_ENABLED", "false").lower() in ("true", "1", "yes")
    draft_windows = draft_cfg.get("alert_windows", [24, 6, 1])

    classic_enabled = classic_cfg.get("enabled", False) or os.getenv("FPL_CLASSIC_ALERTS_ENABLED", "false").lower() in ("true", "1", "yes")
    classic_windows = classic_cfg.get("alert_windows", [24, 6, 1])

    # Data source alert settings (JSON only, no env var fallback)
    ds_settings = settings.get("data_source_alerts", {})
    rotowire_enabled = ds_settings.get("rotowire", {}).get("enabled", False)
    ffp_enabled = ds_settings.get("ffp", {}).get("enabled", False)

    # Mention settings: JSON config first, env var fallback
    discord_cfg = settings.get("discord", {})
    mention_user = discord_cfg.get("mention_user_id", "") or os.getenv("DISCORD_MENTION_USER_ID", "")
    mention_role = discord_cfg.get("mention_role_id", "") or os.getenv("DISCORD_MENTION_ROLE_ID", "")
    mention = ""
    if mention_user:
        mention += f"<@{mention_user}> "
    if mention_role:
        mention += f"<@&{mention_role}> "

    any_enabled = draft_enabled or classic_enabled or rotowire_enabled or ffp_enabled
    if not any_enabled:
        print("[waiver_alerts] All alerts are disabled")
        return

    # Gameweek override
    gw_env = os.getenv("FPL_CURRENT_GAMEWEEK")
    gw = int(gw_env) if gw_env and gw_env.isdigit() else _get_current_gameweek()

    kickoff_et = _earliest_kickoff_et(gw)
    now_et = datetime.now(TZ)

    print(f"[waiver_alerts] GW={gw}")
    print(f"[waiver_alerts] Now: {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Kickoff: {kickoff_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Draft alerts: {'enabled' if draft_enabled else 'disabled'} (windows={draft_windows})")
    print(f"[waiver_alerts] Classic alerts: {'enabled' if classic_enabled else 'disabled'} (windows={classic_windows})")
    print(f"[waiver_alerts] Rotowire data alerts: {'enabled' if rotowire_enabled else 'disabled'}")
    print(f"[waiver_alerts] FFP data alerts: {'enabled' if ffp_enabled else 'disabled'}")

    alerts_sent = 0

    # Check Draft deadline (fixed at 25.5h before kickoff)
    if draft_enabled:
        draft_deadline = kickoff_et - timedelta(hours=DRAFT_OFFSET_HOURS)
        if _check_and_send_alert(webhook, mention, draft_deadline, gw, "Draft", now_et, draft_windows):
            alerts_sent += 1

    # Check Classic deadline (fixed at 1.5h before kickoff)
    if classic_enabled:
        classic_deadline = kickoff_et - timedelta(hours=CLASSIC_OFFSET_HOURS)
        if _check_and_send_alert(webhook, mention, classic_deadline, gw, "Classic", now_et, classic_windows):
            alerts_sent += 1

    # Check data source alerts
    if rotowire_enabled or ffp_enabled:
        alerts_sent += _check_data_source_alerts(webhook, mention, gw, settings, kickoff_et)

    if alerts_sent == 0:
        print("[waiver_alerts] No alerts sent this run")
    else:
        print(f"[waiver_alerts] Sent {alerts_sent} alert(s)")


if __name__ == "__main__":
    main()
