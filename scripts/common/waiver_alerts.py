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

# Where accepted trades need approval, trade offers close a full day before the
# waiver deadline to leave room for the approval window.
TRADE_APPROVAL_LEAD_HOURS = 24.0

# `league.trades` codes whose meaning is confirmed. Deliberately a local copy of the
# table in fpl_draft_api.py: that module imports Streamlit and this one must stay
# importable from GitHub Actions. Only administrator approval has ever been verified
# against a real league — see "Draft Transaction Rules" in CLAUDE.md.
TRADE_SETTING_ADMIN_APPROVAL = "a"
TRADE_SETTINGS_REQUIRING_APPROVAL = frozenset({TRADE_SETTING_ADMIN_APPROVAL})
TRADE_SETTINGS_DISABLED = frozenset()   # the "no trades" code has not been observed
#: Every code whose meaning is established. Anything outside this is as good as
#: unread — it must not fall through to the "no approval needed" branch and buy
#: itself a later deadline on a code we cannot interpret.
KNOWN_TRADE_SETTINGS = TRADE_SETTINGS_REQUIRING_APPROVAL | TRADE_SETTINGS_DISABLED


def resolve_trade_deadline(gw, kickoff_et, trades_code, now_et):
    """(gw, deadline, approval_assumed) for the next *reachable* trade deadline.

    Returns `(gw, None, False)` where trading is disabled.

    **Why this needs a look-ahead when the waiver alert does not.** main() resolves
    one gameweek — the current one until its last match finishes — and derives every
    deadline from that gameweek's earliest kickoff. Under approval, trades close
    49.5h before kickoff, and for a *midweek* gameweek all three alert windows
    (73.5h, 55.5h and 50.5h before kickoff) fall while the previous gameweek is still
    being played. So `gw` is still N, `kickoff_et` is GW N's kickoff in the past,
    every window reads as elapsed, and the alert silently never fires. Worked
    through for a Tuesday 15:00 kickoff: the windows land Sat 13:30, Sun 07:30 and
    Sun 12:30, with GW N running until Sunday evening.

    The Draft waiver alert survives the same arithmetic only by luck — its 25.5h
    deadline leaves the 6h and 1h windows after the rollover. The extra day of lead
    a trade deadline carries pushes all of its windows into the dead zone. 7 of 38
    gameweeks are midweek, so this is a seventh of the season, not an edge case.
    """
    deadline, assumed = trade_deadline_for(kickoff_et, trades_code)
    if deadline is None or deadline > now_et:
        return gw, deadline, assumed

    # This gameweek's trade window has already shut. The one that matters now is the
    # next gameweek's, which is what a manager would actually be planning for.
    try:
        next_kickoff = _earliest_kickoff_et(gw + 1)
    except RuntimeError as e:
        print(f"[waiver_alerts:Trade] No fixtures for GW {gw + 1} ({e})")
        return gw, deadline, assumed

    next_deadline, next_assumed = trade_deadline_for(next_kickoff, trades_code)
    if next_deadline is not None and next_deadline > now_et:
        print(f"[waiver_alerts:Trade] GW {gw} trade window has shut; "
              f"targeting GW {gw + 1}")
        return gw + 1, next_deadline, next_assumed
    return gw, deadline, assumed


def _resolve_draft_league_id():
    """Draft league id for the notifier: locked in-app setting first, then the env.

    The same precedence config.py applies, reimplemented here because this module is
    deliberately config-free for GitHub Actions. league_config is Streamlit-free and
    is already this module's transitive dependency via alert_config.

    Reading the env alone meant a user who configured their league on the League
    Setup page rather than in .env got "your league's trade setting could not be
    read" on every locally-run alert, with the setting sitting readable on disk.
    """
    try:
        from scripts.common.league_config import load_settings as load_league_settings
        draft = load_league_settings().get("draft", {})
        if draft.get("locked") and draft.get("league_id"):
            return str(draft["league_id"])
    except Exception as e:
        print(f"[waiver_alerts] Could not read local league settings ({e})")
    return os.getenv("FPL_DRAFT_LEAGUE_ID")


def _fetch_league_trades_setting(league_id):
    """Read `league.trades` for a Draft league, or None if it cannot be read.

    A plain requests call rather than fpl_draft_api.get_draft_transaction_window(),
    which is Streamlit-cached and would drag Streamlit into the Actions runtime.
    """
    if not league_id:
        return None
    try:
        r = requests.get(
            f"https://draft.premierleague.com/api/league/{league_id}/details",
            timeout=20,
        )
        r.raise_for_status()
        return (r.json().get("league") or {}).get("trades")
    except (requests.RequestException, ValueError, AttributeError) as e:
        print(f"[waiver_alerts] Could not read league trade setting ({e})")
        return None


def trade_deadline_for(kickoff_et, trades_code):
    """(deadline, approval_assumed) for trade offers, or (None, _) if trades are off.

    Returns `approval_assumed=True` when the league's setting could not be read and
    the earlier deadline was used anyway. That is the safe direction — an alert a day
    early is a mild annoyance, an alert after the window has shut is the failure this
    exists to prevent — but the message has to say it was an assumption rather than
    state a deadline this league never confirmed.
    """
    if trades_code in TRADE_SETTINGS_DISABLED:
        return None, False

    waiver_deadline = kickoff_et - timedelta(hours=DRAFT_OFFSET_HOURS)
    if trades_code in TRADE_SETTINGS_REQUIRING_APPROVAL:
        return waiver_deadline - timedelta(hours=TRADE_APPROVAL_LEAD_HOURS), False
    if trades_code not in KNOWN_TRADE_SETTINGS:
        # Unread, or read but unrecognised — both are "we do not know this league".
        return waiver_deadline - timedelta(hours=TRADE_APPROVAL_LEAD_HOURS), True
    # A recognised code that does not require approval: trades close with waivers.
    return waiver_deadline, False


def _get_current_gameweek():
    """Fetch current/next GW from the official Draft endpoint. Returns None if season has ended
    or the endpoint is unavailable/unparseable (e.g. during off-season maintenance windows)."""
    try:
        r = requests.get("https://draft.premierleague.com/api/game", timeout=20)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[waiver_alerts] Could not fetch/parse current gameweek ({e}) — treating as off-season, skipping")
        return None
    gw = data["next_event"] if data.get("current_event_finished") else data["current_event"]
    return int(gw) if gw is not None else None


def _fixtures_for_event(gw: int):
    """Canonical fixtures endpoint; query by GW via params to avoid caching issues."""
    try:
        r = requests.get("https://fantasy.premierleague.com/api/fixtures/", params={"event": int(gw)}, timeout=20)
        r.raise_for_status()
        js = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[waiver_alerts] Could not fetch/parse fixtures for GW {gw} ({e})")
        return []
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
    note: str = "",
) -> bool:
    """
    Check if we're in an alert window and send notification if so.

    Args:
        webhook: Discord webhook URL
        mention: Mention string (user/role pings)
        deadline_et: The deadline datetime
        gw: Gameweek number
        alert_type: "Draft", "Classic" or "Trade"
        now_et: Current time in ET
        alert_windows: List of hours-before-deadline to fire alerts (e.g. [24, 6, 1])
        note: Appended to the message. Used to say when a deadline rests on an
            assumption rather than on the league's stated setting.

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
            elif alert_type == "Trade":
                emoji = "\U0001f500"
                desc = "Draft trade offers"
            else:
                emoji = "\u23f0"
                desc = "Classic transfers"

            msg = f"{mention}{emoji} FPL **{alert_type}** deadline: {desc} for **GW {gw}** are due in ~**{target}h** (deadline **{ts}**)."
            if note:
                msg += f" {note}"
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

    trade_cfg = dl.get("trade", {})
    trade_enabled = trade_cfg.get("enabled", False) or os.getenv("FPL_TRADE_ALERTS_ENABLED", "false").lower() in ("true", "1", "yes")
    trade_windows = trade_cfg.get("alert_windows", [24, 6, 1])

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

    any_enabled = draft_enabled or classic_enabled or trade_enabled or rotowire_enabled or ffp_enabled
    if not any_enabled:
        print("[waiver_alerts] All alerts are disabled")
        return

    # Gameweek override
    gw_env = os.getenv("FPL_CURRENT_GAMEWEEK")
    if gw_env and gw_env.isdigit():
        gw = int(gw_env)
    else:
        gw = _get_current_gameweek()
        if gw is None:
            print("[waiver_alerts] Season has ended (no active gameweek) — skipping all alerts")
            return

    try:
        kickoff_et = _earliest_kickoff_et(gw)
    except RuntimeError as e:
        print(f"[waiver_alerts] {e} — no fixtures scheduled yet, skipping all alerts")
        return
    now_et = datetime.now(TZ)

    print(f"[waiver_alerts] GW={gw}")
    print(f"[waiver_alerts] Now: {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Kickoff: {kickoff_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"[waiver_alerts] Draft alerts: {'enabled' if draft_enabled else 'disabled'} (windows={draft_windows})")
    print(f"[waiver_alerts] Classic alerts: {'enabled' if classic_enabled else 'disabled'} (windows={classic_windows})")
    print(f"[waiver_alerts] Trade alerts: {'enabled' if trade_enabled else 'disabled'} (windows={trade_windows})")
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

    # Check the Draft trade deadline. Distinct from the Draft waiver deadline: where
    # the league requires approval, trade offers shut a full day earlier.
    if trade_enabled:
        trades_code = _fetch_league_trades_setting(_resolve_draft_league_id())
        trade_gw, trade_deadline, approval_assumed = resolve_trade_deadline(
            gw, kickoff_et, trades_code, now_et
        )
        if trade_deadline is None:
            print("[waiver_alerts:Trade] League has trading disabled, skipping")
        else:
            print(f"[waiver_alerts:Trade] league.trades={trades_code!r} "
                  f"approval_assumed={approval_assumed} target_gw={trade_gw}")
            note = (
                "_(Your league's trade setting could not be read, so this assumes "
                "approval is required — the earlier of the two possible deadlines.)_"
                if approval_assumed else ""
            )
            if _check_and_send_alert(webhook, mention, trade_deadline, trade_gw,
                                     "Trade", now_et, trade_windows, note=note):
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
