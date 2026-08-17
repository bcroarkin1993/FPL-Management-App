# scripts/common/league_config.py
#
# JSON config read/write for Draft/Classic league and team ID settings.
# Set via the in-app "League Setup" admin page, read by config.py.
# Zero Streamlit imports (importable from config.py without a cycle).
#
# Persisted to league_settings.json at the repo root, which is gitignored:
# this repo is public, and these IDs today live only in the gitignored .env,
# so this file must not be committed (unlike alert_settings.json).

import json
from pathlib import Path

from scripts.common.alert_config import _deep_merge

DEFAULT_SETTINGS = {
    "version": 1,
    "draft": {
        "league_id": None,
        "team_id": None,
        "team_name": None,
        "locked": False,
        # Past-season Draft league IDs, captured before they're overwritten by
        # the next season's League Setup — Draft leagues don't carry over.
        "history": [],  # [{"season": "2024/25", "league_id": int, "team_id": int, "team_name": str}, ...]
        # Commissioner dues/payout tracking, keyed by season label ("YYYY/YY")
        # so it survives Draft league rollover (league IDs get reissued each
        # season — see Commish Mode / League Setup):
        # {"2026/27": {"buy_in": 75, "payout_pct": {"1": 60, "2": 30, "3": 10},
        #              "locked": True, "dues": {"Team Name": {"paid": False, "notes": ""}}}}
        "commish_seasons": {},
        # Season label (e.g. "2026/27") stamped by display_pl_season_label() each
        # time Save & Lock is clicked — used to detect a season rollover the user
        # hasn't re-confirmed their league ID for yet (see League Setup's nudge banner).
        "last_confirmed_season": None,
    },
    "classic": {
        "leagues": [],  # [{"id": int, "name": str}, ...]
        "team_id": None,
        "team_name": None,
        "locked": False,
        # Past-season Classic/H2H league IDs, captured automatically when the
        # season completes (or reachability fails) — private H2H mini-leagues in
        # particular get recreated with a new ID some seasons, same rollover
        # problem Draft leagues have. Keyed on (season, league_id) rather than
        # season alone since a season can have multiple concurrent leagues.
        # [{"season": "2025/26", "league_id": int, "league_name": str, "manual_stats": dict|None}, ...]
        "league_history": [],
        "last_confirmed_season": None,
        # Manually-entered per-season notes — currently just % finish, since
        # FPL's live entry-history endpoint (get_classic_team_history) has no
        # total-entrant count to derive a percentile from. Keyed by season
        # label. {"2025/26": {"pct_finish": 8.0}, ...}
        "season_notes": {},
    },
}


def _find_config_path() -> Path:
    """Locate league_settings.json at the repo root."""
    here = Path(__file__).resolve().parent
    for ancestor in [here.parent.parent, here.parent, here]:
        candidate = ancestor / "league_settings.json"
        if candidate.exists():
            return candidate
    # Default: repo root (two levels up from scripts/common/)
    return here.parent.parent / "league_settings.json"


def load_settings() -> dict:
    """Read JSON config, deep-merge with defaults for missing keys."""
    path = _find_config_path()
    if not path.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return _deep_merge(DEFAULT_SETTINGS, data)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> bool:
    """Write settings to JSON config. Returns True on success."""
    path = _find_config_path()
    try:
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False


def auto_archive_completed_season() -> bool:
    """Best-effort: once the PL season has concluded, snapshot any locked
    Draft/Classic league IDs into history so they're preserved before next
    season's rollover makes the old IDs stop resolving (Draft leagues are
    recreated every season; private Classic/H2H mini-leagues sometimes are
    too — see league_setup.py's reachability backstop for the case where a
    league goes stale before this ever gets to run).

    Safe to call on every session start: no-ops while the season is still in
    progress, and is idempotent once a season's already archived. Local
    imports avoid pulling `config`/streamlit into this module's import graph
    (see module docstring — must stay importable from config.py without a
    cycle).

    Returns True if anything was written.
    """
    try:
        from scripts.common.fpl_draft_api import is_season_complete
        if not is_season_complete():
            return False
    except Exception:
        return False

    import config as _config

    settings = load_settings()
    season_label = _config.display_pl_season_label()
    changed = False

    draft = settings.setdefault("draft", {})
    if draft.get("locked") and draft.get("league_id"):
        history = draft.setdefault("history", [])
        if not any(h.get("season") == season_label for h in history):
            history.append({
                "season": season_label,
                "league_id": draft["league_id"],
                "team_id": draft.get("team_id"),
                "team_name": draft.get("team_name"),
                "manual_stats": None,
            })
            changed = True

    classic = settings.setdefault("classic", {})
    if classic.get("locked") and classic.get("leagues"):
        history = classic.setdefault("league_history", [])
        archived_ids = {h.get("league_id") for h in history if h.get("season") == season_label}
        for league in classic["leagues"]:
            league_id = league.get("id")
            if league_id not in archived_ids:
                history.append({
                    "season": season_label,
                    "league_id": league_id,
                    "league_name": league.get("name"),
                    "manual_stats": None,
                })
                changed = True

    if changed:
        save_settings(settings)
    return changed
