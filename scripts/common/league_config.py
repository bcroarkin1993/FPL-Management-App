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
        #              "locked": True, "dues": {"Team Name": {"amount_paid": 75, "notes": ""}}}}
        "commish_seasons": {},
    },
    "classic": {
        "leagues": [],  # [{"id": int, "name": str}, ...]
        "team_id": None,
        "team_name": None,
        "locked": False,
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
