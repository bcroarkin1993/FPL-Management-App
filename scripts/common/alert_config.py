# scripts/common/alert_config.py
#
# JSON config read/write for alert settings.
# Importable by waiver_alerts.py (GitHub Actions) and settings.py (Streamlit).
# Zero Streamlit imports.

import json
import os
from pathlib import Path

DEFAULT_SETTINGS = {
    "version": 1,
    "discord": {
        "mention_user_id": "",
        "mention_role_id": "",
    },
    "deadline_alerts": {
        "draft": {"enabled": False, "offset_hours": 25.5},
        "classic": {"enabled": False, "offset_hours": 1.5},
    },
    "data_source_alerts": {
        "rotowire": {"enabled": False},
        "ffp": {"enabled": False},
    },
    "alert_state": {
        "last_rotowire_alert_gw": 0,
        "last_ffp_alert_gw": 0,
    },
}


def _find_config_path() -> Path:
    """Locate alert_settings.json at the repo root."""
    # Walk up from this file (scripts/common/) to find project root
    here = Path(__file__).resolve().parent
    for ancestor in [here.parent.parent, here.parent, here]:
        candidate = ancestor / "alert_settings.json"
        if candidate.exists():
            return candidate
    # Default: repo root (two levels up from scripts/common/)
    return here.parent.parent / "alert_settings.json"


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Deep-merge overrides into defaults, filling missing keys from defaults."""
    merged = dict(defaults)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


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


def update_alert_state(source: str, gw: int) -> None:
    """Read-modify-write just the alert_state section."""
    settings = load_settings()
    key = f"last_{source}_alert_gw"
    settings.setdefault("alert_state", {})[key] = gw
    save_settings(settings)
