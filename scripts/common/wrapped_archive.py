# scripts/common/wrapped_archive.py
#
# Local, permanent archive of past-season League Wrapped data — one JSON
# file per season, plus the original source document (e.g. a PDF export)
# if one was used to reconstruct it. Kept separate from league_settings.json
# (that file is for lightweight settings/IDs; this is rich historical
# content, a different kind of data with its own storage).
#
# Persisted under archive/league_wrapped/ at the repo root, which is
# gitignored: this repo is public, and this data includes personal league
# info (team/manager names). Files placed here are never auto-deleted by
# app code — this module never removes a season once saved.

import json
import re
from pathlib import Path
from typing import Optional


def _archive_dir() -> Path:
    """Locate (and ensure) archive/league_wrapped/ at the repo root."""
    here = Path(__file__).resolve().parent  # scripts/common/
    repo_root = here.parent.parent
    d = repo_root / "archive" / "league_wrapped"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _season_filename(season: str) -> str:
    """'2025/26' -> '2025_26.json' (filesystem-safe)."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", season.strip())
    return f"{safe}.json"


def list_archived_seasons() -> list:
    """Season labels ('YYYY/YY') with archived data, sorted."""
    d = _archive_dir()
    seasons = []
    for path in d.glob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        season = data.get("_season") or path.stem.replace("_", "/", 1)
        seasons.append(season)
    return sorted(seasons)


def load_archived_season(season: str) -> Optional[dict]:
    """Load one season's archived League Wrapped data, or None if not found/unreadable."""
    path = _archive_dir() / _season_filename(season)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_archived_season(season: str, data: dict) -> bool:
    """Save one season's archived League Wrapped data. Returns True on success."""
    path = _archive_dir() / _season_filename(season)
    payload = dict(data)
    payload["_season"] = season
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False
