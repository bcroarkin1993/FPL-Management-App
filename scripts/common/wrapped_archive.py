# scripts/common/wrapped_archive.py
#
# Local, permanent archive of past-season League Wrapped data — one JSON
# file per season, plus the original source document (e.g. a PDF export)
# if one was used to reconstruct it. Kept separate from league_settings.json
# (that file is for lightweight settings/IDs; this is rich historical
# content, a different kind of data with its own storage).
#
# Two ways a season's file gets here: (1) one-time manual transcription from
# a saved PDF, for seasons that predate this feature (e.g. 2025/26); (2)
# automatically, overwritten on every visit to the live League Wrapped page
# for the current season — so by the time next season's Draft league ID
# rollover makes the old league unreachable, the last live snapshot is
# already safely archived. See save_archived_season() call in
# scripts/draft/league_wrapped.py's show_league_wrapped_page().
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


# ---------------------------------------------------------------------------
# Per-team Season Wrapped ("My Team Wrapped") archive
#
# Same rationale and lifecycle as the League Wrapped archive above (auto-
# saved on every live visit, never auto-deleted), but scoped per team since
# that page is a per-manager narrative rather than a league-wide one. One
# file per (season, team): archive/team_wrapped/{season_safe}/{team_safe}.json
# ---------------------------------------------------------------------------

def _season_dir_name(season: str) -> str:
    """'2025/26' -> '2025_26' (filesystem-safe, no extension)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", season.strip())


def _team_filename(team: str) -> str:
    """'Stoned Squirrels' -> 'Stoned_Squirrels.json' (filesystem-safe)."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", team.strip())
    return f"{safe}.json"


def _team_archive_root() -> Path:
    here = Path(__file__).resolve().parent  # scripts/common/
    repo_root = here.parent.parent
    return repo_root / "archive" / "team_wrapped"


def _team_archive_dir(season: str) -> Path:
    """Locate (and ensure) archive/team_wrapped/{season_safe}/ at the repo root."""
    d = _team_archive_root() / _season_dir_name(season)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_archived_team_seasons(team: str) -> list:
    """Season labels for which this specific team has an archived Season
    Wrapped snapshot, sorted."""
    root = _team_archive_root()
    if not root.exists():
        return []
    fname = _team_filename(team)
    seasons = []
    for season_dir in root.iterdir():
        if not season_dir.is_dir():
            continue
        path = season_dir / fname
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        season = data.get("_season") or season_dir.name.replace("_", "/", 1)
        seasons.append(season)
    return sorted(seasons)


def load_archived_team_season(season: str, team: str) -> Optional[dict]:
    """Load one team's archived Season Wrapped data for one season, or None
    if not found/unreadable."""
    path = _team_archive_dir(season) / _team_filename(team)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_archived_team_season(season: str, team: str, data: dict) -> bool:
    """Save one team's Season Wrapped data for one season. Returns True on success."""
    path = _team_archive_dir(season) / _team_filename(team)
    payload = dict(data)
    payload["_season"] = season
    payload["_team"] = team
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False
