# config.py — Python 3.9 safe
# Loads env (if python-dotenv is installed), defines app constants,
# and lazily resolves CURRENT_GAMEWEEK and ROTOWIRE_URL on first access (cached).

import logging
import os

# ----- .env loader (optional) -----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ----- Core league/app settings (IDs only; no network here) -----
FPL_DRAFT_LEAGUE_ID   = int(os.getenv("FPL_DRAFT_LEAGUE_ID", "0"))
FPL_DRAFT_TEAM_ID     = int(os.getenv("FPL_DRAFT_TEAM_ID", "0"))
FPL_CLASSIC_TEAM_ID   = int(os.getenv("FPL_CLASSIC_TEAM_ID", "0"))

def _parse_classic_leagues(env_value: str) -> list:
    """
    Parse FPL_CLASSIC_LEAGUE_IDS env var.

    Supports formats:
      - "123456:My League,789012:Friends" -> [{"id": 123456, "name": "My League"}, ...]
      - "123456,789012" -> [{"id": 123456, "name": None}, ...] (names fetched from API later)

    Returns list of dicts with 'id' (int) and 'name' (str or None).
    """
    if not env_value or not env_value.strip():
        return []

    leagues = []
    for entry in env_value.split(","):
        entry = entry.strip()
        if not entry:
            continue

        if ":" in entry:
            # Format: id:name
            parts = entry.split(":", 1)
            try:
                league_id = int(parts[0].strip())
                league_name = parts[1].strip() if len(parts) > 1 else None
                leagues.append({"id": league_id, "name": league_name or None})
            except ValueError:
                continue
        else:
            # Format: id only
            try:
                league_id = int(entry)
                leagues.append({"id": league_id, "name": None})
            except ValueError:
                continue

    return leagues

# Classic FPL leagues - supports multiple leagues
# Format: "id:name,id:name,..." or "id,id,..."
FPL_CLASSIC_LEAGUE_IDS = _parse_classic_leagues(os.getenv("FPL_CLASSIC_LEAGUE_IDS", ""))

# Resolved lazily below:
# CURRENT_GAMEWEEK
# ROTOWIRE_URL

FORM_LOOKBACK_WEEKS   = int(os.getenv("FORM_LOOKBACK_WEEKS", "4"))

# ----- Fixture APIs -----
FPL_FIXTURES_BY_EVENT = os.getenv(
    "FPL_FIXTURES_BY_EVENT",
    "https://fantasy.premierleague.com/api/fixtures/?event={gw}",
)

# ----- RotoWire (index page for discovery; URL is resolved lazily below) -----
ARTICLES_INDEX = os.getenv(
    "ARTICLES_INDEX",
    "https://www.rotowire.com/soccer/column/fantasy-premier-league-rankings-188",
)
ROTOWIRE_LINEUPS_URL = os.getenv(
    "ROTOWIRE_LINEUPS_URL",
    "https://www.rotowire.com/soccer/lineups.php",
)
ROTOWIRE_SEASON_RANKINGS_URL = os.getenv(
    "ROTOWIRE_SEASON_RANKINGS_URL",
    "https://www.rotowire.com/soccer/article/fantasy-premier-league-rankings-top-400-for-2025-26-season-fpl-94580",
)

# ----- Notifications / Discord -----
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ----- Deadlines / offsets -----
# Keep as float to allow "25.5" in env; your code should split hours/minutes when using timedeltas.
try:
    TRANSACTION_DEADLINE_HOURS_BEFORE_KICKOFF = float(os.getenv("FPL_DEADLINE_OFFSET_HOURS", "25.5"))
except ValueError:
    TRANSACTION_DEADLINE_HOURS_BEFORE_KICKOFF = 25.5

# ----- App defaults -----
UPCOMING_WEEKS_DEFAULT = int(os.getenv("UPCOMING_WEEKS_DEFAULT", "3"))
TZ_NAME = os.getenv("TZ_NAME", "America/New_York")

# ----- Runtime caches (set by app at runtime) -----
LEAGUE_DATA = None  # Cached league entries, populated by get_league_teams()
TRANSACTION_DATA = None  # Cached transaction data, populated by get_transaction_data()

# ----- Team colors (fallbacks included) -----
TEAM_COLORS = {
    "Arsenal":          {"primary": "#EF0107", "secondary": "#FFFFFF"},
    "Aston Villa":      {"primary": "#95BFE5", "secondary": "#670E36"},
    "Bournemouth":      {"primary": "#DA291C", "secondary": "#000000"},
    "Brentford":        {"primary": "#E30613", "secondary": "#FFFFFF"},
    "Brighton":         {"primary": "#0057B8", "secondary": "#FFFFFF"},
    "Chelsea":          {"primary": "#034694", "secondary": "#FFFFFF"},
    "Crystal Palace":   {"primary": "#1B458F", "secondary": "#C4122E"},
    "Everton":          {"primary": "#003399", "secondary": "#FFFFFF"},
    "Fulham":           {"primary": "#FFFFFF", "secondary": "#000000"},
    "Ipswich":          {"primary": "#3A64A3", "secondary": "#FFFFFF"},
    "Leicester":        {"primary": "#0053A0", "secondary": "#FFFFFF"},
    "Liverpool":        {"primary": "#C8102E", "secondary": "#FFFFFF"},
    "Man City":         {"primary": "#6CABDD", "secondary": "#1C2C5B"},
    "Man Utd":          {"primary": "#DA291C", "secondary": "#000000"},
    "Newcastle":        {"primary": "#241F20", "secondary": "#FFFFFF"},
    "Nott'm Forest":    {"primary": "#DD0000", "secondary": "#FFFFFF"},
    "Southampton":      {"primary": "#D71920", "secondary": "#FFFFFF"},
    "Spurs":            {"primary": "#FFFFFF", "secondary": "#132257"},
    "West Ham":         {"primary": "#7A263A", "secondary": "#1BB1E7"},
    "Wolves":           {"primary": "#FDB913", "secondary": "#231F20"},
    # Fallbacks
    "Burnley":          {"primary": "#6C1D45", "secondary": "#99D6EA"},
    "Luton":            {"primary": "#F78F1E", "secondary": "#002D62"},
    "Sheffield Utd":    {"primary": "#EE2737", "secondary": "#000000"},
    "Leeds":            {"primary": "#FFCD00", "secondary": "#1D428A"},
}

# =============================================================================
# Lazy attributes for:
#   - CURRENT_GAMEWEEK (FPL game endpoint; env override supported)
#   - ROTOWIRE_URL     (env override or discovered from ARTICLES_INDEX using CURRENT_GAMEWEEK)
# =============================================================================

_GW_CACHE = None        # type: ignore
_RW_URL_CACHE = None    # type: ignore

def _resolve_current_gameweek():
    """Resolve the current gameweek with env override, else FPL Draft API, else fallback to 1."""
    # Optional env override (handy for offline/dev)
    env_gw = os.getenv("FPL_CURRENT_GAMEWEEK", "").strip()
    if env_gw.isdigit():
        return int(env_gw)

    # Query the official endpoint (same logic as your utils.get_current_gameweek)
    import requests  # local import to avoid cost unless used
    try:
        r = requests.get("https://draft.premierleague.com/api/game", timeout=15)
        j = r.json()
        if j.get("current_event_finished"):
            gw = j.get("next_event", 1)
        else:
            gw = j.get("current_event", 1)
        return int(gw or 1)
    except Exception:
        # Offline or error
        logging.getLogger("fpl_app.config").warning(
            "Failed to resolve gameweek from API, defaulting to GW 1", exc_info=True
        )
        return 1

def _discover_rotowire_article(gw: int):
    """Find the best Rotowire rankings article for the given GW from ARTICLES_INDEX."""
    _logger = logging.getLogger("fpl_app.config")

    # If explicitly pinned in env, use that
    pinned = os.getenv("ROTOWIRE_URL", "").strip()
    if pinned:
        _logger.debug("Using pinned ROTOWIRE_URL from environment: %s", pinned)
        return pinned

    import re
    from urllib.parse import urljoin
    import requests
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        _logger.warning("BeautifulSoup not available, cannot discover Rotowire URL")
        return ""

    index_url = ARTICLES_INDEX or "https://www.rotowire.com/soccer/column/fantasy-premier-league-rankings-188"
    try:
        resp = requests.get(index_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        anchors = soup.select('a[href*="fantasy-premier-league-player-rankings-gameweek-"]')

        # Multiple regex patterns for robustness (most specific to least)
        patterns = [
            # Standard format: gameweek-N-articleID
            re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"),
            # Alternate format without trailing article ID
            re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?$"),
        ]

        candidates = []
        for a in anchors:
            href = (a.get("href") or "").strip()
            for pat in patterns:
                m = pat.search(href)
                if m:
                    gw_found = int(m.group(1))
                    # Article ID may not exist in alternate pattern
                    art_id = int(m.group(2)) if len(m.groups()) > 1 and m.group(2) else 0
                    candidates.append((gw_found, art_id, urljoin(index_url, href)))
                    break

        if not candidates:
            _logger.warning(
                "Rotowire URL discovery: No matching articles found. Found %d anchors on page. "
                "HTML structure may have changed.",
                len(anchors)
            )
            return ""

        _logger.debug("Rotowire: Found %d candidate articles for GW %s", len(candidates), gw)

        exact = [c for c in candidates if c[0] == int(gw)]
        if exact:
            result = max(exact, key=lambda x: x[1])[2]
            _logger.debug("Rotowire: Exact GW match found: %s", result)
            return result

        # nearest GW, tie-break by newest article id
        result = min(candidates, key=lambda x: (abs(x[0] - int(gw)), -x[1]))[2]
        closest_gw = min(candidates, key=lambda x: abs(x[0] - int(gw)))[0]
        _logger.info(
            "Rotowire: No exact match for GW %d, using closest GW %d: %s",
            gw, closest_gw, result
        )
        return result
    except Exception:
        _logger.warning(
            "Failed to discover Rotowire article for GW %s, returning empty URL", gw, exc_info=True
        )
        return ""

def __getattr__(name):  # PEP 562: module-level getattr
    global _GW_CACHE, _RW_URL_CACHE

    if name == "CURRENT_GAMEWEEK":
        if _GW_CACHE is None:
            _GW_CACHE = _resolve_current_gameweek()
        return _GW_CACHE

    if name == "ROTOWIRE_URL":
        if _RW_URL_CACHE is None:
            # ensure we have GW without importing utils to avoid circular import
            gw = _GW_CACHE if _GW_CACHE is not None else _resolve_current_gameweek()
            _RW_URL_CACHE = _discover_rotowire_article(gw)
        return _RW_URL_CACHE

    raise AttributeError(name)
