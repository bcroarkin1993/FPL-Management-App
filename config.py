# config.py — Python 3.9 safe
# Loads env (if python-dotenv is installed), defines app constants,
# and lazily resolves CURRENT_GAMEWEEK, ROTOWIRE_URL, and the Draft/Classic
# league & team IDs on first access (cached; league/team IDs additionally
# resolve from league_settings.json, set via the in-app League Setup page).

import logging
import os
import time

# ----- .env loader (optional) -----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ----- Core league/app settings -----
# FPL_DRAFT_LEAGUE_ID, FPL_DRAFT_TEAM_ID, FPL_CLASSIC_TEAM_ID, and
# FPL_CLASSIC_LEAGUE_IDS are resolved lazily below (see __getattr__), so that
# values saved via the League Setup admin page take effect immediately
# without restarting the app.

def _parse_draft_league_history(env_value: str) -> list:
    """
    Parse FPL_DRAFT_LEAGUE_HISTORY env var.

    Format: "YYYY/YY:league_id,YYYY/YY:league_id"
    Example: "2023/24:123456,2024/25:789012"

    Returns list of (season_label, league_id) tuples, sorted by season_label.
    """
    if not env_value or not env_value.strip():
        return []

    seasons = []
    for entry in env_value.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        parts = entry.split(":", 1)
        try:
            label = parts[0].strip()
            league_id = int(parts[1].strip())
            seasons.append((label, league_id))
        except (ValueError, IndexError):
            continue

    return sorted(seasons, key=lambda x: x[0])


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
# Resolved lazily below (see __getattr__): FPL_CLASSIC_LEAGUE_IDS

# Past Draft league IDs for cross-season history (Season Wrapped).
# Format: "YYYY/YY:league_id,YYYY/YY:league_id"
# Resolved lazily below (see __getattr__): FPL_DRAFT_LEAGUE_HISTORY — merges this
# env var with league_settings.json's draft.history (set via League Setup),
# which takes priority per season label.

# Resolved lazily below:
# CURRENT_GAMEWEEK
# FPL_DRAFT_LEAGUE_ID, FPL_DRAFT_TEAM_ID, FPL_CLASSIC_TEAM_ID, FPL_CLASSIC_LEAGUE_IDS
# FPL_DRAFT_LEAGUE_HISTORY
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
    "https://www.rotowire.com/soccer/article/fantasy-premier-league-fpl-rankings-top-400-for-2026-27-season-124261",
)
# GW1's preseason "best picks" article uses a different slug shape than the
# regular weekly rankings articles (see get_rotowire_rankings_url), so it
# isn't auto-discoverable via ARTICLES_INDEX for GW1 specifically — pin it
# like the season rankings URL above. Only relevant for the Initial Squad
# Optimizer; re-pin each preseason.
ROTOWIRE_GW1_URL = os.getenv(
    "ROTOWIRE_GW1_URL",
    "https://www.rotowire.com/soccer/article/fpl-gameweek-1-best-players-captain-picks-2026-27-rankings-gw1-127487",
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

# ----- Team colors -----
# Intentionally append-only across seasons: teams no longer in the Premier
# League stay listed (rather than being deleted) in case they're promoted
# back in a future season — costs nothing, they just won't appear in the
# current season's fixtures/rosters. Newly promoted teams get added here too.
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
    # Promoted/relegated in past or upcoming seasons — kept for whenever they're back
    "Burnley":          {"primary": "#6C1D45", "secondary": "#99D6EA"},
    "Luton":            {"primary": "#F78F1E", "secondary": "#002D62"},
    "Sheffield Utd":    {"primary": "#EE2737", "secondary": "#000000"},
    "Leeds":            {"primary": "#FFCD00", "secondary": "#1D428A"},
    "Coventry City":    {"primary": "#78D0F2", "secondary": "#000000"},
    "Hull City":        {"primary": "#F18A01", "secondary": "#000000"},
    "Sunderland":       {"primary": "#EB172F", "secondary": "#FFFFFF"},
}

# =============================================================================
# Lazy attributes for:
#   - CURRENT_GAMEWEEK (FPL game endpoint; env override supported)
#   - ROTOWIRE_URL     (env override or discovered from ARTICLES_INDEX using CURRENT_GAMEWEEK)
#   - FPL_DRAFT_LEAGUE_ID, FPL_DRAFT_TEAM_ID, FPL_CLASSIC_TEAM_ID,
#     FPL_CLASSIC_LEAGUE_ID (league_settings.json via the League Setup admin
#     page, when locked; falls back to the .env-based behavior otherwise)
# =============================================================================

# Cache TTL in seconds (5 minutes for gameweek, allows refresh during gameweek transitions)
_GW_CACHE_TTL = 300

# Cache storage: (value, timestamp) tuples for TTL-based expiration
_GW_CACHE = None        # type: ignore  # (gameweek: int, cached_at: float)
_RW_URL_CACHE = None    # type: ignore  # (url: str, for_gw: int)

# League/team ID settings loaded once per process; cleared by refresh_league_settings()
_LEAGUE_SETTINGS_CACHE = None  # type: ignore  # dict from league_config.load_settings()


def _get_league_settings():
    """Load (and cache) the local league_settings.json admin settings, if any."""
    global _LEAGUE_SETTINGS_CACHE
    if _LEAGUE_SETTINGS_CACHE is None:
        from scripts.common.league_config import load_settings
        _LEAGUE_SETTINGS_CACHE = load_settings()
    return _LEAGUE_SETTINGS_CACHE


def refresh_league_settings():
    """Clear the cached league/team ID settings so the next access re-reads
    league_settings.json. Call this after saving via the League Setup page."""
    global _LEAGUE_SETTINGS_CACHE
    _LEAGUE_SETTINGS_CACHE = None


def _env_int(name: str) -> int:
    """Read an integer env var, tolerating a key that is present but empty.

    `int(os.getenv(NAME, "0"))` looks safe and is not: the default only applies
    when the key is *absent*, so a `NAME=` line with nothing after it reaches
    int("") and raises. That fires on the unlocked-settings fallback path only,
    which is the least convenient moment to discover it.
    """
    return int(os.getenv(name, "0") or 0)


def _resolve_draft_league_id():
    settings = _get_league_settings()
    draft = settings.get("draft", {})
    if draft.get("locked") and draft.get("league_id"):
        return int(draft["league_id"])
    return _env_int("FPL_DRAFT_LEAGUE_ID")


def _resolve_draft_team_id():
    settings = _get_league_settings()
    draft = settings.get("draft", {})
    if draft.get("locked") and draft.get("team_id"):
        return int(draft["team_id"])
    return _env_int("FPL_DRAFT_TEAM_ID")


def _resolve_classic_team_id():
    settings = _get_league_settings()
    classic = settings.get("classic", {})
    if classic.get("locked") and classic.get("team_id"):
        return int(classic["team_id"])
    return _env_int("FPL_CLASSIC_TEAM_ID")


def _resolve_classic_league_ids():
    settings = _get_league_settings()
    classic = settings.get("classic", {})
    if classic.get("locked") and classic.get("leagues"):
        return [{"id": int(l["id"]), "name": l.get("name")} for l in classic["leagues"]]
    return _parse_classic_leagues(os.getenv("FPL_CLASSIC_LEAGUE_IDS", ""))


def _resolve_draft_league_history():
    """List of (season_label, league_id) tuples, sorted by season.

    Merges league_settings.json's draft.history (set via the League Setup
    page's history section) with the legacy FPL_DRAFT_LEAGUE_HISTORY env var,
    so nothing already configured in .env is lost. JSON wins on a season-label
    collision; the env var only fills in seasons not already saved in JSON.
    """
    settings = _get_league_settings()
    history = settings.get("draft", {}).get("history", [])
    merged = {
        h["season"]: int(h["league_id"])
        for h in history if h.get("season") and h.get("league_id")
    }
    for season_label, league_id in _parse_draft_league_history(os.getenv("FPL_DRAFT_LEAGUE_HISTORY", "")):
        merged.setdefault(season_label, league_id)
    return sorted(merged.items(), key=lambda x: x[0])


def get_draft_league_history_records():
    """Full merged draft league history records (not collapsed to tuples).

    Unlike FPL_DRAFT_LEAGUE_HISTORY (season, league_id pairs only), this
    preserves each record's manual_stats — final rank/points/W-D-L saved
    directly for a season whose league ID is no longer usable for a live
    lookup (Draft league IDs get reissued to unrelated leagues once a
    season rolls over, so old IDs can't be re-queried after the fact).
    JSON wins on a season-label collision; the env var only fills in
    seasons not already saved in JSON (env-derived records always have
    manual_stats=None, since the env var only ever stores a league_id).
    """
    settings = _get_league_settings()
    history = settings.get("draft", {}).get("history", [])
    by_season = {h["season"]: dict(h) for h in history if h.get("season")}
    for season_label, league_id in _parse_draft_league_history(os.getenv("FPL_DRAFT_LEAGUE_HISTORY", "")):
        by_season.setdefault(season_label, {
            "season": season_label, "league_id": league_id,
            "team_id": None, "team_name": None, "manual_stats": None,
        })
    return sorted(by_season.values(), key=lambda h: h["season"])


def get_classic_league_history_records():
    """Full Classic/H2H league history records, keyed on (season, league_id)
    rather than season alone since Classic supports multiple concurrent
    leagues — a season can have more than one record. Populated via League
    Setup's automatic season-rollover archiving (or its manual "Classic
    League History" fallback section); there is no legacy env-var source for
    this (unlike Draft's FPL_DRAFT_LEAGUE_HISTORY)."""
    settings = _get_league_settings()
    history = settings.get("classic", {}).get("league_history", [])
    return sorted(
        (dict(h) for h in history if h.get("season")),
        key=lambda h: (h["season"], h.get("league_id") or 0),
    )


def get_classic_season_notes() -> dict:
    """Manually-entered per-season notes for Classic — currently just
    pct_finish, since FPL's live entry-history endpoint
    (get_classic_team_history) has no total-entrant count to derive a
    percentile from. Populated via League Setup's Classic League History
    section. Keyed by season label ('YYYY/YY')."""
    settings = _get_league_settings()
    return dict(settings.get("classic", {}).get("season_notes", {}))


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
            next_ev = j.get("next_event")
            if next_ev is None:
                # Season complete — next_event is null after GW38 finishes.
                # Return the last played GW (not +1) so display contexts never show GW39.
                gw = int(j.get("current_event") or 38)
            else:
                gw = next_ev
        else:
            gw = j.get("current_event", 1)
        return int(gw or 1)
    except Exception:
        # Offline or error
        logging.getLogger("fpl_app.config").warning(
            "Failed to resolve gameweek from API, defaulting to GW 1", exc_info=True
        )
        return 1

def current_pl_season_str():
    """Best-effort current Premier League season as 'YYYY-YY' (e.g. '2026-27'),
    derived from today's date. The PL season runs Aug-May; treat Jul 1 as the
    rollover point so pre-season browsing already reflects the upcoming season."""
    from datetime import date
    today = date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_pl_season_label():
    """Current Premier League season as 'YYYY/YY' for display (e.g. '2026/27')."""
    return current_pl_season_str().replace("-", "/")


def display_pl_season_label():
    """The Premier League season currently in progress, or (during the
    off-season gap) the one that most recently concluded, as 'YYYY/YY'.

    Unlike current_pl_season_str() — used for pre-season Rotowire content
    discovery, which deliberately rolls over on July 1st, ahead of the real
    season boundary — this uses the actual Aug-May season window, so
    mid-season and post-season UI (Season Wrapped, League Wrapped, PDF
    exports, the "season concluded" banner) shows the correct year without
    being pulled forward before the new season has actually started.
    """
    from datetime import date
    today = date.today()
    end_year = today.year if today.month <= 8 else today.year + 1
    return f"{end_year - 1}/{str(end_year)[-2:]}"


def _extract_season_from_href(href: str):
    """Pull a 'YYYY-YY' season token out of a Rotowire URL slug, if present.
    Most gameweek-specific articles omit the season; preseason/season-long
    articles (e.g. '...-2026-27-season...') include it."""
    import re
    m = re.search(r"(?<!\d)(20\d{2})-(\d{2})(?!\d)", href)
    return f"{m.group(1)}-{m.group(2)}" if m else None


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
    current_season = current_pl_season_str()
    try:
        resp = requests.get(index_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        anchors = soup.select(
            'a[href*="fantasy-premier-league-player-rankings-gameweek-"], '
            'a[href*="/soccer/article/fpl-gw"], '
            'a[href*="/soccer/article/fpl-gameweek-"], '
            'a[href*="best-fpl-picks-for-gameweeks-"]'
        )

        # Multiple regex patterns for robustness (most specific to least)
        patterns = [
            # Preseason/weekly single-GW slug, e.g.
            # fpl-gameweek-1-best-players-captain-picks-2026-27-rankings-gw1-127487
            # The \1 backreference forces the trailing gwNN to agree with the leading
            # gameweek-NN, which also keeps the Fantrax variant
            # (fantrax-sleeper-...-gameweek-1-gw1-...) out via the /fpl-gameweek- anchor.
            re.compile(r"/soccer/article/fpl-gameweek-(\d+)-[a-z0-9-]*gw\1-(\d+)$"),
            # Old format: ...gameweek-NN-...-ARTICLEID
            re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"),
            # New format (GW33+): fpl-gwNN-...-ARTICLEID
            re.compile(r"/soccer/article/fpl-gw(\d+)-[a-z0-9-]+-(\d+)$"),
            # Old format without article ID (fallback)
            re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?$"),
            # New format without article ID (fallback)
            re.compile(r"/soccer/article/fpl-gw(\d+)-[a-z0-9-]+$"),
        ]
        # Pre-season "best picks for gameweeks X-Y" range articles carry an explicit
        # season token, e.g. best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238
        # These are NEVER a valid projection source: the column is headed "Adj Total"
        # and holds an adjusted value metric accumulated over the whole range, not
        # projected points for a gameweek. Dividing it down would only make a
        # fabricated number look plausible. They are matched here solely to harvest
        # their article id for the current-season floor below.
        range_pattern = re.compile(
            r"/soccer/article/best-fpl-picks-for-gameweeks-(\d+)-(\d+)-fantasy-premier-league-(\d{4}-\d{2})(?:-[a-z0-9-]+)?-(\d+)$"
        )

        # candidates: (gw_found, art_id, url, season_or_None) — single-gameweek
        # rankings articles only. all_article_ids: every season-tagged article id seen on the page, regardless
        # of whether it's selectable for this GW, so the season floor (below) reflects
        # the whole page rather than only articles that happen to cover this GW.
        candidates = []
        current_season_ids = []
        for a in anchors:
            href = (a.get("href") or "").strip()

            rm = range_pattern.search(href)
            if rm:
                season, art_id = rm.group(3), int(rm.group(4))
                if season == current_season:
                    current_season_ids.append(art_id)
                continue  # season floor only -- never a candidate

            for pat in patterns:
                m = pat.search(href)
                if m:
                    gw_found = int(m.group(1))
                    # Article ID may not exist in alternate pattern
                    art_id = int(m.group(2)) if len(m.groups()) > 1 and m.group(2) else 0
                    season = _extract_season_from_href(href)
                    if season == current_season and art_id:
                        current_season_ids.append(art_id)
                    candidates.append((gw_found, art_id, urljoin(index_url, href), season))
                    break

        if not candidates:
            _logger.warning(
                "Rotowire URL discovery: No matching articles found. Found %d anchors on page. "
                "HTML structure may have changed.",
                len(anchors)
            )
            return ""

        _logger.debug("Rotowire: Found %d candidate articles for GW %s", len(candidates), gw)

        # Season-tagged candidates from a prior season are never eligible — an old
        # article being the "closest GW" match is a false positive, not a fallback.
        # Most per-GW articles don't tag their season in the URL at all, so also use
        # article ID as a proxy: Rotowire IDs increase monotonically over time, so
        # once we know the lowest ID published for the current season anywhere on the
        # page, any untagged candidate below that floor predates the season rollover
        # and is stale too.
        season_floor_id = min(current_season_ids) if current_season_ids else None

        def _is_stale(c):
            _gw, art_id, _url, season = c
            if season == current_season:
                return False
            if season is not None:
                return True  # explicitly tagged as a different season
            if season_floor_id is not None and art_id and art_id < season_floor_id:
                return True  # untagged, but older than the earliest known current-season article
            return False

        fresh_candidates = [c for c in candidates if not _is_stale(c)]
        if not fresh_candidates:
            _logger.warning(
                "Rotowire URL discovery: all %d candidates predate the current season "
                "(expected %s); returning empty URL instead of stale data.",
                len(candidates), current_season,
            )
            return ""
        candidates = fresh_candidates

        exact = [c for c in candidates if c[0] == int(gw)]
        if exact:
            # Prefer a current-season-tagged exact match over an untagged one, then newest article id.
            result = max(exact, key=lambda x: (x[3] == current_season, x[1]))[2]
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

def _is_gw_cache_stale():
    """Check if gameweek cache is stale (older than TTL)."""
    if _GW_CACHE is None:
        return True
    _, cached_at = _GW_CACHE
    return (time.time() - cached_at) > _GW_CACHE_TTL


def refresh_gameweek():
    """
    Force refresh the gameweek cache.
    Call this when you need to ensure fresh gameweek data.
    Returns the new gameweek value.
    """
    global _GW_CACHE, _RW_URL_CACHE
    _GW_CACHE = (_resolve_current_gameweek(), time.time())
    # Also clear Rotowire URL cache since it depends on gameweek
    _RW_URL_CACHE = None
    return _GW_CACHE[0]


def __getattr__(name):  # PEP 562: module-level getattr
    global _GW_CACHE, _RW_URL_CACHE

    if name == "CURRENT_GAMEWEEK":
        if _is_gw_cache_stale():
            _GW_CACHE = (_resolve_current_gameweek(), time.time())
        return _GW_CACHE[0]

    if name == "ROTOWIRE_URL":
        # Get current gameweek (may refresh if stale)
        current_gw = __getattr__("CURRENT_GAMEWEEK")

        # Check if we have a cached URL for the current gameweek
        if _RW_URL_CACHE is None or _RW_URL_CACHE[1] != current_gw:
            url = _discover_rotowire_article(current_gw)
            _RW_URL_CACHE = (url, current_gw)
        return _RW_URL_CACHE[0]

    if name == "FPL_DRAFT_LEAGUE_ID":
        return _resolve_draft_league_id()

    if name == "FPL_DRAFT_TEAM_ID":
        return _resolve_draft_team_id()

    if name == "FPL_CLASSIC_TEAM_ID":
        return _resolve_classic_team_id()

    if name == "FPL_CLASSIC_LEAGUE_IDS":
        return _resolve_classic_league_ids()

    if name == "FPL_DRAFT_LEAGUE_HISTORY":
        return _resolve_draft_league_history()

    raise AttributeError(name)
