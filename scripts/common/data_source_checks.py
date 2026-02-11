# scripts/common/data_source_checks.py
#
# Standalone freshness checks for Rotowire and FFP data sources.
# No Streamlit imports — safe for use in GitHub Actions (waiver_alerts.py).

import logging
import re
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

ARTICLES_INDEX = "https://www.rotowire.com/soccer/column/fantasy-premier-league-rankings-188"
FFP_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaiTmUKjtQ7MxiGibN2GAZ8m9NHF3IA2U-yE0PhBpCOXHewhs57PrjZO7GQzZvrEGGBW7HFEE43yX0/pub?output=csv"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def get_current_gw_team_short_names(gw: int) -> set:
    """Fetch team short names playing in the given gameweek."""
    try:
        bootstrap = requests.get(BOOTSTRAP_URL, timeout=15).json()
        team_id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

        fixtures = requests.get(FIXTURES_URL, params={"event": gw}, timeout=15).json()
        teams = set()
        for fx in fixtures:
            h, a = fx.get("team_h"), fx.get("team_a")
            if h:
                teams.add(team_id_to_short.get(h, ""))
            if a:
                teams.add(team_id_to_short.get(a, ""))
        teams.discard("")
        return teams
    except Exception as e:
        _logger.warning("Failed to get GW %d team names: %s", gw, e)
        return set()


def is_rotowire_available_for_gw(gw: int) -> bool:
    """Check if Rotowire has published a rankings article for the given gameweek."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(ARTICLES_INDEX, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        _logger.warning("Rotowire check failed: %s", e)
        return False

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.content, "html.parser")
    anchors = soup.select('a[href*="fantasy-premier-league-player-rankings-gameweek-"]')

    patterns = [
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"),
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?$"),
    ]

    for a in anchors:
        href = a.get("href", "").strip()
        if not href:
            continue
        for pat in patterns:
            m = pat.search(href)
            if m and int(m.group(1)) == gw:
                _logger.debug("Rotowire GW %d article found: %s", gw, href)
                return True

    return False


def is_ffp_available_for_gw(gw: int) -> bool:
    """Check if FFP Google Sheet has data for the given gameweek."""
    try:
        resp = requests.get(FFP_SHEET_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        _logger.warning("FFP check failed: %s", e)
        return False

    if df is None or df.empty or "Fixture" not in df.columns:
        return False

    current_teams = get_current_gw_team_short_names(gw)
    if not current_teams:
        return False

    ffp_teams = set()
    for fixture in df["Fixture"].dropna().unique():
        parts = str(fixture).split()
        if parts:
            ffp_teams.add(parts[0].upper())

    if not ffp_teams:
        return False

    overlap = len(ffp_teams & current_teams)
    return overlap >= len(ffp_teams) * 0.5
