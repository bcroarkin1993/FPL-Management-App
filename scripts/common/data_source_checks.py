# scripts/common/data_source_checks.py
#
# Standalone freshness checks for Rotowire and FFP data sources.
# No Streamlit imports — safe for use in GitHub Actions (waiver_alerts.py).

import logging
import re

import requests

from scripts.common.ffp_feed import (
    fetch_points_predictor,
    fetch_sheet,
    resolve_ffp_gameweek,
)

_logger = logging.getLogger(__name__)

ARTICLES_INDEX = "https://www.rotowire.com/soccer/column/fantasy-premier-league-rankings-188"


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
    # Keep these slug shapes in step with config._discover_rotowire_article(): if this
    # check doesn't recognise the article discovery uses, the "Rotowire published GW N"
    # alert never fires even though the data is there.
    anchors = soup.select(
        'a[href*="fantasy-premier-league-player-rankings-gameweek-"], '
        'a[href*="/soccer/article/fpl-gw"], '
        'a[href*="/soccer/article/fpl-gameweek-"]'
    )

    patterns = [
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"),
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?$"),
        # New format (GW33+): fpl-gwNN-...
        re.compile(r"/soccer/article/fpl-gw(\d+)-[a-z0-9-]+(?:-(\d+))?$"),
        # Preseason/weekly single-GW slug: fpl-gameweek-N-...-gwN-<id>
        re.compile(r"/soccer/article/fpl-gameweek-(\d+)-[a-z0-9-]*gw\1-(\d+)$"),
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
    """Has FFP published projections for this gameweek?

    Answered from FFP's own stated gameweek, falling back to reconstructing it
    from the spreadsheet's fixtures.

    The check this replaces compared the *set* of teams in FFP's fixtures
    against the set playing this gameweek and passed at 50% overlap. Every club
    plays every gameweek, so that set never changes: it returned True for GW2,
    GW3 and GW4 alike, and announced "FFP GW3 projections are now available"
    against a table that was still on GW2 — burning the once-per-gameweek alert
    guard so the real publication went unannounced.
    """
    try:
        rows, feed_gw, _updated = fetch_points_predictor()
        if rows and feed_gw is not None:
            return int(feed_gw) == int(gw)
    except Exception as e:
        _logger.warning("FFP site check failed: %s", e)

    try:
        sheet = fetch_sheet()
    except Exception as e:
        _logger.warning("FFP sheet check failed: %s", e)
        return False
    if sheet is None or sheet.empty:
        return False

    sheet_gw = resolve_ffp_gameweek(sheet)
    return sheet_gw is not None and int(sheet_gw) == int(gw)
