"""
Web Scraping & External Data Source Functions.

Rotowire scraping, Fantasy Football Pundit data, and The Odds API integration.
"""

import os
import hashlib
import threading
import re
from datetime import datetime
from typing import NamedTuple

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import streamlit as st
from typing import Optional
from urllib.parse import urljoin

import config
from scripts.common.error_helpers import get_logger
from scripts.common.text_helpers import TZ_ET, format_last_updated
from scripts.common import ffp_feed

_logger = get_logger("fpl_app.scraping")


# =============================================================================
# ROTOWIRE SCRAPING
# =============================================================================

# Rotowire publishes at least two distinct table shapes under the same CSS
# classes: the standard weekly rankings table (12 columns, includes
# per-position rank breakdown and a Matchup/TSB% column) and a preseason
# "best picks" preview table (6 columns: Rank, Player, Team, Pos, Price,
# Adj Total -- no per-position ranks, no matchup). A fixed column-count/order
# assumption silently drops every row of whichever shape it wasn't written
# for. Since both shapes include a real <thead> with clear column names,
# map by header text instead -- resilient to Rotowire adding, removing, or
# reordering columns, not just these two known shapes.
_ROTOWIRE_HEADER_ALIASES = {
    'Overall Rank': ('overall rank', 'rank'),
    'FW Rank': ('fw rank',),
    'MID Rank': ('mid rank',),
    'DEF Rank': ('def rank',),
    'GK Rank': ('gk rank',),
    'Player': ('player',),
    'Team': ('team',),
    'Matchup': ('matchup',),
    'Position': ('pos', 'position'),
    'Price': ('price',),
    'TSB %': ('tsb%', 'tsb %'),
    'Points': ('pts', 'points', 'adj total'),
}


def _map_rotowire_header_row(header_cells):
    """Map a <thead> row's header text to canonical field -> column index."""
    lower_headers = [h.strip().lower() for h in header_cells]
    mapping = {}
    for canonical, aliases in _ROTOWIRE_HEADER_ALIASES.items():
        for alias in aliases:
            if alias in lower_headers:
                mapping[canonical] = lower_headers.index(alias)
                break
    return mapping


def _rotowire_row_from_header_map(cells, header_map, safe_numeric):
    """Build a canonical row dict from a <td> row using a header->index map.
    Returns None if the row can't be resolved to at least Player + Team
    (e.g. a stray header/subtitle row inside <tbody>)."""
    def get(field, default=""):
        idx = header_map.get(field)
        return cells[idx] if idx is not None and idx < len(cells) else default

    player, team = get('Player'), get('Team')
    if not player or not team:
        return None

    return {
        'Overall Rank': get('Overall Rank'),
        'FW Rank': safe_numeric(get('FW Rank')),
        'MID Rank': safe_numeric(get('MID Rank')),
        'DEF Rank': safe_numeric(get('DEF Rank')),
        'GK Rank': safe_numeric(get('GK Rank')),
        'Player': player,
        'Team': team,
        'Matchup': get('Matchup'),
        'Position': get('Position'),
        'Price': safe_numeric(get('Price')),
        'TSB %': get('TSB %'),
        'Points': safe_numeric(get('Points')),
    }


# Rotowire's "best picks for gameweeks X-Y" articles are not a projection source at
# all. Their points column is headed "Adj Total" and holds an adjusted value metric
# accumulated over the whole range -- not projected points for any single gameweek.
# Scaling it down by the number of gameweeks would not recover a projection, it would
# just make a fabricated number look plausible, so these are refused outright.
# config._discover_rotowire_article() already declines to select them; this is the
# backstop for a URL pinned by hand via ROTOWIRE_URL.
_ROTOWIRE_RANGE_ARTICLE_RE = re.compile(r"best-fpl-picks-for-gameweeks-(\d+)-(\d+)-")

# Above this, a "single gameweek" projection table is not credible (the highest realistic
# single-GW Rotowire projection is well under 10) — used as a tripwire, not a correction.
_ROTOWIRE_MAX_PLAUSIBLE_MEDIAN = 10


@st.cache_data(ttl=3600)
def get_rotowire_player_projections(url, limit=None):
    """
    Fetches fantasy rankings and projected points for players from RotoWire.

    Parameters:
    - url (str): URL to fetch the data from.
    - limit (int, optional): Number of players to display. Defaults to None (displays all players).

    Returns:
    - DataFrame: A DataFrame containing player rankings, projected points, and calculated value.
                 Returns empty DataFrame on error.
    """
    LEGACY_EXPECTED_COLUMNS = 12  # Fallback positional layout, used only if no <thead> is found

    # Helper to safely convert to numeric
    def _safe_numeric(val, default=0):
        if val is None:
            return default
        s = str(val).strip()
        if s in {"#N/A", "N/A", "", "-", "—"}:
            return default
        s = re.sub(r"[£$,%]", "", s)  # Strip currency/formatting
        s = s.replace("\u200b", "").replace("\xa0", "").strip()
        try:
            return float(s)
        except ValueError:
            return default

    # Download the page
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        _logger.warning("Failed to fetch Rotowire projections from %s: %s", url, e)
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find table with fallback selectors (most specific to least specific)
    table = soup.select_one("table.article-table__tablesorter.article-table__standard.article-table__figure")
    if table is None:
        table = soup.select_one("table.article-table__tablesorter")
        if table:
            _logger.info("Rotowire: Using fallback table selector (article-table__tablesorter)")
    if table is None:
        table = soup.select_one("table.ck-table-resized")
        if table:
            _logger.info("Rotowire: Using ck-table-resized selector")
    if table is None:
        table = soup.find("table")
        if table:
            _logger.info("Rotowire: Using generic table selector")
    if table is None:
        _logger.warning("Rotowire: Could not locate any table on page %s", url)
        return pd.DataFrame()

    # Extract rows from table body or table directly
    try:
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")
    except AttributeError as e:
        _logger.warning("Rotowire: Error extracting table rows: %s", e)
        return pd.DataFrame()

    thead = table.find("thead")
    header_map = {}
    if thead:
        header_cells = [th.get_text(strip=True) for th in thead.find_all("th")]
        header_map = _map_rotowire_header_row(header_cells)
        if 'Player' not in header_map or 'Team' not in header_map:
            _logger.warning(
                "Rotowire: <thead> found but couldn't resolve Player/Team columns from %s -- "
                "falling back to legacy positional parsing.",
                header_cells,
            )
            header_map = {}

    # Parse each row
    data = []
    skipped_rows = 0
    for tr in rows:
        tds = tr.find_all("td")
        if not tds:
            skipped_rows += 1
            continue
        cells = [td.get_text(strip=True) for td in tds]

        if header_map:
            row_data = _rotowire_row_from_header_map(cells, header_map, _safe_numeric)
            if row_data is None:
                skipped_rows += 1
                continue
            data.append(row_data)
            continue

        # Legacy fallback: fixed 12-column positional layout (no <thead> on the page)
        if len(cells) < LEGACY_EXPECTED_COLUMNS:
            skipped_rows += 1
            continue
        cells = cells[:LEGACY_EXPECTED_COLUMNS]
        try:
            row_data = {
                'Overall Rank': cells[0],
                'FW Rank': _safe_numeric(cells[1]),
                'MID Rank': _safe_numeric(cells[2]),
                'DEF Rank': _safe_numeric(cells[3]),
                'GK Rank': _safe_numeric(cells[4]),
                'Player': cells[5],
                'Team': cells[6],
                'Matchup': cells[7],
                'Position': cells[8],
                'Price': _safe_numeric(cells[9]),
                'TSB %': cells[10],
                'Points': _safe_numeric(cells[11]),
            }
            data.append(row_data)
        except IndexError as e:
            _logger.warning("Rotowire: Error parsing row, skipping: %s", e)
            skipped_rows += 1
            continue

    if skipped_rows > 0:
        _logger.debug("Rotowire: Skipped %d rows with unexpected structure", skipped_rows)

    if not data:
        _logger.warning("Rotowire: No valid player data extracted from %s", url)
        return pd.DataFrame()

    # Create DataFrame
    player_rankings = pd.DataFrame(data)

    # Refuse multi-gameweek range articles outright -- see _ROTOWIRE_RANGE_ARTICLE_RE.
    # Returning nothing surfaces the app's existing "projections unavailable" warning,
    # which is a far better outcome than plausible-looking invented numbers.
    if _ROTOWIRE_RANGE_ARTICLE_RE.search(url or ""):
        _logger.error(
            "Rotowire: %s is a multi-gameweek 'best picks' article, not a gameweek "
            "projection table (its column is an adjusted value total, not points). "
            "Refusing to use it. Pin a single-gameweek rankings article via "
            "ROTOWIRE_URL, or leave it unset to let discovery find one.",
            url,
        )
        return pd.DataFrame()

    # Tripwire for the next slug shape Rotowire invents: if the table still doesn't look
    # like single-gameweek data, say so loudly rather than silently inflating every score.
    _median_points = player_rankings['Points'].median()
    if pd.notna(_median_points) and _median_points > _ROTOWIRE_MAX_PLAUSIBLE_MEDIAN:
        _logger.warning(
            "Rotowire: median Points is %.1f for %s -- implausible for a single gameweek. "
            "This article may be a multi-GW or season-long table; check ROTOWIRE_URL discovery.",
            _median_points, url,
        )

    # Create 'Pos Rank' by summing the four position ranks
    player_rankings['Pos Rank'] = (
        player_rankings['FW Rank'] + player_rankings['MID Rank'] +
        player_rankings['DEF Rank'] + player_rankings['GK Rank']
    ).astype(int)

    # Drop individual position rank columns
    player_rankings.drop(columns=['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank'], inplace=True)

    # Create the 'Value' column by dividing 'Points' by 'Price'
    player_rankings['Value'] = player_rankings.apply(
        lambda row: row['Points'] / row['Price'] if row['Price'] > 0 else float('nan'), axis=1
    )

    # If a limit is provided, return only the top 'limit' players
    if limit:
        player_rankings = player_rankings.head(limit)

    # Format the DataFrame to remove the index and reset it with a starting value of 1
    player_rankings.reset_index(drop=True, inplace=True)
    player_rankings.index = player_rankings.index + 1

    _logger.debug("Rotowire: Successfully parsed %d players from %s", len(player_rankings), url)
    return player_rankings


# Rotowire stamps every article with "Updated on August 20, 2026 10:54AM EST" in
# a div.article__date. Worth surfacing: a weekly rankings table published before
# the last team-news cycle is materially less useful than one published after it,
# and nothing else on the page tells you which you are looking at.
_ROTOWIRE_UPDATED_RE = re.compile(
    r"Updated\s+on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AaPp][Mm])",
)


@st.cache_data(ttl=1800)
def get_rotowire_article_updated(url: str, timeout: int = 15):
    """Scrape an article's "Updated on ..." timestamp.

    Rotowire labels the time "EST" year-round; it is really wall-clock New York
    time, so it is localized to America/New_York rather than a fixed offset.

    Args:
        url: Rotowire article URL.
        timeout: Request timeout in seconds.

    Returns:
        A timezone-aware datetime, or None if the page is unreachable or the
        stamp is missing. Never raises -- a missing timestamp is cosmetic and
        must not take a page down.
    """
    if not url:
        return None
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        node = soup.find("div", class_="article__date")
        text = node.get_text(" ", strip=True) if node else soup.get_text(" ", strip=True)

        match = _ROTOWIRE_UPDATED_RE.search(" ".join(text.split()))
        if not match:
            _logger.info("Rotowire: no 'Updated on' stamp found at %s", url)
            return None

        stamp = re.sub(r"\s+", " ", match.group(1)).replace(" AM", "AM").replace(" PM", "PM")
        parsed = datetime.strptime(stamp, "%B %d, %Y %I:%M%p")
        return parsed.replace(tzinfo=TZ_ET)
    except Exception as e:
        _logger.warning("Rotowire: could not read update time from %s: %s", url, e)
        return None


def get_rotowire_rankings_url(current_gameweek=None, timeout=15):
    """
    Try to locate the Rotowire 'Fantasy Premier League Player Rankings: Gameweek X'
    article on the /soccer/articles/ index. Handles new slugs with extra words.

    Returns:
        str | None  -> fully qualified article URL or None if not found.
    """
    from scripts.common.fpl_draft_api import get_current_gameweek

    # If you have a helper, use it; otherwise leave current_gameweek optional
    if current_gameweek is None:
        try:
            current_gameweek = get_current_gameweek()
        except Exception:
            current_gameweek = None

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(config.ARTICLES_INDEX, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        _logger.warning("Rotowire URL discovery failed - could not fetch articles index: %s", e)
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # Find any anchors whose href contains our base slug (matches both old and new fpl-gwNN format)
    anchors = soup.select(
        'a[href*="fantasy-premier-league-player-rankings-gameweek-"], '
        'a[href*="/soccer/article/fpl-gw"]'
    )

    # Regex patterns to try (most specific to least specific)
    patterns = [
        # Old format: ...gameweek-NN-...-ARTICLEID
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"),
        # New format (GW33+): fpl-gwNN-...-ARTICLEID
        re.compile(r"/soccer/article/fpl-gw(\d+)-[a-z0-9-]+-(\d+)$"),
        # Old format without article ID (fallback)
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?$"),
        # New format without article ID (fallback)
        re.compile(r"/soccer/article/fpl-gw(\d+)-[a-z0-9-]+$"),
    ]

    candidates = []
    for a in anchors:
        href = a.get("href", "").strip()
        if not href:
            continue

        for pat in patterns:
            m = pat.search(href)
            if m:
                gw = int(m.group(1))
                # Article ID may not exist in alternate pattern
                art_id = int(m.group(2)) if len(m.groups()) > 1 and m.group(2) else 0
                candidates.append((gw, art_id, urljoin(config.ARTICLES_INDEX, href)))
                break  # Don't try other patterns if one matched

    if not candidates:
        total_anchors = len(soup.find_all("a"))
        _logger.warning(
            "Rotowire URL discovery failed - no matching articles found. "
            "Rotowire may have changed their URL format again. "
            "Fix: update regex patterns in scraping.py and config.py, "
            "or pin ROTOWIRE_URL in .env. "
            "Matched anchors: %d, total anchors on page: %d",
            len(anchors),
            total_anchors,
        )
        webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if webhook:
            msg = (
                "⚠️ **Rotowire URL Discovery Failed** — Rotowire may have changed their article URL format.\n"
                "The app cannot load player projections until this is fixed.\n"
                "**Quick fix**: Add to `.env`:\n"
                "`ROTOWIRE_URL=<paste current article URL here>`\n"
                "Then update the regex patterns in `scraping.py` and `config.py` for future GWs."
            )
            try:
                requests.post(webhook, json={"content": msg}, timeout=10)
            except Exception:
                pass
        return None

    _logger.debug("Rotowire: Found %d candidate articles for GW %s", len(candidates), current_gameweek)

    if current_gameweek is not None:
        # Prefer exact gameweek; if multiple, highest article id
        exact = [c for c in candidates if c[0] == current_gameweek]
        if exact:
            result = max(exact, key=lambda x: x[1])[2]
            _logger.debug("Rotowire: Exact GW match found: %s", result)
            return result

        # Else pick closest GW; break ties by newest article id
        result = min(candidates, key=lambda x: (abs(x[0] - current_gameweek), -x[1]))[2]
        closest_gw = min(candidates, key=lambda x: abs(x[0] - current_gameweek))[0]
        _logger.info(
            "Rotowire: No exact match for GW %d, using closest GW %d: %s",
            current_gameweek, closest_gw, result
        )
        return result

    # If we don't know the GW, return the newest relevant article by id
    result = max(candidates, key=lambda x: x[1])[2]
    _logger.debug("Rotowire: Using newest article (no GW specified): %s", result)
    return result


@st.cache_data(ttl=7200)
def get_rotowire_season_rankings(url: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape Rotowire's season-long FPL rankings table.

    Expected columns (12 per row):
      'Overall Rank', 'FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank',
      'Player', 'Team', 'Position', 'Price', 'TSB %', 'Points', 'PP/90'

    Enhancements:
      - Robust parsing of '#N/A', 'N/A', '-', '—' -> treated as missing
      - Infer Position from which of the rank columns has a valid rank if Position is missing/#N/A
      - Default Price to 4.5 if missing/nonpositive
      - Default TSB % to 0.0 if missing
      - Compute Pos Rank (sum of positional ranks) and Value (Points/Price)
      - Index starts at 1
    """
    # ---- Fetch & parse page ----
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    table = soup.select_one("table.article-table__tablesorter.article-table__standard.article-table__figure")
    if table is None:
        table = soup.select_one("table.article-table__tablesorter")
    if table is None:
        table = soup.select_one("table.ck-table-resized")
    if table is None:
        table = soup.find("table")
    if table is None:
        raise ValueError("Could not locate a rankings table on the page.")

    # ---- Helpers ----
    def _to_float(x):
        if x is None:
            return np.nan
        s = str(x).strip()
        if s in {"#N/A", "N/A", "", "-", "—"}:
            return np.nan
        s = re.sub(r"[£$,%]", "", s)
        s = s.replace("\u200b", "").replace("\xa0", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    def _to_int(x):
        val = _to_float(x)
        if np.isnan(val):
            return np.nan
        return int(round(val))

    def _normalize_pos_text(txt):
        if pd.isna(txt):
            return np.nan
        s = str(txt).upper().strip()
        if s in {"F", "FW", "FWD", "FORWARD"}: return "F"
        if s in {"M", "MID", "MIDFIELDER"}:    return "M"
        if s in {"D", "DEF", "DEFENDER"}:      return "D"
        if s in {"G", "GK", "GKP", "GOALKEEPER"}: return "G"
        if s in {"#N/A", "N/A", "", "-", "—"}: return np.nan
        return s

    def _infer_position(row):
        ranks = {
            "F": row.get("FW Rank"),
            "M": row.get("MID Rank"),
            "D": row.get("DEF Rank"),
            "G": row.get("GK Rank"),
        }
        valid = {k: v for k, v in ranks.items() if pd.notna(v) and v > 0}
        if not valid:
            return np.nan
        return min(valid, key=valid.get)  # best (lowest) rank wins

    # ---- Extract rows ----
    rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
    data = []
    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) != 12:
            continue
        cells = [td.get_text(strip=True) for td in tds]
        data.append({
            "Overall Rank": cells[0],
            "FW Rank":      cells[1],
            "MID Rank":     cells[2],
            "DEF Rank":     cells[3],
            "GK Rank":      cells[4],
            "Player":       cells[5],
            "Team":         cells[6],
            "Position":     cells[7],
            "Price":        cells[8],
            "TSB %":        cells[9],
            "Points":       cells[10],
            "PP/90":        cells[11],
        })

    if not data:
        raise ValueError("No ranking rows found; table structure may have changed.")

    df = pd.DataFrame(data)

    # ---- Type coercion ----
    for col in ["FW Rank", "MID Rank", "DEF Rank", "GK Rank", "Points", "PP/90", "Price"]:
        df[col] = df[col].apply(_to_float)
    df["TSB %"] = df["TSB %"].apply(_to_float)
    df["Overall Rank"] = df["Overall Rank"].apply(_to_int)

    # Normalize provided Position text (if any)
    df["Position"] = df["Position"].apply(_normalize_pos_text)

    # ---- Infer Position where missing/#N/A ----
    missing_pos_mask = df["Position"].isna()
    if missing_pos_mask.any():
        df.loc[missing_pos_mask, "Position"] = df[missing_pos_mask].apply(_infer_position, axis=1)

    # ---- Defaults ----
    df["Price"] = df["Price"].apply(lambda x: 4.5 if (pd.isna(x) or x <= 0) else x)
    df["TSB %"] = df["TSB %"].fillna(0.0)

    # ---- Derived metrics ----
    df["Pos Rank"] = (
        df[["FW Rank", "MID Rank", "DEF Rank", "GK Rank"]]
        .fillna(0)
        .sum(axis=1)
        .round()
        .astype(int)
    )
    df["Value"] = df.apply(
        lambda r: (r["Points"] / r["Price"]) if (pd.notna(r["Points"]) and r["Price"] > 0) else np.nan,
        axis=1
    )

    # ---- Optional limiting ----
    if limit:
        if df["Overall Rank"].notna().any():
            df = df.sort_values(["Overall Rank", "Player"], na_position="last").head(limit)
        else:
            df = df.sort_values("Points", ascending=False, na_position="last").head(limit)

    # ---- Final cleanup ----
    df = df.reset_index(drop=True)
    df.index = df.index + 1

    desired_cols = [
        "Overall Rank", "FW Rank", "MID Rank", "DEF Rank", "GK Rank",
        "Player", "Team", "Position", "Price", "TSB %", "Points", "PP/90",
        "Pos Rank", "Value"
    ]
    df = df[[c for c in desired_cols if c in df.columns]]

    return df


# =============================================================================
# FANTASY FOOTBALL PUNDIT DATA
# =============================================================================

#: Every FFP address now lives in ``ffp_feed`` -- re-exported here so the
#: existing importers keep working.
FFP_SHEET_URL = ffp_feed.FFP_SHEET_URL

#: The human-readable page behind the feed. The site payload is the data source;
#: this is where a person goes to read it, so it is what the UI should link to.
FFP_POINTS_PREDICTOR_URL = ffp_feed.FFP_POINTS_PREDICTOR_URL
FFP_GOAL_ASSIST_URL = ffp_feed.FFP_GOAL_ASSIST_URL
FFP_CLEAN_SHEET_URL = ffp_feed.FFP_CLEAN_SHEET_URL


class FFPFeed(NamedTuple):
    """An FFP table plus everything needed to judge whether to trust it.

    ``gameweek`` and ``provenance`` are the point, not a nicety. The failure
    this exists to prevent is a table that is wrong but looks right: FFP rolls
    its numbers forward on its own clock, and consuming last week's -- or next
    week's -- projections under this week's heading is invisible, because every
    individual value is plausible. Same reasoning as ``resolve_classic_squad()``.
    """

    df: Optional[pd.DataFrame]
    gameweek: Optional[int]
    updated: Optional[datetime]
    provenance: str                     # "site" | "sheet" | "none"
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.df.empty

    def is_stale(self, expected_gw: Optional[int]) -> bool:
        """True when the feed describes a gameweek other than the one asked for.

        An unknown gameweek is *not* stale -- it is unknown. Refusing to blend on
        "we could not tell" would take FFP off every page whenever the fixture
        vote is inconclusive.
        """
        if expected_gw is None or self.gameweek is None:
            return False
        return int(self.gameweek) != int(expected_gw)


@st.cache_data(ttl=300, show_spinner=False)
def get_ffp_feed() -> FFPFeed:
    """Fetch FFP, site first, published Google Sheet second.

    The sheet was the app's only source until FFP stopped keeping it in step
    with their own site -- measured a full gameweek behind, and internally
    inconsistent within a single download. It stays as a fallback so a change to
    FFP's frontend degrades the feed rather than removing it.

    **A failure is never cached as a success.** Returning a bare ``None`` from a
    ``cache_data`` function pinned a transient timeout for five minutes, which is
    how FFP came to read "temporarily unavailable" in the app while the website
    was plainly working. On failure this returns ``provenance="none"`` with the
    reason in ``note``, and ``ok`` is False, so a caller can say what went wrong.
    """
    try:
        rows, gw, updated = ffp_feed.fetch_points_predictor()
        if rows:
            code_to_id = ffp_feed.bootstrap_code_to_id()
            df = ffp_feed.to_sheet_schema(rows, gw, code_to_id)
            if df is not None and not df.empty:
                goal_rows, _, _ = ffp_feed.fetch_goal_assist()
                cs_rows, _, _ = ffp_feed.fetch_clean_sheet()
                df = ffp_feed.attach_odds_columns(df, goal_rows, cs_rows, gw)
                return FFPFeed(df, gw, updated, "site")
    except Exception as e:
        _logger.warning("FFP site feed failed, falling back to the sheet: %s", e)

    try:
        sheet = ffp_feed.fetch_sheet()
    except Exception as e:
        _logger.warning("FFP sheet fetch failed: %s", e)
        sheet = None

    if sheet is None or sheet.empty:
        return FFPFeed(None, None, None, "none",
                       "Neither the Fantasy Football Pundit site nor its published "
                       "spreadsheet could be read.")

    sheet_gw = ffp_feed.resolve_ffp_gameweek(sheet)
    if sheet_gw is not None:
        sheet = sheet.copy()
        sheet["FFP_GW"] = sheet_gw
    return FFPFeed(sheet, sheet_gw, None, "sheet",
                   "Read from FFP's published spreadsheet; their site payload was "
                   "unavailable.")


def get_ffp_projections_data() -> Optional[pd.DataFrame]:
    """Fetch Fantasy Football Pundit projections.

    Returns a DataFrame in the long-standing sheet schema -- Name, Team,
    Position, Fixture, Ownership, Start, Price, CS, AnytimeGoal/Assist/Return,
    Predicted, StartingPredicted, GW2-GW6 and Next2GWs-Next6GWs -- plus
    ``FFP_GW``, ``Player_Code`` and ``Player_ID`` when the site payload was
    reachable. None if no source could be read.

    Prefer :func:`get_ffp_feed` in new code: this returns the table without the
    gameweek it belongs to, and that gameweek is the thing that made the numbers
    wrong.
    """
    return get_ffp_feed().df


def get_ffp_goalscorer_odds() -> Optional[pd.DataFrame]:
    """
    Get FFP anytime goalscorer odds data.

    Returns DataFrame with goal/assist probabilities.
    """
    df = get_ffp_projections_data()
    if df is None:
        return None

    cols = {
        'Name': 'Player',
        'Team': 'Team',
        'Position': 'Position',
        'Fixture': 'Fixture',
        'Start': 'Start %',
        'AnytimeGoal': 'Goal %',
        'AnytimeAssist': 'Assist %',
        'AnytimeReturn': 'Return %',
    }

    available = {k: v for k, v in cols.items() if k in df.columns}
    result = df[list(available.keys())].rename(columns=available)

    # Filter to players with goal probability > 0
    if 'Goal %' in result.columns:
        result = result[result['Goal %'] > 0].copy()
        result = result.sort_values('Goal %', ascending=False)

    return result.reset_index(drop=True)


def get_ffp_clean_sheet_odds() -> Optional[pd.DataFrame]:
    """
    Get FFP clean sheet odds aggregated by team.

    Returns DataFrame with team-level CS probabilities.
    """
    df = get_ffp_projections_data()
    if df is None:
        return None

    # Aggregate by team (CS is the same for all players on a team)
    if 'CS' not in df.columns or 'Team' not in df.columns:
        return None

    # Get unique team entries with their fixtures
    team_df = df.groupby('Team').agg({
        'CS': 'first',
        'Fixture': 'first',
    }).reset_index()

    team_df = team_df.rename(columns={'CS': 'CS Prob %'})
    team_df = team_df.sort_values('CS Prob %', ascending=False)

    return team_df.reset_index(drop=True)


# =============================================================================
# THE ODDS API INTEGRATION
# =============================================================================

@st.cache_data(ttl=300)
def get_odds_api_match_odds(api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch EPL match odds from The Odds API.

    Args:
        api_key: The Odds API key (falls back to env var ODDS_API_KEY)

    Returns DataFrame with columns:
    - Home Team, Away Team, Kickoff
    - Home Win %, Draw %, Away Win % (converted from decimal odds)
    - BTTS Yes %, Over 2.5 %
    """
    import os
    key = api_key or os.getenv("ODDS_API_KEY", "")
    if not key:
        _logger.debug("No ODDS_API_KEY configured")
        return None

    base_url = "https://api.the-odds-api.com/v4/sports/soccer_epl"

    try:
        # Fetch h2h odds
        h2h_resp = requests.get(
            f"{base_url}/odds",
            params={"apiKey": key, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"},
            timeout=15
        )
        h2h_resp.raise_for_status()
        h2h_data = h2h_resp.json()

        if not h2h_data:
            return None

        matches = []
        for match in h2h_data:
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            kickoff = match.get("commence_time", "")

            # Get average odds across bookmakers
            h2h_odds = {"home": [], "draw": [], "away": []}
            for bm in match.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            if name == home:
                                h2h_odds["home"].append(price)
                            elif name == away:
                                h2h_odds["away"].append(price)
                            elif name == "Draw":
                                h2h_odds["draw"].append(price)

            # Convert average odds to implied probability
            def odds_to_prob(odds_list):
                if not odds_list:
                    return None
                avg_odds = sum(odds_list) / len(odds_list)
                return round((1 / avg_odds) * 100, 1) if avg_odds > 0 else None

            matches.append({
                "Home Team": home,
                "Away Team": away,
                "Kickoff": kickoff[:16].replace("T", " ") if kickoff else "",
                "Home Win %": odds_to_prob(h2h_odds["home"]),
                "Draw %": odds_to_prob(h2h_odds["draw"]),
                "Away Win %": odds_to_prob(h2h_odds["away"]),
            })

        df = pd.DataFrame(matches)
        _logger.debug("Odds API: fetched %d matches", len(df))
        return df

    except requests.exceptions.RequestException as e:
        _logger.warning("Failed to fetch Odds API data: %s", str(e))
        return None
    except Exception as e:
        _logger.warning("Error processing Odds API data: %s", str(e))
        return None


@st.cache_data(ttl=300)
def get_odds_api_match_details(event_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """
    Fetch detailed odds for a specific match including BTTS and totals.

    Args:
        event_id: The Odds API event ID
        api_key: API key (falls back to env var)

    Returns dict with detailed odds data.
    """
    import os
    key = api_key or os.getenv("ODDS_API_KEY", "")
    if not key:
        return None

    try:
        resp = requests.get(
            f"https://api.the-odds-api.com/v4/sports/soccer_epl/events/{event_id}/odds",
            params={
                "apiKey": key,
                "regions": "uk",
                "markets": "h2h,btts,totals",
                "oddsFormat": "decimal"
            },
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        _logger.warning("Failed to fetch match details for %s: %s", event_id, str(e))
        return None


# --- Transfer news -----------------------------------------------------------
# transfer_feeds.py is deliberately Streamlit-free so GitHub Actions can import
# it. The caching lives here instead, keyed *per player* rather than per batch:
# scans are then incremental (raising the scan depth fetches only the new names),
# and a background warm can fill the cache before anyone presses a button.

#: One request per player is the cost model, so cache hard. Six hours still
#: catches a deadline-day medical, and the page carries a manual refresh.
TRANSFER_NEWS_TTL_SECONDS = 6 * 3600

_PREFETCH_STARTED = set()
_PREFETCH_LOCK = threading.Lock()


def _news_cache_key(player) -> str:
    digest = hashlib.sha1(str(player).strip().lower().encode("utf-8")).hexdigest()
    return "transfer_news:v2:%s" % digest


def _read_cached_news(pairs):
    """Split players into cached rows and the ones still needing a fetch.

    A player with genuinely no news caches an empty list, which is a *hit* — the
    distinction from a miss (``None``) is what stops the quiet ones being
    refetched on every single scan.
    """
    from scripts.common.cache import cache_get

    rows, missing = [], []
    for name, team in pairs:
        cached = cache_get(_news_cache_key(name))
        if cached is None:
            missing.append((name, team))
        else:
            rows.extend(cached)
    return rows, missing


def _write_cached_news(player, records) -> None:
    from scripts.common.cache import cache_set

    try:
        cache_set(_news_cache_key(player), records or [], ttl=TRANSFER_NEWS_TTL_SECONDS)
    except Exception:
        _logger.warning("Could not cache transfer news for %s", player, exc_info=True)


def transfer_news_cache_status(players):
    """``(cached, missing)`` counts, for telling the user what a scan will cost."""
    pairs = [tuple(p) if isinstance(p, (list, tuple)) else (p, None) for p in (players or [])]
    if not pairs:
        return 0, 0
    _rows, missing = _read_cached_news(pairs)
    return len(pairs) - len(missing), len(missing)


def get_transfer_news(players, force_refresh: bool = False, cached_only: bool = False,
                      progress=None):
    """Transfer headlines for many players, cached per player in SQLite.

    ``cached_only`` never touches the network — pages default to it so a render
    is always instant.

    Degrades to an empty frame on any failure: a news outage must leave every
    player undiscounted rather than take the page down.
    """
    import pandas as _pd

    from scripts.common.transfer_feeds import NEWS_COLUMNS, fetch_transfer_news_batch

    pairs = [tuple(p) if isinstance(p, (list, tuple)) else (p, None) for p in (players or [])]
    if not pairs:
        return _pd.DataFrame(columns=NEWS_COLUMNS)

    if force_refresh:
        cached_rows, missing = [], pairs
    else:
        cached_rows, missing = _read_cached_news(pairs)

    if cached_only or not missing:
        return _pd.DataFrame(cached_rows, columns=NEWS_COLUMNS) if cached_rows \
            else _pd.DataFrame(columns=NEWS_COLUMNS)

    try:
        fetched = fetch_transfer_news_batch(
            missing, progress=progress, on_result=_write_cached_news
        )
    except Exception as e:
        _logger.warning("Transfer news batch failed: %s", e)
        fetched = _pd.DataFrame(columns=NEWS_COLUMNS)

    frames = []
    if cached_rows:
        frames.append(_pd.DataFrame(cached_rows, columns=NEWS_COLUMNS))
    if not fetched.empty:
        frames.append(fetched)
    if not frames:
        return _pd.DataFrame(columns=NEWS_COLUMNS)
    return _pd.concat(frames, ignore_index=True)


#: Club news is 20 requests, not 150, so it can be refreshed far more eagerly
#: than the per-player scan.
CLUB_NEWS_TTL_SECONDS = 3 * 3600


def _club_news_cache_key(club) -> str:
    digest = hashlib.sha1(str(club).strip().lower().encode("utf-8")).hexdigest()
    return "club_transfer_news:v1:%s" % digest


def _read_cached_club_news(clubs):
    """Split clubs into cached rows and the ones still needing a fetch.

    Same empty-list-is-a-hit rule as the player cache: a quiet club must not be
    refetched on every render.
    """
    from scripts.common.cache import cache_get

    rows, missing = [], []
    for club in clubs:
        cached = cache_get(_club_news_cache_key(club))
        if cached is None:
            missing.append(club)
        else:
            rows.extend(cached)
    return rows, missing


def _write_cached_club_news(club, records) -> None:
    from scripts.common.cache import cache_set

    try:
        cache_set(_club_news_cache_key(club), records or [], ttl=CLUB_NEWS_TTL_SECONDS)
    except Exception:
        _logger.warning("Could not cache club transfer news for %s", club, exc_info=True)


def club_news_cache_status(clubs):
    """``(cached, missing)`` counts for the club-level signings scan."""
    names = [str(c) for c in (clubs or []) if str(c).strip()]
    if not names:
        return 0, 0
    _rows, missing = _read_cached_club_news(names)
    return len(names) - len(missing), len(missing)


def get_club_transfer_news(clubs, force_refresh: bool = False,
                           cached_only: bool = False, progress=None):
    """Signing headlines per club, cached in SQLite exactly like player news.

    Degrades to an empty frame on any failure: without it every incumbent simply
    keeps a neutral multiplier, which is the pre-existing behaviour.
    """
    import pandas as _pd

    from scripts.common.transfer_feeds import (
        CLUB_NEWS_COLUMNS, fetch_club_transfer_news_batch,
    )

    names = [str(c) for c in (clubs or []) if str(c).strip()]
    if not names:
        return _pd.DataFrame(columns=CLUB_NEWS_COLUMNS)

    if force_refresh:
        cached_rows, missing = [], names
    else:
        cached_rows, missing = _read_cached_club_news(names)

    if cached_only or not missing:
        return _pd.DataFrame(cached_rows, columns=CLUB_NEWS_COLUMNS) if cached_rows \
            else _pd.DataFrame(columns=CLUB_NEWS_COLUMNS)

    try:
        fetched = fetch_club_transfer_news_batch(
            missing, progress=progress, on_result=_write_cached_club_news
        )
    except Exception as e:
        _logger.warning("Club transfer news batch failed: %s", e)
        fetched = _pd.DataFrame(columns=CLUB_NEWS_COLUMNS)

    frames = []
    if cached_rows:
        frames.append(_pd.DataFrame(cached_rows, columns=CLUB_NEWS_COLUMNS))
    if not fetched.empty:
        frames.append(fetched)
    if not frames:
        return _pd.DataFrame(columns=CLUB_NEWS_COLUMNS)
    return _pd.concat(frames, ignore_index=True)


def start_club_news_prefetch(clubs, label: str = "default") -> bool:
    """Warm the club-signings cache on a background thread. See the player
    version for why this runs before anyone asks for it."""
    names = [str(c) for c in (clubs or []) if str(c).strip()]
    if not names:
        return False

    key = "clubs:%s" % label
    with _PREFETCH_LOCK:
        if key in _PREFETCH_STARTED:
            return False
        _PREFETCH_STARTED.add(key)

    def _worker():
        try:
            from scripts.common.transfer_feeds import fetch_club_transfer_news_batch

            _rows, missing = _read_cached_club_news(names)
            if not missing:
                return
            _logger.info("Prefetching signings news for %d club(s)", len(missing))
            fetch_club_transfer_news_batch(missing, on_result=_write_cached_club_news)
        except Exception:
            _logger.warning("Club news prefetch failed", exc_info=True)

    threading.Thread(target=_worker, name="club-news-prefetch", daemon=True).start()
    return True


def start_transfer_news_prefetch(players, label: str = "default") -> bool:
    """Warm the transfer-news cache on a background thread.

    Fetching is the slow part of the draft board, and it does not depend on
    anything the user does — so start it as soon as the app knows who is on the
    board, and let it finish while they read. By the time the Scan button is
    pressed the work is usually already done.

    The worker touches only Streamlit-free code (``transfer_feeds`` and the SQLite
    cache); calling ``st.*`` from a thread with no ScriptRunContext would fail.
    Returns True if it started a thread, False if one already ran this session.
    """
    pairs = [tuple(p) if isinstance(p, (list, tuple)) else (p, None) for p in (players or [])]
    if not pairs:
        return False

    with _PREFETCH_LOCK:
        if label in _PREFETCH_STARTED:
            return False
        _PREFETCH_STARTED.add(label)

    def _worker():
        try:
            from scripts.common.transfer_feeds import fetch_transfer_news_batch

            _rows, missing = _read_cached_news(pairs)
            if not missing:
                return
            _logger.info("Prefetching transfer news for %d player(s)", len(missing))
            fetch_transfer_news_batch(missing, on_result=_write_cached_news)
        except Exception:
            _logger.warning("Transfer news prefetch failed", exc_info=True)

    threading.Thread(target=_worker, name="transfer-news-prefetch", daemon=True).start()
    return True


# --- Transfer odds -----------------------------------------------------------
# Same split as the news feeds above: odds_feeds.py stays Streamlit-free for
# GitHub Actions, and the SQLite caching lives here. Ladders are cached per
# player so expanding a row twice costs one request, and the index is cached
# whole because it is a single page.

#: The index is one cheap request and prices do move within a day.
ODDS_INDEX_TTL_SECONDS = 3 * 3600

#: Ladders move slower than the index, and the live feed's own stamps are often
#: months old — refetching hourly would not make a five-month-old quote fresher.
ODDS_LADDER_TTL_SECONDS = 12 * 3600

_ODDS_INDEX_CACHE_KEY = "transfer_odds_index:v1"


def _odds_ladder_cache_key(slug) -> str:
    digest = hashlib.sha1(str(slug).strip().lower().encode("utf-8")).hexdigest()
    return "transfer_odds_ladder:v1:%s" % digest


def get_transfer_odds_index(force_refresh: bool = False, cached_only: bool = False):
    """Every player with a live next-club market, cached in SQLite.

    Degrades to an empty frame on any failure — a dead odds feed must leave the
    page standing, just with no odds column.
    """
    from scripts.common.cache import cache_get, cache_set
    from scripts.common.odds_feeds import fetch_odds_index
    from scripts.common.transfer_odds import ODDS_INDEX_COLUMNS

    if not force_refresh:
        cached = cache_get(_ODDS_INDEX_CACHE_KEY)
        if cached is not None:
            return pd.DataFrame(cached, columns=ODDS_INDEX_COLUMNS)
    if cached_only:
        return pd.DataFrame(columns=ODDS_INDEX_COLUMNS)

    try:
        df = fetch_odds_index()
    except Exception:
        _logger.warning("Transfer odds index fetch failed", exc_info=True)
        return pd.DataFrame(columns=ODDS_INDEX_COLUMNS)

    if df is None or df.empty:
        return pd.DataFrame(columns=ODDS_INDEX_COLUMNS)
    try:
        cache_set(_ODDS_INDEX_CACHE_KEY, df.to_dict("records"),
                  ttl=ODDS_INDEX_TTL_SECONDS)
    except Exception:
        _logger.warning("Could not cache transfer odds index", exc_info=True)
    return df


def get_player_odds_ladder(player, slug=None, force_refresh: bool = False,
                           cached_only: bool = False):
    """One player's destination ladder, cached per player.

    An empty ladder caches as an empty list, which is a *hit* — most players
    have no market at all, and treating that as a miss would refetch the quiet
    majority on every expand.
    """
    from scripts.common.cache import cache_get, cache_set
    from scripts.common.odds_feeds import fetch_player_odds_ladder, player_slug
    from scripts.common.transfer_odds import ODDS_LADDER_COLUMNS

    resolved = slug or player_slug(player)
    if not resolved:
        return pd.DataFrame(columns=ODDS_LADDER_COLUMNS)

    key = _odds_ladder_cache_key(resolved)
    if not force_refresh:
        cached = cache_get(key)
        if cached is not None:
            return pd.DataFrame(cached, columns=ODDS_LADDER_COLUMNS)
    if cached_only:
        return pd.DataFrame(columns=ODDS_LADDER_COLUMNS)

    try:
        df = fetch_player_odds_ladder(resolved, player=player)
    except Exception:
        _logger.warning("Odds ladder fetch failed for %s", resolved, exc_info=True)
        return pd.DataFrame(columns=ODDS_LADDER_COLUMNS)

    records = [] if df is None or df.empty else df.to_dict("records")
    try:
        cache_set(key, records, ttl=ODDS_LADDER_TTL_SECONDS)
    except Exception:
        _logger.warning("Could not cache odds ladder for %s", resolved, exc_info=True)
    return pd.DataFrame(records, columns=ODDS_LADDER_COLUMNS)


# =============================================================================
# FFP status rendering
# =============================================================================

def render_ffp_status(feed: "FFPFeed", expected_gw: Optional[int] = None,
                      contributed: bool = True) -> None:
    """One line telling the reader which gameweek FFP contributed, if any.

    Lives here, next to the feed, so every page says the same thing rather than
    inventing its own wording. Kept deliberately quiet when FFP is current: the
    caption is for the case where it is not, and a warning that fires every week
    gets ignored the week it matters.
    """
    if feed is None:
        return
    if not feed.ok:
        st.caption(
            "⚠️ Fantasy Football Pundit is unavailable, so projections are "
            "Rotowire-only. " + (feed.note or "")
        )
        return

    if expected_gw is None:
        expected_gw = config.CURRENT_GAMEWEEK

    if feed.is_stale(expected_gw):
        st.warning(
            f"Fantasy Football Pundit has published **GW{feed.gameweek}**, not "
            f"**GW{expected_gw}** — it is excluded from the numbers below, which "
            f"are Rotowire-only. Blending a different gameweek's projections is "
            f"invisible once it is in the totals, so it is left out rather than "
            f"quietly mixed in."
        )
        return

    if not contributed:
        return

    bits = ["Fantasy Football Pundit GW%s" % feed.gameweek] if feed.gameweek else ["Fantasy Football Pundit"]
    if feed.updated is not None:
        bits.append("published %s" % format_last_updated(feed.updated))
    if feed.provenance == "sheet":
        bits.append("read from their spreadsheet (site unreachable)")
    st.caption("Projections blended with " + " · ".join(bits) + ".")
