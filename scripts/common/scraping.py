"""
Web Scraping & External Data Source Functions.

Rotowire scraping, Fantasy Football Pundit data, and The Odds API integration.
"""

import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import streamlit as st
from typing import Optional
from urllib.parse import urljoin

import config
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.scraping")


# =============================================================================
# ROTOWIRE SCRAPING
# =============================================================================

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
    EXPECTED_COLUMNS = 12  # Number of columns expected per row

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

    # Parse each row
    data = []
    skipped_rows = 0
    for tr in rows:
        tds = tr.find_all("td")

        # Skip rows that don't have expected column count (likely headers or malformed)
        if len(tds) < EXPECTED_COLUMNS:
            skipped_rows += 1
            continue

        # Extract cell text (handle extra columns gracefully by only taking first 12)
        cells = [td.get_text(strip=True) for td in tds[:EXPECTED_COLUMNS]]

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

    # Find any anchors whose href contains our base slug
    anchors = soup.select('a[href*="fantasy-premier-league-player-rankings-gameweek-"]')

    # Regex patterns to try (most specific to least specific)
    # Pattern 1: Standard format with optional slug words
    patterns = [
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"),
        # Pattern 2: Alternate format without trailing article ID
        re.compile(r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?$"),
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
        _logger.warning(
            "Rotowire URL discovery failed - no matching articles found. "
            "HTML structure may have changed. Found %d anchors on page.",
            len(anchors)
        )
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
        table = soup.select_one("table.article-table__tablesorter") or soup.find("table")
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

# FFP Google Sheets URL (public CSV export)
FFP_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaiTmUKjtQ7MxiGibN2GAZ8m9NHF3IA2U-yE0PhBpCOXHewhs57PrjZO7GQzZvrEGGBW7HFEE43yX0/pub?output=csv"


@st.cache_data(ttl=300)
def get_ffp_projections_data() -> Optional[pd.DataFrame]:
    """
    Fetch Fantasy Football Pundit projections data from their public Google Sheet.

    Returns DataFrame with columns:
    - Name, Team, Position, Fixture, Ownership, Start %, Price
    - CS (Clean Sheet odds), AnytimeGoal, AnytimeAssist, AnytimeReturn
    - Predicted, StartingPredicted (points predictions)
    - GW2-GW6, Next2GWs-Next6GWs (multi-GW forecasts)
    """
    try:
        from io import StringIO
        resp = requests.get(FFP_SHEET_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        if df.empty:
            _logger.warning("FFP data fetch returned empty DataFrame")
            return None

        # Clean up percentage columns (remove % sign, convert to float)
        pct_cols = ['Ownership', 'Start', 'LongStart', 'CS', 'AnytimeAssist', 'AnytimeGoal', 'AnytimeReturn']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean up price column (remove £ and m)
        if 'Price' in df.columns:
            df['Price'] = df['Price'].astype(str).str.replace('£', '').str.replace('m', '').str.strip()
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

        # Ensure numeric columns are properly typed
        numeric_cols = ['Predicted', 'StartingPredicted', 'GW2', 'GW3', 'GW4', 'GW5', 'GW6',
                        'Next2GWs', 'Next3GWs', 'Next4GWs', 'Next5GWs', 'Next6GWs',
                        'Next2GWsStart', 'Next3GWsStart', 'Next4GWsStart', 'Next5GWsStart', 'Next6GWsStart']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        _logger.debug("FFP data fetched successfully: %d players", len(df))
        return df

    except Exception as e:
        _logger.warning("Failed to fetch FFP projections data: %s", str(e))
        return None


def get_ffp_points_predictor() -> Optional[pd.DataFrame]:
    """
    Get FFP points predictor data formatted for display.

    Returns DataFrame with key columns for points predictions.
    """
    df = get_ffp_projections_data()
    if df is None:
        return None

    # Select and rename columns for display
    cols = {
        'Name': 'Player',
        'Team': 'Team',
        'Position': 'Position',
        'Fixture': 'Fixture',
        'Price': 'Price',
        'Ownership': 'Ownership %',
        'Start': 'Start %',
        'Predicted': 'Predicted Pts',
        'StartingPredicted': 'Pts (if starts)',
        'Next2GWs': 'Next 2 GWs',
        'Next3GWs': 'Next 3 GWs',
        'Next6GWs': 'Next 6 GWs',
    }

    available = {k: v for k, v in cols.items() if k in df.columns}
    result = df[list(available.keys())].rename(columns=available)

    # Filter to players with >0% start chance for meaningful data
    if 'Start %' in result.columns:
        result = result[result['Start %'] > 0].copy()

    # Sort by predicted points
    if 'Predicted Pts' in result.columns:
        result = result.sort_values('Predicted Pts', ascending=False)

    return result.reset_index(drop=True)


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
