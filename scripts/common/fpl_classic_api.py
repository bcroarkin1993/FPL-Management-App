"""
FPL Classic Mode API Functions.

All functions that interact with the Classic FPL API endpoints
(fantasy.premierleague.com/api/).
"""

import pandas as pd
import requests
import streamlit as st
from typing import Dict, List, Any, Optional

from scripts.common.error_helpers import get_logger
from scripts.common.text_helpers import position_converter

_logger = get_logger("fpl_app.fpl_classic_api")


# =============================================================================
# BOOTSTRAP & STANDINGS
# =============================================================================

@st.cache_data(show_spinner=False, ttl=300)
def get_classic_bootstrap_static() -> Optional[Dict[str, Any]]:
    """
    Fetch Classic FPL bootstrap data (players, teams, events).

    Returns the full player pool with prices, stats, ownership percentages,
    team info, and gameweek events.

    Endpoint: https://fantasy.premierleague.com/api/bootstrap-static/

    Returns:
    - Dictionary containing 'elements' (players), 'teams', 'events', etc.
    - None if the request fails.
    """
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch classic bootstrap-static data", exc_info=True)
        return None


@st.cache_data(show_spinner=False, ttl=300)
def get_classic_league_standings(league_id: int, page: int = 1) -> Optional[Dict[str, Any]]:
    """
    Fetch Classic FPL league standings with pagination.

    Returns league metadata and standings for the specified page
    (50 entries per page by default).

    Endpoint: https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page}

    Parameters:
    - league_id: The Classic FPL league ID.
    - page: Page number for pagination (default: 1).

    Returns:
    - Dictionary with 'league' (metadata) and 'standings' (results).
    - None if the request fails or league not found.
    """
    if not league_id:
        return None
    try:
        url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
        params = {"page_standings": page}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch classic league standings for league %s", league_id, exc_info=True)
        return None


@st.cache_data(show_spinner=False, ttl=300)
def get_h2h_league_standings(league_id: int, page: int = 1) -> Optional[Dict[str, Any]]:
    """
    Fetch Head-to-Head (H2H) FPL league standings with pagination.

    Returns league metadata and standings for the specified page.

    Endpoint: https://fantasy.premierleague.com/api/leagues-h2h/{league_id}/standings/?page_standings={page}

    Parameters:
    - league_id: The H2H FPL league ID.
    - page: Page number for pagination (default: 1).

    Returns:
    - Dictionary with 'league' (metadata) and 'standings' (results).
    - None if the request fails or league not found.
    """
    if not league_id:
        return None
    try:
        url = f"https://fantasy.premierleague.com/api/leagues-h2h/{league_id}/standings/"
        params = {"page_standings": page}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch H2H league standings for league %s", league_id, exc_info=True)
        return None


# =============================================================================
# H2H MATCHES
# =============================================================================

@st.cache_data(ttl=300)
def get_h2h_league_matches(league_id: int, event: int = None, page: int = 1) -> Optional[Dict[str, Any]]:
    """
    Fetch Head-to-Head (H2H) league matches/fixtures.

    Returns matchups for the specified gameweek with team IDs, names, and scores.

    Endpoint: https://fantasy.premierleague.com/api/leagues-h2h-matches/league/{league_id}/

    Parameters:
    - league_id: The H2H FPL league ID.
    - event: The gameweek number (optional, defaults to current if not specified).
    - page: Page number for pagination (default: 1).

    Returns:
    - Dictionary with 'results' containing match data.
    - None if the request fails or league not found.
    """
    if not league_id:
        return None
    try:
        url = f"https://fantasy.premierleague.com/api/leagues-h2h-matches/league/{league_id}/"
        params = {"page": page}
        if event:
            params["event"] = event
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch H2H league matches for league %s", league_id, exc_info=True)
        return None


@st.cache_data(ttl=600)
def get_all_h2h_league_matches(league_id: int) -> List[Dict[str, Any]]:
    """
    Fetch all matches from an H2H league (all gameweeks, all pages).

    Parameters:
    - league_id: The H2H FPL league ID.

    Returns:
    - List of all match dictionaries.
    """
    all_matches = []
    page = 1

    while True:
        data = get_h2h_league_matches(league_id, page=page)
        if not data or not data.get("results"):
            break

        all_matches.extend(data["results"])

        if not data.get("has_next", False):
            break

        page += 1
        if page > 50:  # Safety limit
            break

    return all_matches


def get_classic_h2h_record(league_id: int, team1_id: int, team2_id: int) -> Dict[str, Any]:
    """
    Calculate head-to-head record between two teams in a Classic H2H league.

    Parameters:
    - league_id: The H2H FPL league ID.
    - team1_id: The entry ID of team 1.
    - team2_id: The entry ID of team 2.

    Returns:
    - Dictionary with 'wins', 'draws', 'losses', 'points_for', 'points_against',
      'record_str' (e.g., "3-1-2"), and 'matches' (list of individual matchups).
    """
    result = {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points_for": 0,
        "points_against": 0,
        "record_str": "0-0-0",
        "matches": []
    }

    all_matches = get_all_h2h_league_matches(league_id)
    if not all_matches:
        return result

    for match in all_matches:
        entry_1 = match.get("entry_1_entry")
        entry_2 = match.get("entry_2_entry")
        pts_1 = match.get("entry_1_points", 0)
        pts_2 = match.get("entry_2_points", 0)
        gw = match.get("event")

        # Skip unplayed matches
        if pts_1 is None or pts_2 is None:
            continue
        if pts_1 == 0 and pts_2 == 0 and not match.get("finished"):
            continue

        # Check if this match involves both teams
        if (entry_1 == team1_id and entry_2 == team2_id):
            my_pts, opp_pts = pts_1, pts_2
        elif (entry_1 == team2_id and entry_2 == team1_id):
            my_pts, opp_pts = pts_2, pts_1
        else:
            continue

        result["points_for"] += my_pts
        result["points_against"] += opp_pts

        if my_pts > opp_pts:
            result["wins"] += 1
            outcome = "W"
        elif my_pts < opp_pts:
            result["losses"] += 1
            outcome = "L"
        else:
            result["draws"] += 1
            outcome = "D"

        result["matches"].append({
            "gameweek": gw,
            "my_pts": my_pts,
            "opp_pts": opp_pts,
            "outcome": outcome
        })

    result["record_str"] = f"{result['wins']}-{result['draws']}-{result['losses']}"
    return result


def get_league_standings(league_id: int, page: int = 1) -> Optional[Dict[str, Any]]:
    """
    Fetch league standings, automatically detecting if it's Classic or H2H.

    Tries the classic endpoint first, then falls back to H2H if that fails.

    Parameters:
    - league_id: The FPL league ID.
    - page: Page number for pagination (default: 1).

    Returns:
    - Dictionary with 'league' (metadata) and 'standings' (results).
    - None if both endpoints fail.
    """
    # Try classic endpoint first
    data = get_classic_league_standings(league_id, page)
    if data:
        return data

    # Fall back to H2H endpoint
    data = get_h2h_league_standings(league_id, page)
    return data


# =============================================================================
# TEAM HISTORY & PICKS
# =============================================================================

@st.cache_data(show_spinner=False, ttl=300)
def get_classic_team_history(team_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a Classic FPL team's full season history.

    Returns gameweek-by-gameweek results (points, rank, bank, team value),
    chips used, and past season summaries.

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/history/

    Parameters:
    - team_id: The FPL Classic team ID.

    Returns:
    - Dictionary with 'current' (GW history), 'chips', 'past' (past seasons).
    - None if the request fails or team not found.
    """
    if not team_id:
        return None
    try:
        url = f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch classic team history for team %s", team_id, exc_info=True)
        return None


@st.cache_data(show_spinner=False, ttl=60)
def get_classic_team_picks(team_id: int, gw: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a Classic FPL team's picks for a specific gameweek.

    Returns squad picks, captain selection, active chip (if any),
    and entry history for that gameweek.

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/

    Parameters:
    - team_id: The FPL Classic team ID.
    - gw: The gameweek number.

    Returns:
    - Dictionary with 'picks', 'active_chip', 'entry_history', etc.
    - None if the request fails or team/gw not found.
    """
    if not team_id or not gw:
        return None
    try:
        url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch classic team picks for team %s GW %s", team_id, gw, exc_info=True)
        return None


@st.cache_data(show_spinner=False, ttl=300)
def get_classic_transfers(team_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch all transfers made by a Classic FPL team.

    Returns a list of all transfers with element_in, element_out,
    event (gameweek), and timestamp.

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/transfers/

    Parameters:
    - team_id: The FPL Classic team ID.

    Returns:
    - List of transfer dictionaries.
    - None if the request fails or team not found.
    """
    if not team_id:
        return None
    try:
        url = f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch classic transfers for team %s", team_id, exc_info=True)
        return None


@st.cache_data(ttl=600)
def get_entry_details(team_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetches entry details for a Classic FPL team.

    Parameters:
    - team_id: The FPL Classic team ID.

    Returns:
    - Dictionary with team details or None if not found.
    """
    if not team_id:
        return None
    try:
        return requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/", timeout=10).json()
    except Exception:
        _logger.warning("Failed to fetch entry details for team %s", team_id, exc_info=True)
        return None


# =============================================================================
# CLASSIC LIVE POINTS & POSITION DATA
# =============================================================================

@st.cache_data(ttl=3600)
def _get_classic_gw_live_points(gw: int) -> dict:
    """Returns {element_id: gw_points} from the Classic live endpoint for a single GW."""
    try:
        url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        return {
            elem["id"]: elem.get("stats", {}).get("total_points", 0)
            for elem in data.get("elements", [])
        }
    except Exception:
        _logger.warning("Failed to fetch classic GW %s live points", gw, exc_info=True)
        return {}


@st.cache_data(ttl=3600)
def _get_classic_team_picks_for_gw(team_id: int, gw: int) -> list:
    """Returns list of element IDs for a Classic team's picks in a single GW."""
    picks_data = get_classic_team_picks(team_id, gw)
    if not picks_data:
        return []
    return [p["element"] for p in picks_data.get("picks", [])]


@st.cache_data(ttl=600)
def get_classic_team_position_data(team_id: int, max_gw: int) -> dict:
    """
    GW-by-GW position point accumulation for a single Classic FPL team.

    Returns:
    - {"positions": {"GK": pts, "DEF": pts, "MID": pts, "FWD": pts},
       "players": [{"player": str, "position": str, "total_points": int, "team": str}, ...]}
    """
    POS_DISPLAY = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        return {"positions": {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}, "players": []}

    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams_map = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    positions = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    player_accum = {}  # {element_id: {player, position, total_points, team}}

    for gw in range(1, max_gw + 1):
        live_points = _get_classic_gw_live_points(gw)
        pick_ids = _get_classic_team_picks_for_gw(team_id, gw)

        for eid in pick_ids:
            elem = elements.get(eid, {})
            pos_short = position_converter(elem.get("element_type", 0))
            pos_display = POS_DISPLAY.get(pos_short, "Unknown")
            gw_pts = live_points.get(eid, 0)

            if pos_display in positions:
                positions[pos_display] += gw_pts

            if eid not in player_accum:
                player_accum[eid] = {
                    "player": elem.get("web_name", "Unknown"),
                    "position": pos_display,
                    "total_points": 0,
                    "team": teams_map.get(elem.get("team"), "???"),
                }
            player_accum[eid]["total_points"] += gw_pts

    return {"positions": positions, "players": list(player_accum.values())}
