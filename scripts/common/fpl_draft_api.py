"""
FPL Draft League API Functions.

All functions that interact with the FPL Draft API endpoints
(draft.premierleague.com/api/).
"""

import pandas as pd
import requests
import streamlit as st
from typing import Dict, List, Any, Optional

import config
from scripts.common.error_helpers import get_logger
from scripts.common.text_helpers import (
    normalize_apostrophes,
    position_converter,
)

_logger = get_logger("fpl_app.fpl_draft_api")


# =============================================================================
# GAMEWEEK & LEAGUE BASICS
# =============================================================================

def get_current_gameweek():
    """
    Fetches the current gameweek based on the game status from the FPL Draft API.

    Returns:
    - current_gameweek: An integer representing the current gameweek.
    """
    game_url = "https://draft.premierleague.com/api/game"
    try:
        response = requests.get(game_url, timeout=20)
        game_data = response.json()

        # Check if the current event is finished
        if game_data['current_event_finished']:
            current_gameweek = game_data['next_event']
        else:
            current_gameweek = game_data['current_event']

        return current_gameweek
    except Exception as e:
        _logger.warning("Failed to fetch current gameweek: %s", e)
        return config.CURRENT_GAMEWEEK


def get_draft_picks(league_id):
    """
    Fetches the draft picks for each team in the league and returns a dictionary with team_id as the key,
    including player names instead of player IDs.

    Note: This is an alias for get_starting_team_composition for backward compatibility.

    Parameters:
    - league_id: The ID of the FPL Draft league.

    Returns:
    - draft_picks: A dictionary where keys are team IDs, and values are dictionaries with team name and player names.
    """
    return get_starting_team_composition(league_id)


@st.cache_data(ttl=3600)
def get_fpl_player_mapping():
    """
    Fetches FPL player data from the FPL Draft API and returns it as a dictionary to link player ids to player names.

    Returns:
    - fpl_player_data: Dictionary with Player_ID as key and dict with 'Player', 'Web_Name', 'Team', 'Position'.
    """
    # Fetch data from the FPL Draft API
    player_url = "https://draft.premierleague.com/api/bootstrap-static"
    try:
        response = requests.get(player_url, timeout=30)
        player_data = response.json()
    except Exception as e:
        _logger.warning("Failed to fetch FPL player mapping: %s", e)
        return {}

    # Extract relevant player information
    players = player_data['elements']
    teams = player_data.get('teams', [])

    # Create a mapping of player IDs to player information
    fpl_player_map = {}

    for player in players:
        player_id = player.get('id')
        first_name = player.get('first_name', '')
        second_name = player.get('second_name', '')
        full_name = f"{first_name} {second_name}"

        web_name = player.get('web_name', '').strip()
        if not web_name or web_name == full_name:
            web_name = None  # treat as missing if it's the same as full name or blank

        team_index = player.get('team', 0) - 1  # team index is 1-based
        position_index = player.get('element_type', 1) - 1

        team_short_name = teams[team_index]['short_name'] if 0 <= team_index < len(teams) else 'Unknown'
        position = ['G', 'D', 'M', 'F'][position_index] if 0 <= position_index < 4 else 'Unknown'

        fpl_player_map[player_id] = {
            'Player': full_name,
            'Web_Name': web_name,
            'Team': team_short_name,
            'Position': position
        }

    return fpl_player_map


def get_gameweek_fixtures(league_id, gameweek):
    """
    Fetches gameweek fixtures for a draft league.

    Parameters:
    - league_id: The ID of the FPL Draft league.
    - gameweek: The gameweek number.

    Returns:
    - gameweek_fixtures: List of fixture strings.
    """
    # Draft League API URLs
    league_url = f"https://draft.premierleague.com/api/league/{league_id}/details"

    # Get league details, fixtures, and team details
    try:
        league_response = requests.get(league_url, timeout=30).json()
    except Exception as e:
        _logger.warning("Failed to fetch gameweek fixtures for league %s: %s", league_id, e)
        return []

    # Extract the standings and league_entries sections of the JSON
    fixtures = league_response['matches']
    league_entries = league_response['league_entries']

    # Create a dictionary mapping entry_ids to team names and managers
    team_info = {entry['id']: (entry['entry_name'], entry['player_first_name'] + ' ' + entry['player_last_name'])
                 for entry in league_entries}

    # Create an empty list to add gameweek fixtures to
    gameweek_fixtures = []

    # Iterate over the fixtures, filter for the current_gameweek, and then format and add the fixture to list
    for fixture in fixtures:
        if fixture['event'] == gameweek:
            team1_id = fixture['league_entry_1']
            team2_id = fixture['league_entry_2']
            team1_name, team1_manager = team_info.get(team1_id, ("Unknown Team", "Unknown Manager"))
            team2_name, team2_manager = team_info.get(team2_id, ("Unknown Team", "Unknown Manager"))
            gameweek_fixtures.append(f"{team1_name} ({team1_manager}) vs {team2_name} ({team2_manager})")

    return gameweek_fixtures


def get_historical_team_scores(league_id: int) -> pd.DataFrame:
    """
    Pull per-team weekly scores from the Draft league details endpoint.

    Returns a DataFrame with at least:
      ['event', 'entry_id', 'entry_name', 'points', 'total_points']
    (future GWs without scores are excluded)

    Notes:
    - Uses /api/league/{league_id}/details
    - Reads 'matches' -> league_entry_1/2(_points)
    """
    try:
        league_id = int(league_id)
    except (TypeError, ValueError):
        raise ValueError("league_id must be an integer")

    url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    data = requests.get(url, timeout=30).json()

    entries = {}
    for e in data.get("league_entries", []) or []:
        # 'id' is the league-entry id in this payload
        eid = e.get("entry_id", e.get("id"))
        name = e.get("entry_name")
        if eid is not None and name:
            entries[int(eid)] = str(name)

    rows = []
    for m in data.get("matches", []) or []:
        gw = m.get("event")
        e1 = m.get("league_entry_1")
        e2 = m.get("league_entry_2")
        p1 = m.get("league_entry_1_points")
        p2 = m.get("league_entry_2_points")

        # Only record rows with realized points
        if isinstance(gw, int) and p1 is not None and e1 is not None:
            rows.append({
                "event": int(gw),
                "entry_id": int(e1),
                "entry_name": entries.get(int(e1), f"Team {e1}"),
                "points": float(p1),
            })
        if isinstance(gw, int) and p2 is not None and e2 is not None:
            rows.append({
                "event": int(gw),
                "entry_id": int(e2),
                "entry_name": entries.get(int(e2), f"Team {e2}"),
                "points": float(p2),
            })

    if not rows:
        return pd.DataFrame(columns=["event", "entry_id", "entry_name", "points", "total_points"])

    df = pd.DataFrame(rows)
    df["total_points"] = df["points"]  # alias used by the win-prob std estimator
    df = df.sort_values(["event", "entry_id"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=600)
def get_league_entries(league_id):
    """
    Fetches the league entries and creates a mapping of entry IDs to team names.

    Parameters:
    - league_id: The ID of the league.

    Returns:
    - A dictionary where keys are entry IDs, and values are team names.
    """
    url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    try:
        response = requests.get(url, timeout=30).json()
    except Exception as e:
        _logger.warning("Failed to fetch league entries for league %s: %s", league_id, e)
        return {}

    return {entry['entry_id']: entry['entry_name'] for entry in response['league_entries']}


@st.cache_data(ttl=300)
def get_league_player_ownership(league_id):
    """
    Fetch the current gameweek ownership for a Draft league and group by team (ID) and position.

    Returns:
      {
        <team_id>: {
          "team_name": <str>,
          "players": {"G": [..], "D":[..], "M":[..], "F":[..]}
        },
        ...
      }
    """
    # Endpoints
    element_status_url = f"https://draft.premierleague.com/api/league/{league_id}/element-status"
    league_details_url = f"https://draft.premierleague.com/api/league/{league_id}/details"

    # --- Fetch data
    element_status = requests.get(element_status_url, timeout=30).json().get("element_status", [])
    league_details = requests.get(league_details_url, timeout=30).json()

    # --- Build owner (entry) map robustly
    # Try the helper first (preferred)
    owner_map = {}
    try:
        owner_map = get_league_entries(league_id)  # expected {entry_id: entry_name}
    except Exception:
        _logger.warning("Failed to fetch league ownership for league %s", league_id, exc_info=True)
        owner_map = {}

    # Fallback/augment from raw payload; handle both 'entry_id' and 'id'
    for entry in league_details.get("league_entries", []):
        eid = entry.get("entry_id", entry.get("id"))
        name = entry.get("entry_name")
        if eid is not None and name:
            owner_map.setdefault(eid, name)

    # --- Player mapping (id -> {Player, Team, Position})
    player_map = get_fpl_player_mapping()

    # --- Build ownership dict keyed by team_id
    league_ownership = {}  # not defaultdict to keep final structure explicit

    for status in element_status:
        owner_id = status.get("owner")
        player_id = status.get("element")

        # Skip unowned
        if owner_id is None:
            continue

        # Ensure container
        if owner_id not in league_ownership:
            team_name = owner_map.get(owner_id, f"Unknown Team ({owner_id})")
            league_ownership[owner_id] = {
                "team_name": team_name,
                "players": {"G": [], "D": [], "M": [], "F": []},
            }

        # Resolve player info
        pinfo = player_map.get(player_id, {"Player": f"Unknown ({player_id})", "Position": None})
        pname = pinfo.get("Player")
        pos = pinfo.get("Position")

        # Only add to valid buckets
        if pos in {"G", "D", "M", "F"}:
            league_ownership[owner_id]["players"][pos].append(pname)

    # Sort player names within each position (optional, nicer display)
    for blob in league_ownership.values():
        for pos in ("G", "D", "M", "F"):
            blob["players"][pos].sort()

    return league_ownership


# =============================================================================
# LIVE POINTS & PICKS
# =============================================================================

@st.cache_data(ttl=3600)
def _get_draft_gw_live_points(gw: int) -> dict:
    """Returns {element_id (int): gw_points} from the Draft live endpoint for a single GW.

    The Draft live endpoint returns elements as a dict keyed by string IDs:
    {"1": {"stats": {"total_points": 10, ...}}, "2": {...}, ...}
    We normalise keys to int so they match the int element IDs from picks.
    """
    try:
        url = f"https://draft.premierleague.com/api/event/{gw}/live"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        elements = data.get("elements", {})
        return {
            int(eid): edata.get("stats", {}).get("total_points", 0)
            for eid, edata in elements.items()
        }
    except Exception:
        _logger.warning("Failed to fetch draft GW %s live points", gw, exc_info=True)
        return {}


@st.cache_data(ttl=60)  # Short TTL for live data
def get_live_gameweek_stats(gw: int) -> dict:
    """
    Returns live stats for all players in a gameweek.

    Returns:
        dict: {element_id: {'points': int, 'minutes': int, 'has_played': bool}}
    """
    try:
        url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        result = {}
        for elem in data.get("elements", []):
            stats = elem.get("stats", {})
            minutes = stats.get("minutes", 0)
            result[elem["id"]] = {
                'points': stats.get("total_points", 0),
                'minutes': minutes,
                'has_played': minutes > 0,
                'goals': stats.get("goals_scored", 0),
                'assists': stats.get("assists", 0),
                'bonus': stats.get("bonus", 0),
            }
        return result
    except Exception:
        _logger.warning("Failed to fetch live GW %s stats", gw, exc_info=True)
        return {}


@st.cache_data(ttl=300)
def is_gameweek_live(gw: int) -> bool:
    """
    Check if any fixtures in the gameweek have started (players have minutes).

    Returns:
        bool: True if at least one fixture has kicked off
    """
    live_stats = get_live_gameweek_stats(gw)
    if not live_stats:
        return False
    # Check if any player has played minutes
    return any(stats.get('has_played', False) for stats in live_stats.values())


@st.cache_data(ttl=3600)
def _get_draft_entry_picks_for_gw(entry_id: int, gw: int) -> list:
    """Returns list of element IDs for a Draft entry's picks in a single GW."""
    try:
        url = f"https://draft.premierleague.com/api/entry/{entry_id}/event/{gw}"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        return [p["element"] for p in data.get("picks", [])]
    except Exception:
        _logger.warning("Failed to fetch draft entry %s picks for GW %s", entry_id, gw, exc_info=True)
        return []


# =============================================================================
# POSITION DATA & POINTS
# =============================================================================

@st.cache_data(ttl=600)
def _fetch_draft_position_data(league_id: int) -> dict:
    """
    GW-by-GW position point accumulation for all teams in a Draft league.

    Iterates over every completed gameweek, fetching each team's actual roster
    and the points each player scored that week, so traded players are only
    counted for the weeks they were on a given team.

    Returns dict with:
    - 'team_data': {team_name: {'GK': pts, 'DEF': pts, 'MID': pts, 'FWD': pts}}
    - 'player_data': {team_name: [{'player': str, 'position': str, 'total_points': int, 'team': str}, ...]}
    """
    POS_DISPLAY = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

    # Fetch bootstrap-static for element_type / name mapping
    bootstrap_url = "https://draft.premierleague.com/api/bootstrap-static"
    bootstrap_data = requests.get(bootstrap_url, timeout=30).json()
    elements = {p["id"]: p for p in bootstrap_data.get("elements", [])}
    teams_list = bootstrap_data.get("teams", [])
    teams_map = {t["id"]: t["short_name"] for t in teams_list}

    # League entries: {entry_id: team_name}
    owner_map = get_league_entries(league_id)

    # Determine range of completed GWs
    current_gw = get_current_gameweek()  # next GW if current finished, else current

    # Initialise accumulators
    team_data = {}
    player_accum = {}  # {team_name: {element_id: {...}}}
    for team_name in owner_map.values():
        team_data[team_name] = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        player_accum[team_name] = {}

    # Iterate over completed gameweeks
    for gw in range(1, current_gw):
        live_points = _get_draft_gw_live_points(gw)

        for entry_id, team_name in owner_map.items():
            pick_ids = _get_draft_entry_picks_for_gw(entry_id, gw)

            for eid in pick_ids:
                elem = elements.get(eid, {})
                pos_short = position_converter(elem.get("element_type", 0))
                pos_display = POS_DISPLAY.get(pos_short, "Unknown")
                gw_pts = live_points.get(eid, 0)

                if pos_display in team_data[team_name]:
                    team_data[team_name][pos_display] += gw_pts

                if eid not in player_accum[team_name]:
                    player_accum[team_name][eid] = {
                        "player": elem.get("web_name", "Unknown"),
                        "position": pos_display,
                        "total_points": 0,
                        "team": teams_map.get(elem.get("team"), "???"),
                    }
                player_accum[team_name][eid]["total_points"] += gw_pts

    # Convert player accumulators to list format
    player_data = {
        team_name: list(players.values())
        for team_name, players in player_accum.items()
    }

    return {"team_data": team_data, "player_data": player_data}


def get_draft_points_by_position(league_id: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns [Team, GK, DEF, MID, FWD, Total]
    showing season-to-date points by position for each team in the draft league.
    """
    data = _fetch_draft_position_data(league_id)
    team_data = data["team_data"]

    rows = []
    for team_name, positions in team_data.items():
        total = sum(positions.values())
        rows.append({
            "Team": team_name,
            "GK": positions["GK"],
            "DEF": positions["DEF"],
            "MID": positions["MID"],
            "FWD": positions["FWD"],
            "Total": total,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Total", ascending=False).reset_index(drop=True)
    return df


def get_draft_team_players_with_points(league_id: int) -> dict:
    """
    Returns {team_name: [{'player': str, 'position': str, 'total_points': int, 'team': str}, ...]}
    with per-player detail for each team in the draft league.
    """
    data = _fetch_draft_position_data(league_id)
    return data["player_data"]


def get_league_teams(league_id):
    """
    Fetches league entries from the FPL Draft API and stores them in config.LEAGUE_DATA if not already fetched.

    Note: This wraps get_league_entries with caching via config.LEAGUE_DATA.

    Returns:
    - team_dict: A dictionary mapping entry IDs to team names.
    """
    if config.LEAGUE_DATA is None:  # Fetch only if not already fetched
        config.LEAGUE_DATA = get_league_entries(league_id)

    return config.LEAGUE_DATA


# =============================================================================
# TEAM COMPOSITION & LINEUP
# =============================================================================

@st.cache_data(ttl=3600)
def get_starting_team_composition(league_id):
    """
    Fetches the draft picks for each team in the league and returns a dictionary with team_id as the primary key
    and the team_name field plus a player field with all the player names.

    Parameters:
    - league_id: The ID of the FPL Draft league.

    Returns:
    - draft_picks: A dictionary where keys are team IDs, and values are dictionaries with team name and player names.
    """
    # Endpoints for draft picks and player data
    draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
    league_details_url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    player_data_url = "https://draft.premierleague.com/api/bootstrap-static"

    # Fetch data
    try:
        draft_data = requests.get(draft_url, timeout=30).json()
        league_details = requests.get(league_details_url, timeout=30).json()
        player_data = requests.get(player_data_url, timeout=30).json()
    except Exception as e:
        _logger.warning("Failed to fetch team composition for league %s: %s", league_id, e)
        return {}

    # Create a mapping of entry ID to team name
    team_names = {entry['id']: entry['entry_name'] for entry in league_details['league_entries']}

    # Create a mapping of player ID to player name
    player_mapping = {
        player['id']: f"{player['first_name']} {player['second_name']}"
        for player in player_data['elements']
    }

    # Initialize the draft picks dictionary
    draft_picks = {}

    # Populate draft picks
    for choice in draft_data['choices']:
        team_id = choice['entry']         # Team ID
        team_name = choice['entry_name']  # Team name
        player_id = choice['element']     # Player ID

        # Ensure the team_id key exists in the dictionary
        if team_id not in draft_picks:
            draft_picks[team_id] = {
                'team_name': team_name,
                'players': []  # Initialize player list
            }

        # Add the player name to the list
        player_name = player_mapping.get(player_id, f"Unknown ({player_id})")
        draft_picks[team_id]['players'].append(player_name)

    return draft_picks


@st.cache_data(ttl=60)
def get_team_actual_lineup(team_id: int, gameweek: int) -> pd.DataFrame:
    """
    Get the actual starting 11 picks for a team in a specific gameweek.

    Uses the FPL Draft picks API to get the real lineup the manager selected,
    not an "optimal" calculated lineup.

    Parameters:
    - team_id: The team/entry ID
    - gameweek: The gameweek number

    Returns:
    - DataFrame with columns ['Player', 'Team', 'Position', 'Is_Starter'] for all 15 players
      Is_Starter is True for positions 1-11, False for bench (12-15)
    """
    player_map = get_fpl_player_mapping()

    url = f"https://draft.premierleague.com/api/entry/{team_id}/event/{gameweek}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        _logger.warning("Failed to fetch team picks for team %s GW %s: %s", team_id, gameweek, e)
        return pd.DataFrame(columns=['Player', 'Team', 'Position', 'Is_Starter'])

    picks = data.get('picks', [])
    if not picks:
        return pd.DataFrame(columns=['Player', 'Team', 'Position', 'Is_Starter'])

    rows = []
    for pick in picks:
        element_id = pick.get('element')
        position_slot = pick.get('position', 0)  # 1-11 = starters, 12-15 = bench
        is_starter = position_slot <= 11

        player_info = player_map.get(element_id, {})
        rows.append({
            'Player': player_info.get('Player', f'Unknown ({element_id})'),
            'Team': player_info.get('Team', '???'),
            'Position': player_info.get('Position', '?'),
            'Is_Starter': is_starter,
            'Player_ID': element_id,
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def get_team_composition_for_gameweek(league_id, team_id, gameweek):
    """
    Determines the composition of a given FPL team for a specified gameweek.

    Parameters:
    - league_id (int): The ID of the league.
    - team_id (int): The team ID of the team to fetch.
    - gameweek (int | None): The gameweek for which to determine the team's composition.

    Returns:
    - DataFrame with columns ['Player', 'Team', 'Position'] for the specified gameweek.
    """
    # Coerce IDs and settle the gameweek
    try:
        league_id = int(league_id)
    except (TypeError, ValueError):
        raise ValueError("league_id must be an integer")

    try:
        team_id = int(team_id)
    except (TypeError, ValueError):
        raise ValueError("team_id must be an integer")

    if gameweek is None:
        # Fallback to your helper if you allow None
        try:
            gameweek = int(get_current_gameweek())
        except Exception:
            # If even that fails, treat as "no cap" on transactions
            gameweek = None
    else:
        try:
            gameweek = int(gameweek)
        except (TypeError, ValueError):
            gameweek = None

    # Player map: {player_id: {'Player', 'Team', 'Position'}}
    player_map = get_fpl_player_mapping()

    # Reverse map for fast name->id lookup (built from the same source as starting comp)
    name_to_id = {v['Player']: k for k, v in player_map.items()}

    # Starting composition (names)
    starting = get_starting_team_composition(league_id)  # {team_id: {'team_name':..., 'players': [names...]}, ...}
    team_start = starting.get(int(team_id), {})
    team_composition = set(team_start.get('players', []))  # set of player NAMES

    # Apply approved transactions up to gameweek (convert IDs -> names using player_map)
    transactions = get_waiver_transactions_up_to_gameweek(league_id, gameweek)
    for tx in transactions:
        if tx.get('entry') == int(team_id) and tx.get('result') == 'a':
            pid_in = tx.get('element_in')
            pid_out = tx.get('element_out')

            # Map IDs to FPL names; if missing, keep a label that won't break
            name_in = player_map.get(pid_in, {}).get('Player', f"Unknown ({pid_in})")
            name_out = player_map.get(pid_out, {}).get('Player', f"Unknown ({pid_out})")

            # Update the name-based composition defensively
            team_composition.discard(name_out)   # discard avoids KeyError if not present
            team_composition.add(name_in)

    # Build the output rows using the map; if a name can't be mapped back, fill Unknowns
    rows = []
    for name in sorted(team_composition):
        pid = name_to_id.get(name)
        if pid is not None and pid in player_map:
            rows.append({
                'Player': player_map[pid]['Player'],
                'Team':   player_map[pid]['Team'],
                'Position': player_map[pid]['Position'],
            })
        else:
            rows.append({
                'Player': name,
                'Team':   'Unknown',
                'Position': 'Unknown',
            })

    return pd.DataFrame(rows, columns=['Player', 'Team', 'Position'])


def get_team_id_by_name(league_id, team_name):
    """
    Converts a team name to its corresponding team ID in the specified FPL league.

    Parameters:
    - league_id (int): The ID of the FPL league.
    - team_name (str): The name of the team to search for.

    Returns:
    - team_id (int): The ID of the team with the given name.

    Raises:
    - ValueError: If the team name is not found in the league.
    """
    # Fetch league entries to map team names to IDs
    team_map = dict(get_league_entries(league_id))  # Ensure team_map is a dictionary

    # Normalize the input team name
    normalized_team_name = normalize_apostrophes(team_name)

    # Search for the team ID by normalized team name
    team_id = next((id for id, name in team_map.items() if normalize_apostrophes(name) == normalized_team_name), None)

    if team_id is None:
        raise ValueError(f"Team '{team_name}' not found in the league.")

    return team_id


def get_team_lineup(entry_id, gameweek):
    """
    Fetches the lineup for a given team (entry_id) for a given gameweek, linking player IDs to player names.

    Parameters:
    - entry_id: The team ID.
    - gameweek: The gameweek number.

    Returns:
    - lineup: A list of players with their details (name, position, captain status, multiplier, etc.).
    """
    # Fetch the team lineup from the API
    lineup_url = f"https://draft.premierleague.com/api/entry/{entry_id}/event/{gameweek}"
    try:
        lineup_response = requests.get(lineup_url, timeout=30)
        lineup_data = lineup_response.json()
    except Exception as e:
        _logger.warning("Failed to fetch team lineup for entry %s GW %s: %s", entry_id, gameweek, e)
        return {}

    # Get player data (ID to name mapping)
    player_dict = get_fpl_player_mapping()

    # Extract picks (player selections)
    picks = lineup_data['picks']

    # Create a list to store the lineup details
    lineup = []

    for pick in picks:
        player_id = pick['element']
        player_name = player_dict.get(player_id, "Unknown Player")
        position = pick['position']
        is_captain = pick['is_captain']
        is_vice_captain = pick['is_vice_captain']
        multiplier = pick['multiplier']

        # Append player details to the lineup list
        lineup.append({
            'Player Name': player_name,
            'Position': position,
            'Captain': 'Yes' if is_captain else 'No',
            'Vice Captain': 'Yes' if is_vice_captain else 'No',
            'Multiplier': multiplier
        })

    return lineup


def get_team_projections(player_rankings, league_id, team_id):
    """
    Fetches projected scores for the specified team from player rankings.

    Parameters:
    - player_rankings (df): DataFrame containing player rankings and projected points.
    - league_id (int): The ID of the FPL Draft league.
    - team_id (int): ID of the team (unique identifier for the FPL team).

    Returns:
    - DataFrame of the team's players with their projected points.
    """
    # Import here to avoid circular dependency
    from scripts.common.player_matching import merge_fpl_players_and_projections

    # Step 1: Fetch team composition using the team ID
    team_players_df = get_team_composition_for_gameweek(int(league_id), int(team_id), config.CURRENT_GAMEWEEK)

    # Step 2: Merge team players with player rankings using fuzzy matching
    team_projections_df = merge_fpl_players_and_projections(team_players_df, player_rankings)

    # Step 3: Return the relevant columns with proper formatting
    return team_projections_df[['Player', 'Position', 'Points']].sort_values(by='Points', ascending=False)


# =============================================================================
# TRANSACTIONS
# =============================================================================

def get_transaction_data(league_id):
    """
    Fetches waiver transactions from the FPL Draft API and stores them in config.TRANSACTION_DATA if not already fetched.
    Returns:
    - transaction_data: A list of transactions from the API.
    """
    if config.TRANSACTION_DATA is None:  # Fetch only if not already fetched
        transaction_url = f"https://draft.premierleague.com/api/draft/league/{league_id}/transactions"
        try:
            transaction_response = requests.get(transaction_url, timeout=30)
            config.TRANSACTION_DATA = transaction_response.json()['transactions']  # Store transactions in config
        except Exception as e:
            _logger.warning("Failed to fetch transaction data for league %s: %s", league_id, e)
            return []

    return config.TRANSACTION_DATA


def get_waiver_transactions_up_to_gameweek(league_id, gameweek):
    """
    Fetches all transactions (waivers, free agent moves, etc.) up to (and including) the selected gameweek.
    Safely handles cases where the API returns transactions with a null/absent 'event'.

    Parameters:
    - league_id (int|str): The league ID for transactions.
    - gameweek (int|None): The gameweek to filter transactions up to. If None, returns only transactions
                           that have a valid integer 'event' (no upper limit).

    Returns:
    - List[dict]: Transactions up to the given gameweek with valid integer 'event' values.
    """
    # Coerce inputs
    try:
        league_id = int(league_id)
    except (TypeError, ValueError):
        raise ValueError("league_id must be an integer")

    # Coerce gameweek if provided; allow None
    if gameweek is not None:
        try:
            gameweek = int(gameweek)
        except (TypeError, ValueError):
            # If coercion fails, treat as None (no upper bound)
            gameweek = None

    # Fetch transactions
    url = f"https://draft.premierleague.com/api/draft/league/{league_id}/transactions"
    resp = requests.get(url, timeout=30)
    try:
        data = resp.json()
    except Exception:
        data = {}

    transactions = data.get('transactions', []) or []

    # Keep only transactions that have a valid integer event
    tx_with_event = [tx for tx in transactions if isinstance(tx.get('event'), int)]

    # If no upper bound, return all with valid event
    if gameweek is None:
        return tx_with_event

    # Otherwise, filter by event <= gameweek
    return [tx for tx in tx_with_event if tx['event'] <= gameweek]


def pull_fpl_player_stats():
    """
    Fetches FPL player statistics from the Classic FPL API.

    Returns:
    - player_df: DataFrame with player statistics sorted by total_points.
    """
    from scripts.common.text_helpers import remove_duplicate_words

    # Set Classic FPL API endpoint (has more comprehensive data than Draft API)
    fpl_api = 'https://fantasy.premierleague.com/api/bootstrap-static/'

    # Fetch from the endpoint
    try:
        data = requests.get(fpl_api, timeout=30)
        data_json = data.json()
    except Exception as e:
        _logger.warning("Failed to fetch FPL player stats: %s", e)
        return pd.DataFrame()

    # extracting data in json format

    # Create a dataframe for the positions ('element_types')
    position_df = pd.DataFrame.from_records(data_json['element_types'])

    # Format df
    cols = ['id', 'singular_name', 'singular_name_short']
    position_df = position_df[cols]

    # Rename columns
    position_df.columns = ['position_id', 'position_name', 'position_abbrv']

    # Create a dataframe for the teams
    team_df = pd.DataFrame.from_records(data_json['teams'])

    # Format df
    cols = ['id', 'name', 'short_name']
    team_df = team_df[cols]

    # Rename columns
    team_df.columns = ['team_id', 'team_name', 'team_name_abbrv']

    # Create a DataFrame from the Player dictionary ('elements')
    player_df = pd.DataFrame.from_records(data_json['elements'])

    # Create Full Name Column
    player_df['player'] = player_df['first_name'] + ' ' + player_df['second_name']
    # Apply the remove_duplicate_words function to the 'Player' column
    player_df['player'] = player_df['player'].apply(remove_duplicate_words)

    # Merge in team_name
    player_df = pd.merge(player_df, team_df, left_on='team', right_on='team_id')

    # Merge in position name
    player_df = pd.merge(player_df, position_df, left_on='element_type', right_on='position_id')

    # Organize columns
    cols = ['id', 'player', 'position_abbrv', 'team_name', 'team_name_abbrv', 'clean_sheets', 'goals_scored',
            'assists', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'yellow_cards',
            'starts', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'creativity', 'influence', 'threat', 'ict_index', 'bonus', 'bps', 'form', 'points_per_game', 'total_points',
            'goals_conceded', 'saves', 'now_cost', 'selected_by_percent',
            'corners_and_indirect_freekicks_order', 'corners_and_indirect_freekicks_text', 'direct_freekicks_order',
            'direct_freekicks_text', 'penalties_order', 'penalties_text', 'chance_of_playing_this_round',
            'chance_of_playing_next_round', 'status', 'news', 'news_added']
    player_df = player_df[cols]

    # Ensure expected_goal_involvements is numeric
    player_df["expected_goal_involvements"] = pd.to_numeric(
        player_df["expected_goal_involvements"], errors="coerce"
    )

    # Create actual goal involvements column and ensure that it is numeric (sum of goals + assists)
    player_df["actual_goal_involvements"] = pd.to_numeric(
        player_df["goals_scored"], errors="coerce"
    ) + pd.to_numeric(player_df["assists"], errors="coerce")

    # Sort dataframe by goals_scored
    player_df = player_df.sort_values(by='total_points', ascending=False)

    # Return df
    return (player_df)


# =============================================================================
# HEAD-TO-HEAD RECORD FUNCTIONS (DRAFT)
# =============================================================================

@st.cache_data(ttl=600)
def get_draft_league_details(league_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch full draft league details including matches, entries, and standings.

    Parameters:
    - league_id: The ID of the FPL Draft league.

    Returns:
    - Dictionary with 'matches', 'league_entries', 'standings', etc.
    - None if the request fails.
    """
    if not league_id:
        return None
    try:
        url = f"https://draft.premierleague.com/api/league/{league_id}/details"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        _logger.warning("Failed to fetch draft league details for league %s", league_id, exc_info=True)
        return None


def get_draft_h2h_record(league_id: int, team1_id: int, team2_id: int) -> Dict[str, Any]:
    """
    Calculate head-to-head record between two teams in a Draft league.

    Parameters:
    - league_id: The ID of the FPL Draft league.
    - team1_id: The entry ID of team 1 (from get_team_id_by_name).
    - team2_id: The entry ID of team 2 (from get_team_id_by_name).

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

    league_data = get_draft_league_details(league_id)
    if not league_data:
        return result

    # Build mapping from entry_id to league entry id (used in matches)
    # The API has two IDs: entry_id (used by get_team_id_by_name) and id (used in matches)
    entries = league_data.get("league_entries", [])
    entry_id_to_league_id = {entry["entry_id"]: entry["id"] for entry in entries}

    # Convert input IDs to league entry IDs
    league_team1_id = entry_id_to_league_id.get(team1_id, team1_id)
    league_team2_id = entry_id_to_league_id.get(team2_id, team2_id)

    matches = league_data.get("matches", [])
    if not matches:
        return result

    for match in matches:
        entry_1 = match.get("league_entry_1")
        entry_2 = match.get("league_entry_2")
        pts_1 = match.get("league_entry_1_points", 0)
        pts_2 = match.get("league_entry_2_points", 0)
        gw = match.get("event")

        # Skip unplayed matches
        if pts_1 == 0 and pts_2 == 0:
            continue

        # Check if this match involves both teams
        if (entry_1 == league_team1_id and entry_2 == league_team2_id):
            my_pts, opp_pts = pts_1, pts_2
        elif (entry_1 == league_team2_id and entry_2 == league_team1_id):
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


@st.cache_data(ttl=600)
def get_draft_all_h2h_records(league_id: int, team_id: int) -> List[Dict[str, Any]]:
    """
    Calculate head-to-head records for a team against all opponents in a Draft league.

    Parameters:
    - league_id: The ID of the FPL Draft league.
    - team_id: The entry ID of the team to analyze (from get_team_id_by_name).

    Returns:
    - List of dictionaries, each containing opponent info and H2H record.
    """
    league_data = get_draft_league_details(league_id)
    if not league_data:
        return []

    # Build mappings for ID conversion
    # The API has two IDs: entry_id (used by get_team_id_by_name) and id (used in matches)
    entries = league_data.get("league_entries", [])
    entry_id_to_league_id = {entry["entry_id"]: entry["id"] for entry in entries}
    league_id_to_entry_id = {entry["id"]: entry["entry_id"] for entry in entries}
    team_names = {entry["id"]: entry["entry_name"] for entry in entries}

    # Convert input team_id (entry_id) to league entry id
    league_team_id = entry_id_to_league_id.get(team_id, team_id)

    # Find all opponents (using league entry IDs from matches)
    matches = league_data.get("matches", [])
    opponents = set()

    for match in matches:
        entry_1 = match.get("league_entry_1")
        entry_2 = match.get("league_entry_2")

        if entry_1 == league_team_id:
            opponents.add(entry_2)
        elif entry_2 == league_team_id:
            opponents.add(entry_1)

    # Calculate H2H record against each opponent
    records = []
    for opp_league_id in opponents:
        # Convert opponent's league_id back to entry_id for get_draft_h2h_record
        opp_entry_id = league_id_to_entry_id.get(opp_league_id, opp_league_id)
        h2h = get_draft_h2h_record(league_id, team_id, opp_entry_id)
        records.append({
            "opponent_id": opp_league_id,
            "opponent_name": team_names.get(opp_league_id, f"Team {opp_league_id}"),
            "wins": h2h["wins"],
            "draws": h2h["draws"],
            "losses": h2h["losses"],
            "record_str": h2h["record_str"],
            "points_for": h2h["points_for"],
            "points_against": h2h["points_against"],
            "matches": h2h["matches"]
        })

    # Sort by wins descending, then by points_for descending
    records.sort(key=lambda x: (-x["wins"], -x["points_for"]))
    return records
