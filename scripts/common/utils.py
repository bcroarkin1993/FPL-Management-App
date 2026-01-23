"""
FPL Management App - Utility Functions

This module contains all utility functions organized into logical sections:
1. CONSTANTS & CONFIGURATION
2. TEXT & STRING NORMALIZATION
3. POSITION & TEAM MAPPING
4. FPL API - DRAFT LEAGUE
5. FPL API - CLASSIC MODE
6. ROTOWIRE SCRAPING
7. FIXTURES & SCHEDULING
8. PLAYER MATCHING & NORMALIZATION
9. PLAYER ANALYTICS
10. LINEUP OPTIMIZATION
"""

from bs4 import BeautifulSoup
import config
from datetime import datetime, timedelta
from fuzzywuzzy import process, fuzz
import numpy as np
import pandas as pd
import re
import requests
import streamlit as st
from typing import Dict, List, Any, Optional
import unicodedata
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from scripts.common.player_matching import canonical_normalize


# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================

# Timezone
TZ_ET = ZoneInfo("America/New_York")

# Team name mappings (RotoWire full names -> FPL short codes)
TEAM_FULL_TO_SHORT = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man Utd": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Southampton": "SOU", "Spurs": "TOT",
    "West Ham": "WHU", "Wolves": "WOL",
    # Common variations
    "Manchester City": "MCI", "Manchester United": "MUN",
    "Manchester Utd": "MUN", "Nottingham Forest": "NFO",
    "Tottenham": "TOT", "Tottenham Hotspur": "TOT",
}

# Position mappings (various formats -> G/D/M/F)
POS_MAP_TO_RW = {
    "GK": "G", "GKP": "G", "G": "G", "Goalkeeper": "G",
    "DEF": "D", "D": "D", "Defender": "D",
    "MID": "M", "M": "M", "Midfielder": "M",
    "FWD": "F", "FW": "F", "F": "F", "Forward": "F",
}


# =============================================================================
# 2. TEXT & STRING NORMALIZATION
# =============================================================================

def _clean_player_name(s: str) -> str:
    """Lowercase, remove accents and non-alphanumerics for robust matching keys."""
    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_text(x: str) -> str:
    """Lowercase, strip accents, collapse spaces for fuzzy matching."""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = " ".join(s.lower().split())
    return s


def _strip_accents(s: str) -> str:
    """Remove diacritics/accents and normalize whitespace."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


def clean_text(s: Any) -> str:
    """Clean and normalize text by collapsing whitespace."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_apostrophes(text):
    """
    Normalizes text by converting different apostrophe types to a standard straight apostrophe.

    Parameters:
    - text (str): The text to normalize.

    Returns:
    - str: The normalized text.
    """
    if text is None:
        return None
    # Normalize Unicode and replace curly apostrophes with straight apostrophes
    return unicodedata.normalize('NFKC', text).replace("\u2019", "'").strip().lower()


def normalize_name(name: str) -> str:
    """Remove diacritics, normalize spacing/case for matching."""
    if name is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r"\s+", " ", ascii_str).strip()


def remove_duplicate_words(name):
    """Function to remove duplicate consecutive words."""
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', name)


# =============================================================================
# 3. POSITION & TEAM MAPPING
# =============================================================================

def _map_position_to_rw(pos_val):
    """Map any reasonable position variant to {'G','D','M','F'}."""
    if pd.isna(pos_val):
        return ""
    p = str(pos_val).strip()
    # direct mapping
    if p in POS_MAP_TO_RW:
        return POS_MAP_TO_RW[p]

    # If it's numeric (FPL element_type: 1..4)
    if p.isdigit():
        return {"1": "G", "2": "D", "3": "M", "4": "F"}.get(p, "")

    # Heuristics
    p_up = p.upper()
    for key, val in POS_MAP_TO_RW.items():
        if p_up.startswith(key):
            return val
    return p_up[:1]  # fallback: first letter


def _to_short_team_code(team_val, teams_df=None):
    """
    Convert a team value to a 3-letter short code.
    - If `teams_df` (FPL bootstrap teams) is provided, it should contain id + short_name.
    - If `team_val` already looks like a 3-letter code, keep it.
    - Else try mapping via TEAM_FULL_TO_SHORT.
    """
    if pd.isna(team_val):
        return ""
    s = str(team_val).strip()

    # Already like 'MCI'
    if re.fullmatch(r"[A-Z]{3}", s):
        return s

    # Try dictionary mapping (RotoWire-style team strings)
    if s in TEAM_FULL_TO_SHORT:
        return TEAM_FULL_TO_SHORT[s]

    # If it's a number and we have teams_df (FPL team id path)
    if teams_df is not None:
        try:
            tid = int(s)
            row = teams_df.loc[teams_df["id"] == tid]
            if not row.empty and "short_name" in row.columns:
                return str(row.iloc[0]["short_name"])
        except Exception:
            pass

    # Best effort: return uppercase 3-letter heuristic
    guess = re.sub(r"[^A-Za-z]", "", s).upper()[:3]
    return guess if len(guess) == 3 else s


def format_team_name(name):
    """
    Formats a team name by normalizing apostrophes and capitalizing each word.

    Parameters:
    - name (str): The team name to format.

    Returns:
    - str: The formatted team name.
    """
    if name is None:
        return None
    # Normalize Unicode and replace curly apostrophes with straight apostrophes
    normalized_name = unicodedata.normalize('NFKC', name).replace("\u2019", "'").strip()
    # Capitalize the first letter of each word
    return ' '.join(word.capitalize() for word in normalized_name.split())


def position_converter(element_type):
    """Converts element type to position name."""
    return {1: 'G', 2: 'D', 3: 'M', 4: 'F'}.get(element_type, 'Unknown')


# =============================================================================
# 4. FPL API - DRAFT LEAGUE
# =============================================================================

def get_current_gameweek():
    """
    Fetches the current gameweek based on the game status from the FPL Draft API.

    Returns:
    - current_gameweek: An integer representing the current gameweek.
    """
    game_url = "https://draft.premierleague.com/api/game"
    response = requests.get(game_url)
    game_data = response.json()

    # Check if the current event is finished
    if game_data['current_event_finished']:
        current_gameweek = game_data['next_event']
    else:
        current_gameweek = game_data['current_event']

    return current_gameweek


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


def get_fpl_player_mapping():
    """
    Fetches FPL player data from the FPL Draft API and returns it as a dictionary to link player ids to player names.

    Returns:
    - fpl_player_data: Dictionary with Player_ID as key and dict with 'Player', 'Web_Name', 'Team', 'Position'.
    """
    # Fetch data from the FPL Draft API
    player_url = "https://draft.premierleague.com/api/bootstrap-static"
    response = requests.get(player_url)

    # Extract relevant player information
    player_data = response.json()
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
    league_response = requests.get(league_url).json()

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


def get_league_entries(league_id):
    """
    Fetches the league entries and creates a mapping of entry IDs to team names.

    Parameters:
    - league_id: The ID of the league.

    Returns:
    - A dictionary where keys are entry IDs, and values are team names.
    """
    url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    response = requests.get(url).json()

    return {entry['entry_id']: entry['entry_name'] for entry in response['league_entries']}


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
    draft_data = requests.get(draft_url).json()
    league_details = requests.get(league_details_url).json()
    player_data = requests.get(player_data_url).json()

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
    lineup_response = requests.get(lineup_url)
    lineup_data = lineup_response.json()

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
    # Step 1: Fetch team composition using the team ID
    team_players_df = get_team_composition_for_gameweek(int(league_id), int(team_id), config.CURRENT_GAMEWEEK)

    # Step 2: Merge team players with player rankings using fuzzy matching
    team_projections_df = merge_fpl_players_and_projections(team_players_df, player_rankings)

    # Step 3: Return the relevant columns with proper formatting
    return team_projections_df[['Player', 'Position', 'Points']].sort_values(by='Points', ascending=False)


def get_transaction_data(league_id):
    """
    Fetches waiver transactions from the FPL Draft API and stores them in config.TRANSACTION_DATA if not already fetched.
    Returns:
    - transaction_data: A list of transactions from the API.
    """
    if config.TRANSACTION_DATA is None:  # Fetch only if not already fetched
        transaction_url = f"https://draft.premierleague.com/api/draft/league/{league_id}/transactions"
        transaction_response = requests.get(transaction_url)
        config.TRANSACTION_DATA = transaction_response.json()['transactions']  # Store transactions in config

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
    resp = requests.get(url)
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
    Fetches FPL player statistics from the FPL Draft API.

    Returns:
    - player_df: DataFrame with player statistics sorted by total_points.
    """
    # Set FPL Draft API endpoint
    draft_api = 'https://draft.premierleague.com/api/bootstrap-static'

    # Test the endpoint
    data = requests.get(draft_api)

    # extracting data in json format
    data_json = data.json()

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
    cols = ['id', 'player', 'position_abbrv', 'team_name', 'clean_sheets', 'goals_scored',
            'assists', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'yellow_cards',
            'starts', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'creativity', 'influence', 'bonus', 'bps', 'form', 'points_per_game', 'total_points',
            'corners_and_indirect_freekicks_order', 'corners_and_indirect_freekicks_text', 'direct_freekicks_order',
            'direct_freekicks_text', 'penalties_order', 'penalties_text', 'chance_of_playing_this_round',
            'chance_of_playing_next_round', 'status', 'added']
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
# 5. FPL API - CLASSIC MODE
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
        return None


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
        return None


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
        return None


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
        return None


# =============================================================================
# 6. ROTOWIRE SCRAPING
# =============================================================================

def get_rotowire_player_projections(url, limit=None):
    """
    Fetches fantasy rankings and projected points for players from RotoWire.

    Parameters:
    - url (str): URL to fetch the data from.
    - limit (int, optional): Number of players to display. Defaults to None (displays all players).

    Returns:
    - DataFrame: A DataFrame containing player rankings, projected points, and calculated value.
    """
    # Download the page using the requests library
    website = requests.get(url)
    soup = BeautifulSoup(website.content, 'html.parser')

    # Isolate BeautifulSoup output to the table of interest
    my_classes = soup.find(class_='article-table__tablesorter article-table__standard article-table__figure')
    players = my_classes.find_all("td")

    # Create lists for each field to collect
    overall_rank, fw_rank, mid_rank, def_rank, gk_rank = [], [], [], [], []
    player, team, matchup, position, price, tsb, points = [], [], [], [], [], [], []

    # Loop through the list of players in batches of 12
    batch_size = 12
    for i in range(0, len(players), batch_size):
        overall_rank.append(players[i].text)
        fw_rank.append(players[i + 1].text)
        mid_rank.append(players[i + 2].text)
        def_rank.append(players[i + 3].text)
        gk_rank.append(players[i + 4].text)
        player.append(players[i + 5].text)
        team.append(players[i + 6].text)
        matchup.append(players[i + 7].text)
        position.append(players[i + 8].text)
        price.append(players[i + 9].text)
        tsb.append(players[i + 10].text)
        points.append(players[i + 11].text)

    # Create a DataFrame with formatted column names
    player_rankings = pd.DataFrame(
        list(zip(overall_rank, fw_rank, mid_rank, def_rank, gk_rank, player, team,
                 matchup, position, price, tsb, points)),
        columns=[
            'Overall Rank', 'FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank',
            'Player', 'Team', 'Matchup', 'Position', 'Price', 'TSB %', 'Points'
        ]
    )

    # Replace empty strings with 0 and convert columns to numeric where appropriate
    for col in ['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank', 'Points', 'Price']:
        player_rankings[col] = pd.to_numeric(player_rankings[col], errors='coerce').fillna(0)

    # Create 'Pos Rank' by summing the four position ranks
    player_rankings['Pos Rank'] = (
        player_rankings['FW Rank'] + player_rankings['MID Rank'] +
        player_rankings['DEF Rank'] + player_rankings['GK Rank']
    ).astype(int)

    # Drop individual position rank columns
    player_rankings.drop(columns=['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank'], inplace=True)

    # Ensure 'Price' is numeric
    player_rankings['Price'] = pd.to_numeric(player_rankings['Price'], errors='coerce').fillna(0)

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

    return player_rankings


def get_rotowire_rankings_url(current_gameweek=None, timeout=15):
    """
    Try to locate the Rotowire 'Fantasy Premier League Player Rankings: Gameweek X'
    article on the /soccer/articles/ index. Handles new slugs with extra words.

    Returns:
        str | None  -> fully qualified article URL or None if not found.
    """
    # If you have a helper, use it; otherwise leave current_gameweek optional
    if current_gameweek is None:
        try:
            current_gameweek = get_current_gameweek()  # your existing function
        except Exception:
            current_gameweek = None

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(config.ARTICLES_INDEX, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # Find any anchors whose href contains our base slug
    anchors = soup.select('a[href*="fantasy-premier-league-player-rankings-gameweek-"]')

    # Regex: capture GW and the trailing numeric article id
    # works for both:
    # ...gameweek-9-86285
    # ...gameweek-1-fpl-west-ham-jarrod-bowen-95015
    pat = re.compile(
        r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$"
    )

    candidates = []
    for a in anchors:
        href = a.get("href", "").strip()
        if not href:
            continue
        m = pat.search(href)
        if m:
            gw = int(m.group(1))
            art_id = int(m.group(2))
            candidates.append((gw, art_id, urljoin(config.ARTICLES_INDEX, href)))

    if not candidates:
        return None

    if current_gameweek is not None:
        # Prefer exact gameweek; if multiple, highest article id
        exact = [c for c in candidates if c[0] == current_gameweek]
        if exact:
            return max(exact, key=lambda x: x[1])[2]

        # Else pick closest GW; break ties by newest article id
        return min(candidates, key=lambda x: (abs(x[0] - current_gameweek), -x[1]))[2]

    # If we don't know the GW, return the newest relevant article by id
    return max(candidates, key=lambda x: x[1])[2]


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
# 7. FIXTURES & SCHEDULING
# =============================================================================

def _bootstrap_teams_df() -> pd.DataFrame:
    """Return FPL bootstrap teams as a 2-col DF: id, short_name."""
    try:
        resp = requests.get("https://draft.premierleague.com/api/bootstrap-static", timeout=20)
        resp.raise_for_status()
        teams = resp.json().get("teams", [])
        return pd.DataFrame(teams)[["id", "short_name"]]
    except Exception:
        # Fallback: empty DF (the normalizer will still attempt heuristics)
        return pd.DataFrame(columns=["id", "short_name"])


def get_earliest_kickoff_et(gw: int) -> datetime:
    """
    Pull fixtures for the given GW from the classic FPL endpoint and
    return the earliest kickoff in ET.
    """
    r = requests.get(config.FPL_FIXTURES_BY_EVENT.format(gw=gw), timeout=20)
    r.raise_for_status()
    fixtures = r.json()
    # Filter fixtures that actually have a kickoff_time
    times = []
    for fx in fixtures:
        k = fx.get("kickoff_time")
        if not k:
            continue
        # k is ISO string in UTC, e.g., "2024-12-03T19:30:00Z"
        # Normalize 'Z' to '+00:00' for fromisoformat
        k2 = k.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(k2)
        times.append(dt_utc.astimezone(TZ_ET))
    if not times:
        raise RuntimeError(f"No kickoff times found for GW {gw}")
    return min(times)


def get_fixture_difficulty_grid(weeks: int = 6):
    """
    Returns:
      disp  : display DF with first col 'Team', then GW columns (strings like 'WHU (H)')
      diffs : numeric DF (index=team short, cols=GW) with difficulty (1..5, avg if DGW)
      avg   : Series of per-team average difficulty across horizon (NaN->3 neutral)
    """
    current_gw = int(get_current_gameweek())

    teams = _bootstrap_teams_df()  # id, short_name
    id2short = {int(r.id): str(r.short_name) for _, r in teams.iterrows()}

    cols = [f"GW{gw}" for gw in range(current_gw, current_gw + weeks)]
    idx = [id2short[i] for i in sorted(id2short)]
    disp_core = pd.DataFrame("—", index=idx, columns=cols)
    diffs = pd.DataFrame(np.nan, index=idx, columns=cols)

    def _fixtures_for_event(gw: int):
        """
        Return fixtures for a single gameweek `gw` from the canonical FPL endpoint.
        Uses explicit query params so the server actually filters by GW.
        """
        url = "https://fantasy.premierleague.com/api/fixtures/"
        headers = {"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
        try:
            r = requests.get(url, params={"event": int(gw)}, headers=headers, timeout=20)
            r.raise_for_status()
            js = r.json()
            # This endpoint returns only the requested GW as a list
            return js if isinstance(js, list) else []
        except Exception:
            return []

    # Fill each team's own schedule per GW (no carry-over across columns)
    for gw, col in zip(range(current_gw, current_gw + weeks), cols):
        fx = _fixtures_for_event(gw)
        for f in fx:
            h, a = f.get("team_h"), f.get("team_a")
            dh, da = f.get("team_h_difficulty"), f.get("team_a_difficulty")
            if h is None or a is None:
                continue
            hs, as_ = id2short.get(int(h), str(h)), id2short.get(int(a), str(a))

            # Home team cell
            prev_h = disp_core.at[hs, col]
            disp_core.at[hs, col] = f"{as_} (H)" if prev_h == "—" else f"{prev_h} / {as_} (H)"
            diffs.at[hs, col] = np.nanmean([diffs.at[hs, col], float(dh) if dh is not None else np.nan])

            # Away team cell
            prev_a = disp_core.at[as_, col]
            disp_core.at[as_, col] = f"{hs} (A)" if prev_a == "—" else f"{prev_a} / {hs} (A)"
            diffs.at[as_, col] = np.nanmean([diffs.at[as_, col], float(da) if da is not None else np.nan])

    # Sort rows by easiest average run (NaN -> neutral 3)
    avg = diffs.fillna(3).mean(axis=1)
    order = avg.sort_values().index
    disp_core = disp_core.loc[order]
    diffs = diffs.loc[order]
    avg = avg.loc[order]

    # Add sticky Team column for Y-axis labels
    disp = disp_core.copy()
    disp.insert(0, "Team", disp.index)
    disp["Avg FDR"] = avg.round(2)
    return disp, diffs, avg


def get_next_transaction_deadline(offset_hours: int, gw: int):
    """
    Returns (deadline_et, gameweek) where deadline = earliest kickoff - offset_hours.
    Uses ET, leveraging your get_earliest_kickoff_et(gw).
    """
    if offset_hours is None:
        offset_hours = getattr(config, "TRANSACTION_DEADLINE_HOURS_BEFORE_KICKOFF", 24)
    if gw is None:
        gw = int(get_current_gameweek())
    kickoff_et = get_earliest_kickoff_et(gw)
    return kickoff_et - timedelta(hours=offset_hours), gw


def style_fixture_difficulty(disp: pd.DataFrame, diffs: pd.DataFrame):
    """
    Return a Styler with colors per difficulty.
    'Team' column is left uncolored (acts as row labels).
    """
    PALETTE = {1: "#86efac", 2: "#bbf7d0", 3: "#e5e7eb", 4: "#fecaca", 5: "#b91c1c"}
    gw_cols = [c for c in disp.columns if c not in ("Team", "Avg FDR")]  # only color GW cells

    def _cell_styles(_):
        S = pd.DataFrame("", index=disp.index, columns=disp.columns)
        for r in disp.index:
            for c in gw_cols:
                d = diffs.at[r, c]
                k = 3 if pd.isna(d) else max(1, min(5, int(round(float(d)))))
                color = PALETTE[k]
                txt = "#ffffff" if k == 5 else "#111111"
                S.at[r, c] = f"background-color:{color};color:{txt};text-align:center;font-weight:600;"
        # keep Team column readable
        S["Team"] = "font-weight:700;text-align:left;"
        return S

    sty = (
        disp.style
            .apply(_cell_styles, axis=None)
            .set_properties(subset=gw_cols,
                            **{"border": "1px solid #ddd", "white-space": "nowrap", "font-size": "0.9rem"})
            .set_properties(subset=["Team"],
                            **{"border": "1px solid #ddd", "position": "sticky", "left": "0", "background": "#fff",
                               "font-weight": "700", "text-align": "left"})
            .set_properties(subset=["Avg FDR"],
                            **{"border": "1px solid #ddd", "font-weight": "700", "text-align": "right"})
            .set_table_styles(
            [{"selector": "th", "props": [("position", "sticky"), ("top", "0"), ("background", "#fff")]}])
            .set_table_attributes('class="fdr-table" style="width:100%;"')
    )
    return sty


# =============================================================================
# 8. PLAYER MATCHING & NORMALIZATION
# =============================================================================

def _backfill_player_ids(roster_df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where Player_ID is NaN, fill via fuzzy match against fpl_stats
    constrained by Team and Position (prefer exact team+pos matches).
    Safe if any expected columns are missing.
    """
    df = roster_df.copy()

    # Ensure required columns exist
    for col in ("Player", "Team", "Position"):
        if col not in df.columns:
            df[col] = np.nan
    if "Player_ID" not in df.columns:
        df["Player_ID"] = np.nan

    # Prepare candidate table safely
    cand = fpl_stats.copy()
    for col in ("Player", "Team", "Position", "Player_ID"):
        if col not in cand.columns:
            cand[col] = np.nan

    # Normalized names for matching (uses module-level _norm_text)
    cand["__name_norm"] = cand["Player"].apply(_norm_text)
    df["__name_norm"] = df["Player"].apply(_norm_text)

    # Indices to backfill
    try:
        missing_idx = df[df["Player_ID"].isna()].index
    except KeyError:
        # Safety: create it and mark all as missing
        df["Player_ID"] = np.nan
        missing_idx = df.index

    for i in missing_idx:
        name_norm = df.at[i, "__name_norm"]
        team = df.at[i, "Team"]
        pos = df.at[i, "Position"]

        # Try team+pos scope; then pos; then all
        scope = cand[(cand["Team"] == team) & (cand["Position"] == pos)]
        if scope.empty:
            scope = cand[cand["Position"] == pos]
        if scope.empty:
            scope = cand

        if scope.empty or scope["__name_norm"].isna().all():
            continue

        match = process.extractOne(name_norm, scope["__name_norm"].dropna().tolist(), scorer=fuzz.WRatio)
        if match:
            m_name, score = match[0], match[1]
            if score >= 85:
                pid = scope.loc[scope["__name_norm"] == m_name, "Player_ID"]
                if not pid.empty and pd.notna(pid.iloc[0]):
                    df.at[i, "Player_ID"] = float(pid.iloc[0])

    return df.drop(columns=["__name_norm"], errors="ignore")


def _fuzzy_match_player(fpl_player, fpl_team, fpl_position, candidates, projections_df,
                        fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Consolidated fuzzy matching function for player names.

    Finds the best match for a player using fuzzy matching with context-aware thresholds.
    If team and position match, uses a lower threshold; otherwise requires higher confidence.

    Parameters:
    - fpl_player: Player name to match.
    - fpl_team: Player's team.
    - fpl_position: Player's position.
    - candidates: List of candidate player names.
    - projections_df: DataFrame with projection data containing 'Player', 'Team', 'Position'.
    - fuzzy_threshold: Default threshold for matches (default: 80).
    - lower_fuzzy_threshold: Threshold when team+position match (default: 60).

    Returns:
    - Matched player name or None if no good match found.
    """
    if not candidates:
        return None

    result = process.extractOne(str(fpl_player), candidates)
    if not result:
        return None

    match, score = result[0], result[1]

    matched_row = projections_df[projections_df['Player'] == match]
    if not matched_row.empty:
        match_team = matched_row.iloc[0]['Team']
        match_position = matched_row.iloc[0]['Position']

        # Lower threshold if team and position match
        if match_team == fpl_team and match_position == fpl_position and score >= lower_fuzzy_threshold:
            return match

    # Higher threshold for general matches
    if score >= fuzzy_threshold:
        return match

    return None


def clean_fpl_player_names(fpl_players_df, projections_df, fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Cleans the player names in the FPL DataFrame by replacing them with their best matches from the projections DataFrame.

    Parameters:
    - fpl_players_df: DataFrame with FPL players ['Player', 'Team', 'Position'].
    - projections_df: DataFrame with Rotowire projections ['Player', 'Team', 'Position'].
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.

    Returns:
    - fpl_players_df: Updated FPL DataFrame with cleaned player names.
    """
    # Extract candidate names from projections
    projection_names = projections_df['Player'].tolist()

    def find_best_match(row):
        result = _fuzzy_match_player(
            row['player'], row['team_name'], row['position_abbrv'],
            projection_names, projections_df,
            fuzzy_threshold, lower_fuzzy_threshold
        )
        return result if result else row['player']

    # Update FPL DataFrame with cleaned player names
    fpl_players_df['player'] = fpl_players_df.apply(find_best_match, axis=1)

    return fpl_players_df


def find_best_match(fpl_player, fpl_team, fpl_position, candidates, projections_df,
                    fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Finds the best match for a player using fuzzy matching.

    Note: This is a public wrapper around _fuzzy_match_player for backward compatibility.

    Parameters:
    - fpl_player: Player name to match.
    - fpl_team: Player's team.
    - fpl_position: Player's position.
    - candidates: List of candidate names.
    - projections_df: DataFrame with Rotowire projections.
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.

    Returns:
    - Matched player name or None.
    """
    return _fuzzy_match_player(fpl_player, fpl_team, fpl_position, candidates, projections_df,
                               fuzzy_threshold, lower_fuzzy_threshold)


def merge_fpl_players_and_projections(fpl_players_df, projections_df,
                                      fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Robust merge of FPL players (Player/Team/Position) with projections.
    - Normalizes projections_df to RotoWire schema inside the function.
    - Uses canonical name normalization (strips accents) for reliable matching.
    - Tries exact match on normalized names first, then falls back to fuzzy matching.
    - Returns a table with players that *did* or *did not* match; unmatched get NA fields.
    """

    # -------- normalize projections to RW schema --------
    def _normalize_proj(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # case-insensitive rename
        rename = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ('player', 'name', 'player_name'):                     rename[c] = 'Player'
            elif lc in ('team', 'team_short', 'teamname', 'team_name'):     rename[c] = 'Team'
            elif lc in ('matchup', 'fixture', 'opp', 'opponent'):           rename[c] = 'Matchup'
            elif lc in ('position', 'pos'):                                 rename[c] = 'Position'
            elif lc in ('points', 'point', 'proj', 'projection'):           rename[c] = 'Points'
            elif lc in ('pos rank','pos_rank','position rank','position_rank'):
                rename[c] = 'Pos Rank'
            elif lc in ('price',):                                          rename[c] = 'Price'
            elif lc in ('tsb','tsb%','tsb %','ownership'):                  rename[c] = 'TSB %'
        if rename:
            df = df.rename(columns=rename)

        # ensure required columns exist
        req_defaults = {
            'Player': None,
            'Team': None,
            'Matchup': '',
            'Position': None,
            'Points': np.nan,
            'Pos Rank': 'NA'
        }
        for k, v in req_defaults.items():
            if k not in df.columns:
                df[k] = v

        # optional (don't fail if they don't exist)
        if 'Price' not in df.columns:
            df['Price'] = np.nan
        if 'TSB %' not in df.columns:
            df['TSB %'] = np.nan

        # numeric coercions
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
        # leave Pos Rank as-is; we'll handle at the end

        # return only the columns we use/expect
        keep = ['Player','Team','Matchup','Position','Points','Pos Rank','Price','TSB %']
        return df[[c for c in keep if c in df.columns]]

    proj_norm = _normalize_proj(projections_df)

    # guard: candidates for fuzzy
    if 'Player' not in proj_norm.columns:
        raise ValueError(f"Normalized projections missing 'Player' column. Have: {list(proj_norm.columns)}")

    # Add normalized name column for matching (strips accents, lowercase, etc.)
    proj_norm['__norm_name'] = proj_norm['Player'].apply(canonical_normalize)

    # Build lookup dict: normalized_name -> list of original Player names
    norm_to_players = {}
    for _, row in proj_norm.iterrows():
        norm = row['__norm_name']
        player = row['Player']
        if norm and pd.notna(player):
            if norm not in norm_to_players:
                norm_to_players[norm] = []
            if player not in norm_to_players[norm]:
                norm_to_players[norm].append(player)

    # Normalized candidates for fuzzy matching fallback
    normalized_candidates = list(norm_to_players.keys())

    # Build team-filtered lookup for prioritized matching
    # Maps (norm_name, team) -> list of original Player names
    norm_team_to_players = {}
    for _, row in proj_norm.iterrows():
        norm = row['__norm_name']
        player = row['Player']
        team = str(row.get('Team', ''))
        if norm and pd.notna(player):
            key = (norm, team)
            if key not in norm_team_to_players:
                norm_team_to_players[key] = []
            if player not in norm_team_to_players[key]:
                norm_team_to_players[key].append(player)

    # -------- matching strategy --------
    def _best_match(fpl_player, fpl_team, fpl_position):
        """
        Match strategy:
        1. Try exact match on canonically normalized name (O(1) lookup)
        2. Try fuzzy match WITHIN the same team first (prioritize team context)
        3. Fall back to fuzzy match across all players
        Uses lower threshold when team+position agree for context-aware matching.
        """
        if not normalized_candidates:
            return None

        # Normalize the FPL player name
        fpl_norm = canonical_normalize(str(fpl_player))
        if not fpl_norm:
            return None

        fpl_team_str = str(fpl_team) if fpl_team else ''

        # Step 1: Try exact match on normalized name
        if fpl_norm in norm_to_players:
            exact_players = norm_to_players[fpl_norm]
            # If only one match, use it
            if len(exact_players) == 1:
                return exact_players[0]
            # Multiple matches: prefer one with matching team+position
            for player in exact_players:
                rows = proj_norm[proj_norm['Player'] == player]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                if str(row.get('Team')) == fpl_team_str and str(row.get('Position')) == str(fpl_position):
                    return player
            # No team+position match, return first
            return exact_players[0]

        # Step 2: Try fuzzy match WITHIN the same team first
        # This ensures "Carlos Henrique Casimiro" (MUN) matches "Casemiro" (MUN) not "Carlos Baleba" (BHA)
        same_team_candidates = [
            norm for norm in normalized_candidates
            if (norm, fpl_team_str) in norm_team_to_players
        ]

        if same_team_candidates:
            res = process.extractOne(fpl_norm, same_team_candidates)
            if res:
                matched_norm, score = res[0], res[1]
                if score >= lower_fuzzy_threshold:
                    # Found a good match within the same team
                    original_players = norm_team_to_players.get((matched_norm, fpl_team_str), [])
                    if original_players:
                        return original_players[0]

        # Step 3: Fall back to fuzzy match across all players
        res = process.extractOne(fpl_norm, normalized_candidates)
        if not res:
            return None
        matched_norm, score = res[0], res[1]

        # Get the original player name(s) for this normalized name
        original_players = norm_to_players.get(matched_norm, [])
        if not original_players:
            return None

        # Find best matching player considering team+position
        for player in original_players:
            rows = proj_norm[proj_norm['Player'] == player]
            if rows.empty:
                continue
            row = rows.iloc[0]
            match_team = row.get('Team')
            match_pos = row.get('Position')

            # If team+position agrees, allow lower threshold
            if (str(match_team) == fpl_team_str) and (str(match_pos) == str(fpl_position)):
                if score >= lower_fuzzy_threshold:
                    return player

        # No team+position match, use stricter threshold
        if score >= fuzzy_threshold:
            return original_players[0]

        return None

    # -------- iterate FPL rows and assemble output --------
    out = []
    # case-insensitive accessors for FPL columns
    fpl_cols = {c.lower(): c for c in fpl_players_df.columns}
    need = {'player','team','position'}
    missing = need - set(fpl_cols.keys())
    if missing:
        raise ValueError(
            "fpl_players_df must include columns ['Player','Team','Position'] "
            f"(case-insensitive). Missing: {sorted(missing)}. "
            f"Columns seen: {list(fpl_players_df.columns)}"
        )

    for _, r in fpl_players_df.iterrows():
        fpl_player = r[fpl_cols['player']]
        fpl_team = r[fpl_cols['team']]
        fpl_position = r[fpl_cols['position']]

        match = _best_match(fpl_player, fpl_team, fpl_position)

        if match:
            mrow = proj_norm.loc[proj_norm['Player'] == match].iloc[0]
            out.append({
                'Player': mrow.get('Player'),
                'Team': mrow.get('Team'),
                'Matchup': mrow.get('Matchup', ''),
                'Position': mrow.get('Position'),
                'Price': mrow.get('Price', np.nan),
                'TSB %': mrow.get('TSB %', np.nan),
                'Points': mrow.get('Points'),
                'Pos Rank': mrow.get('Pos Rank', 'NA')
            })
        else:
            # keep the original FPL row but with NA projections
            out.append({
                'Player': fpl_player,
                'Team': fpl_team,
                'Matchup': 'N/A',
                'Position': fpl_position,
                'Price': np.nan,
                'TSB %': np.nan,
                'Points': np.nan,
                'Pos Rank': 'NA'
            })

    merged = pd.DataFrame(out)

    # clean Pos Rank => ints or 'NA'
    pr = pd.to_numeric(merged['Pos Rank'], errors='coerce')
    pr = pr.round().astype('Int64')  # pandas nullable int
    # convert to object with 'NA' for missing
    merged['Pos Rank'] = pr.astype(object).where(pr.notna(), 'NA')

    # final column order (don't fail if some are missing)
    order = ['Player','Team','Matchup','Position','Price','TSB %','Points','Pos Rank']
    merged = merged[[c for c in order if c in merged.columns]]

    # 1-based index
    merged.index = pd.RangeIndex(start=1, stop=len(merged) + 1, step=1)
    return merged


def normalize_for_merge(fpl_df: pd.DataFrame,
                        rotowire_df: pd.DataFrame,
                        teams_df: pd.DataFrame = None):
    """
    Convenience wrapper: returns (fpl_norm, rw_norm) aligned to the same
    schema so downstream code can rely on ['Player','Team','Position'] and helpers.
    """
    fpl_norm = normalize_fpl_players_to_rotowire_schema(fpl_df, teams_df=teams_df)
    rw_norm = normalize_rotowire_players(rotowire_df)

    # IMPORTANT: align Team to the same representation (short codes).
    # RotoWire often uses full names; we add Team_Short there too, and then
    # overwrite 'Team' to be the short code for both dataframes to make
    # equality checks reliable.
    fpl_norm["Team"] = fpl_norm["Team_Short"]
    rw_norm["Team"] = rw_norm["Team_Short"]

    return fpl_norm, rw_norm


def normalize_fpl_players_to_rotowire_schema(fpl_df: pd.DataFrame,
                                             teams_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert an FPL players DataFrame (from bootstrap-static or your in-app tables) into
    the RotoWire-aligned schema.

    Expected columns (we handle flexible inputs):
      - Player name source: ('Player') OR ('first_name' + 'second_name') OR ('web_name')
      - Team source: ('Team') OR numeric `team` id (with teams_df) OR already short code
      - Position source: ('Position') OR numeric `element_type` OR text variants.

    Returns DF with at least: ['Player','Team','Position','Player_ID','Team_Short','Player_Clean']
    (we keep other columns you had, too).
    """
    df = fpl_df.copy()

    # Column finder (case-insensitive)
    cmap = {c.lower(): c for c in df.columns}
    def _c(name):
        return cmap.get(name.lower())

    # --- Player name ---
    player_col = _c("Player")
    if player_col is None:
        fn, sn, wn = _c("first_name"), _c("second_name"), _c("web_name")
        if fn and sn:
            df["Player"] = (df[fn].astype(str).str.strip() + " " + df[sn].astype(str).str.strip()).str.strip()
        elif wn:
            df["Player"] = df[wn].astype(str).str.strip()
        else:
            raise ValueError("FPL df needs 'Player' or ('first_name' and 'second_name') or 'web_name'.")

    # --- Team ---
    # Prefer short codes; if the frame has numeric 'team' id + teams_df (bootstrap teams), we can map.
    if _c("Team") is None:
        if _c("team") is not None:
            # numeric team id -> short code via teams_df
            if teams_df is None or not {"id", "short_name"}.issubset(set(teams_df.columns)):
                # fallback: try to coerce directly
                df["Team"] = df[_c("team")].apply(lambda v: _to_short_team_code(v, teams_df=None))
            else:
                df["Team"] = df[_c("team")].apply(lambda v: _to_short_team_code(v, teams_df=teams_df))
        else:
            # If there's no team col at all, create empty; you can fill later
            df["Team"] = ""

    # --- Position ---
    if _c("Position") is None:
        pos_src = _c("element_type") or _c("pos") or _c("position_abbrv") or _c("position")
        if pos_src is None:
            raise ValueError("FPL df needs 'Position' or a mappable source like 'element_type' or 'pos'.")
        df["Position"] = df[pos_src].apply(_map_position_to_rw)

    # --- Player_ID (optional if present) ---
    pid_col = _c("id") or _c("player_id")
    if pid_col:
        df.rename(columns={pid_col: "Player_ID"}, inplace=True)

    # Normalize values / helpers
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Team_Short"] = df["Team"].apply(_to_short_team_code)
    df["Player_Clean"] = df["Player"].map(_clean_player_name)

    # Reorder helpful basics at front if present
    front = [c for c in ["Player_ID","Player","Team","Position","Team_Short","Player_Clean"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df


def normalize_rotowire_players(rotowire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take your RotoWire projections DataFrame in any reasonable form and return:
    columns ['Player','Team','Position', ...], plus helper columns:
    - 'Team_Short' (3-letter code)
    - 'Player_Clean' (accent- and punctuation-stripped lower key)
    No columns are dropped; we just add/fix and standardize casing.
    """
    df = rotowire_df.copy()

    # Unify column names (case-insensitive)
    colmap = {c.lower(): c for c in df.columns}
    def _get(col):
        for k in colmap:
            if k == col.lower():
                return colmap[k]
        return None

    # Ensure the three core columns exist
    pcol = _get("player") or _get("name")
    tcol = _get("team")
    ccol = _get("position") or _get("pos")

    if pcol is None or tcol is None or ccol is None:
        raise ValueError("RotoWire df must contain columns for player, team, and position.")

    # Standardize names
    if pcol != "Player":
        df.rename(columns={pcol: "Player"}, inplace=True)
    if tcol != "Team":
        df.rename(columns={tcol: "Team"}, inplace=True)
    if ccol != "Position":
        df.rename(columns={ccol: "Position"}, inplace=True)

    # Normalize values
    df["Player"] = df["Player"].astype(str).map(lambda s: s.strip())
    df["Team_Short"] = df["Team"].apply(_to_short_team_code)
    df["Position"] = df["Position"].apply(_map_position_to_rw)
    df["Player_Clean"] = df["Player"].map(_clean_player_name)

    return df


# =============================================================================
# 9. PLAYER ANALYTICS
# =============================================================================

def _add_fdr_and_form(
    df: pd.DataFrame,
    fpl_player_statistics_df: pd.DataFrame,
    current_gw: int,
    weeks: int
) -> pd.DataFrame:
    """
    Join AvgFDRNextN and Form onto df.

    - Requires df to have Player/Team/Position (will create if missing).
    - Merges in Player_ID, then computes Form via element-summary.
    - Fallback chain for Form: element-summary -> FPL 'form' -> 'points_per_game' -> 0.

    NOTE: This function references _avg_fdr_for_team and _avg_form_last_n which
    are not yet implemented. It will fail at runtime if called until those are added.
    """
    base = df.copy()
    # Ensure join keys exist
    for col in ("Player", "Team", "Position"):
        if col not in base.columns:
            base[col] = np.nan

    # Safely select merge cols from stats
    stats = fpl_player_statistics_df.copy()
    for col in ("Player", "Team", "Position", "Player_ID", "form", "points_per_game"):
        if col not in stats.columns:
            stats[col] = np.nan

    # Merge in Player_ID + FPL fallback stats
    base = base.merge(
        stats[["Player", "Team", "Position", "Player_ID", "form", "points_per_game"]],
        on=["Player", "Team", "Position"],
        how="left"
    )

    # Ensure Player_ID exists (for downstream)
    if "Player_ID" not in base.columns:
        base["Player_ID"] = np.nan

    # Compute Avg FDR next N GWs
    # NOTE: _avg_fdr_for_team is not implemented - this will fail at runtime
    base["AvgFDRNextN"] = base["Team"].apply(lambda t: _avg_fdr_for_team(str(t), current_gw, weeks))
    base["AvgFDRNextN"] = pd.to_numeric(base["AvgFDRNextN"], errors="coerce")

    # Robust Form calculation with fallbacks
    def _safe_form(pid, fallback_form, fallback_ppg):
        # element-summary average of last N
        val = None
        if pd.notna(pid):
            try:
                # NOTE: _avg_form_last_n is not implemented - this will fail at runtime
                val = _avg_form_last_n(int(pid), config.FORM_LOOKBACK_WEEKS)
            except Exception:
                val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_form, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_ppg, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        return float(val)

    base["Form"] = base.apply(
        lambda r: _safe_form(r.get("Player_ID"), r.get("form"), r.get("points_per_game")),
        axis=1
    )
    base["Form"] = pd.to_numeric(base["Form"], errors="coerce").fillna(0.0)

    return base


def apply_availability_penalty(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """
    Multiply a score column by (PlayPct/100) so low availability downweights adds/drops.
    """
    out = df.copy()
    out[out_col] = pd.to_numeric(out[score_col], errors="coerce") * (pd.to_numeric(out["PlayPct"], errors="coerce")/100.0)
    return out


def attach_availability(df: pd.DataFrame, avail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge PlayPct/StatusBucket/News onto df using Player+Team first,
    then try Player_ID if present.
    """
    base = df.copy()
    cols_keep = ["PlayPct","StatusBucket","News"]
    # Try exact Player+Team first (your normalizers map Team to short codes)
    base = base.merge(
        avail_df[["Player","Team","PlayPct","StatusBucket","News"]],
        on=["Player","Team"], how="left", suffixes=("","")
    )
    # If still missing and we have IDs, try by Player_ID
    if "Player_ID" in base.columns and "Player_ID" in avail_df.columns:
        mask = base["PlayPct"].isna()
        if mask.any():
            left = base.loc[mask, ["Player_ID"]].copy()
            right = avail_df[["Player_ID"] + cols_keep].copy()
            joined = left.merge(right, on="Player_ID", how="left")
            base.loc[mask, cols_keep] = joined[cols_keep].values

    # Fill PlayPct neutral defaults
    base["PlayPct"] = pd.to_numeric(base["PlayPct"], errors="coerce").fillna(50.0)
    base["StatusBucket"] = base["StatusBucket"].fillna("Questionable")
    base["News"] = base["News"].fillna("")
    return base


def team_optimizer(player_rankings):
    """
    Optimize team lineup based on player rankings.

    NOTE: This function references config.TEAM_LIST which may not exist.
    """
    # Loop over each team in the team_list
    for team_name, team_data in config.TEAM_LIST.items():
        # Convert the JSON to a DataFrame
        team_player_list = []
        for position, players in team_data.items():
            for player in players:
                team_player_list.append({'player_name': player, 'position': position})

        team_df = pd.DataFrame(team_player_list)

        # Perform a left join, filling missing Pts with 0
        merged_df = team_df.merge(player_rankings, left_on='player_name', right_on='Player', how='left')

        # Fill NaN values in the Pts column with 0
        merged_df['Pts'] = merged_df['Pts'].fillna(0)

        # Select relevant columns for the final output and rename them
        final_df = merged_df[['player_name', 'position', 'Pts']]
        final_df.columns = ['Player', 'Position', 'Projected_Points']

        # Ensure the 'Projected_Points' column is numeric
        final_df.loc[:, 'Projected_Points'] = pd.to_numeric(final_df['Projected_Points'], errors='coerce')

        # Return the DataFrame
        return(final_df)


# =============================================================================
# 10. LINEUP OPTIMIZATION
# =============================================================================

def check_valid_lineup(df):
    """
    Given a dataframe with a lineup, check to see if it is a valid lineup.
    Requirements:
    - 11 total players
    - 1 GK
    - Min of 3 DEF
    - Max of 5 DEF
    - Min of 3 MID
    - Max of 5 MID
    - Min of 1 FWD
    - Max of 3 FWD
    """
    # Check the total players count
    players = len(df)

    # Count occurrences of each value in the 'Position' column
    position_counts = df['position'].value_counts()

    # Perform the checks
    player_check = players == 11
    gk_check = position_counts['G'] == 1
    def_check = position_counts['D'] >= 3 and position_counts['D'] <= 5
    mid_check = position_counts['M'] >= 3 and position_counts['M'] <= 5
    fwd_check = position_counts['F'] >= 1 and position_counts['F'] <= 3

    # Lineup is valid is all checks are true
    return (player_check & gk_check & def_check & mid_check & fwd_check)


def find_optimal_lineup(df):
    """
    Function to find a team's optimal lineup given their player_projections_df
    :param df: a dataframe of the team's player projections
    :return: optimal 11-player lineup DataFrame
    """
    # 1. Find the top scoring GK
    top_gk = df[df['Position'] == 'G'].sort_values(by='Points', ascending=False).head(1)

    # 2. Find the top 3 scoring DEF
    top_def = df[df['Position'] == 'D'].sort_values(by='Points', ascending=False).head(3)

    # 3. Find the top 3 scoring MID
    top_mid = df[df['Position'] == 'M'].sort_values(by='Points', ascending=False).head(3)

    # 4. Find the top scoring FWD
    top_fwd = df[df['Position'] == 'F'].sort_values(by='Points', ascending=False).head(1)

    # 5. Combine the selected players
    selected_players = pd.concat([top_gk, top_def, top_mid, top_fwd])

    # 6. Find the remaining top 3 scoring players not in the selected players
    remaining_players = df[~df['Player'].isin(selected_players['Player'])]
    top_remaining = remaining_players.sort_values(by='Points', ascending=False).head(3)

    # 7. Combine all selected players
    final_selection = pd.concat([selected_players, top_remaining])

    # 8. Organize the final selection by Position and descending Projected_Points
    final_selection = final_selection.sort_values(
        by=['Position', 'Points'],
        key=lambda x: x.map({'G': 0, 'D': 1, 'M': 2, 'F': 3}),
        ascending=[True, False]
    ).reset_index(drop=True)

    # Display the final selection
    return(final_selection)
