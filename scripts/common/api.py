# scripts/common/api.py

import config
import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

# Constants
TZ_ET = ZoneInfo("America/New_York")
FPL_CLASSIC_BASE = "https://fantasy.premierleague.com/api"
FPL_DRAFT_BASE = "https://draft.premierleague.com/api"


# ==============================================================================
# GLOBAL / SHARED ENDPOINTS
# ==============================================================================

def get_current_gameweek() -> int:
    """
    Fetches the current gameweek from the Draft API (works for Classic too).
    Checks 'current_event_finished' to decide if we are in the current or next GW.
    """
    url = f"{FPL_DRAFT_BASE}/game"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('current_event_finished'):
            return data['next_event']
        return data['current_event']
    except Exception as e:
        print(f"Error fetching GW: {e}")
        # Fallback to 1 if API fails
        return 1


def get_earliest_kickoff_et(gw: int) -> datetime:
    """
    Pull fixtures for the given GW from the classic FPL endpoint and
    return the earliest kickoff in ET. Used for deadline calculation.
    """
    # Uses the fixture URL from config (e.g., fixtures/?event={gw})
    url = config.FPL_FIXTURES_BY_EVENT.format(gw=gw)

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        fixtures = r.json()

        times = []
        for fx in fixtures:
            k = fx.get("kickoff_time")
            if not k:
                continue
            # Normalize 'Z' to '+00:00' for isoformat
            k2 = k.replace("Z", "+00:00")
            dt_utc = datetime.fromisoformat(k2)
            times.append(dt_utc.astimezone(TZ_ET))

        if not times:
            raise RuntimeError(f"No kickoff times found for GW {gw}")

        return min(times)

    except Exception as e:
        st.error(f"Error fetching kickoffs: {e}")
        # Return a safe fallback (now) to prevent crash
        return datetime.now(TZ_ET)


def pull_fpl_player_stats() -> Dict[str, Any]:
    """
    Fetches the 'bootstrap-static' endpoint containing all Players, Teams, and Positions.
    Returns raw JSON.
    """
    url = f"{FPL_DRAFT_BASE}/bootstrap-static"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# ==============================================================================
# CLASSIC MODE ENDPOINTS
# ==============================================================================

def get_classic_league_standings(league_id: int, page: int = 1) -> pd.DataFrame:
    """
    Fetches the standings for a specific Classic League.
    Returns a minimally processed DataFrame for display.
    """
    url = f"{FPL_CLASSIC_BASE}/leagues-classic/{league_id}/standings/?page_new_entries=1&page_standings={page}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        standings_data = data.get('standings', {}).get('results', [])

        if not standings_data:
            return pd.DataFrame()

        df = pd.DataFrame(standings_data)

        # Renaming for UI consistency
        df = df.rename(columns={
            'rank': 'Rank',
            'entry_name': 'Team',
            'player_name': 'Manager',
            'event_total': 'GW Points',
            'total': 'Total Points',
            'entry': 'Team_ID'
        })

        # Calculate simple movement
        df['last_rank'] = df['last_rank'].replace(0, df['Rank'])
        df['Movement'] = df['last_rank'] - df['Rank']

        return df[['Rank', 'Team', 'Manager', 'GW Points', 'Total Points', 'Movement', 'Team_ID']]

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching classic standings: {e}")
        return pd.DataFrame()


def get_classic_leagues_for_team(team_id: int) -> Dict[int, str]:
    """
    Returns a dictionary of {league_id: league_name} for all classic leagues
    the team is participating in.
    """
    data = get_entry_details(team_id)
    if not data or 'leagues' not in data:
        return {}

    leagues = data['leagues'].get('classic', [])
    return {l['id']: l['name'] for l in leagues}


def get_entry_details(team_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetches details for a specific Classic Team (Entry).
    Includes league participation lists.
    """
    if not team_id:
        return None

    url = f"{FPL_CLASSIC_BASE}/entry/{team_id}/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching team details: {e}")
        return None


def get_entry_history(team_id: int) -> pd.DataFrame:
    """
    Fetches the history (points per GW) for the user's classic team.
    Returns DataFrame of the 'current' season history.
    """
    url = f"{FPL_CLASSIC_BASE}/entry/{team_id}/history/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data.get('current', []))
    except Exception:
        return pd.DataFrame()


# ==============================================================================
# DRAFT MODE ENDPOINTS
# ==============================================================================

def get_draft_picks(league_id: int) -> Dict[int, Dict[str, Any]]:
    """
    Fetches the initial draft picks for each team in the league.
    Returns a dictionary keyed by Team ID.
    """
    draft_url = f"{FPL_DRAFT_BASE}/draft/{league_id}/choices"
    player_data = pull_fpl_player_stats()  # Reuse cached/global pull if possible in future

    try:
        draft_data = requests.get(draft_url, timeout=10).json()

        # Map Player ID -> Name
        player_mapping = {
            p['id']: f"{p['first_name']} {p['second_name']}"
            for p in player_data['elements']
        }

        draft_picks = {}

        for choice in draft_data['choices']:
            team_id = choice['entry']
            team_name = choice['entry_name']
            player_id = choice['element']

            if team_id not in draft_picks:
                draft_picks[team_id] = {'team_name': team_name, 'players': []}

            player_name = player_mapping.get(player_id, f"Unknown ({player_id})")
            draft_picks[team_id]['players'].append(player_name)

        return draft_picks

    except Exception as e:
        st.error(f"Error fetching draft picks: {e}")
        return {}


def get_league_details(league_id: int) -> Dict[str, Any]:
    """
    Fetches the 'details' endpoint: Standings, Matches, and Entries.
    """
    url = f"{FPL_DRAFT_BASE}/league/{league_id}/details"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error fetching league details: {e}")
        return {}


def get_transaction_data(league_id: int) -> List[Dict[str, Any]]:
    """
    Fetches waiver/free agent transactions for the league.
    """
    # Check global config cache first (if utilizing config caching pattern)
    if getattr(config, "TRANSACTION_DATA", None) is not None:
        return config.TRANSACTION_DATA

    url = f"{FPL_DRAFT_BASE}/draft/{league_id}/transactions"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Update config cache if used
        config.TRANSACTION_DATA = data.get('transactions', [])
        return config.TRANSACTION_DATA

    except Exception as e:
        st.error(f"Error fetching transactions: {e}")
        return []