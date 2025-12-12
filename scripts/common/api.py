import config
import pandas as pd
import numpy as np
import requests
import re
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

# Constants
TZ_ET = ZoneInfo("America/New_York")
FPL_CLASSIC_BASE = "https://fantasy.premierleague.com/api"
FPL_DRAFT_BASE = "https://draft.premierleague.com/api"


# ==============================================================================
# 1. CORE FETCHERS & HELPERS (ORIGINAL LOGIC)
# ==============================================================================

def get_current_gameweek() -> int:
    """
    Returns the next relevant gameweek (event).
    """
    try:
        # Try 'game' endpoint first
        r = requests.get(f"{FPL_DRAFT_BASE}/game", timeout=10)
        data = r.json()
        if data.get('current_event_finished'):
            return int(data['next_event'])
        return int(data['current_event'])
    except:
        # Fallback to bootstrap
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            r = requests.get(url, timeout=10)
            data = r.json()
            for event in data.get("events", []):
                if event.get("finished") is False:
                    return int(event["id"])
            return 38
        except Exception as e:
            print(f"Error fetching current gameweek: {e}")
            return 1


@st.cache_data(ttl=3600, show_spinner=False)
def get_bootstrap_static() -> Dict[str, Any]:
    url = f"{FPL_DRAFT_BASE}/bootstrap-static"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def pull_fpl_player_stats() -> pd.DataFrame:
    """Fetches all player statistics (elements)."""
    data = get_bootstrap_static()
    elements = data.get("elements", [])
    if not elements:
        return pd.DataFrame()
    return pd.DataFrame(elements)


def get_draft_league_details(league_id: int) -> Dict[str, Any]:
    url = f"{FPL_DRAFT_BASE}/league/{league_id}/details"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error fetching league details: {e}")
        return {}


def get_draft_picks_raw(league_id: int) -> Dict[str, Any]:
    url = f"{FPL_DRAFT_BASE}/draft/{league_id}/choices"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def get_element_status(league_id: int) -> List[Dict[str, Any]]:
    url = f"{FPL_DRAFT_BASE}/league/{league_id}/element-status"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("element_status", [])
    except Exception:
        return []


def get_league_details(league_id: int) -> dict:
    return get_draft_league_details(league_id)


def get_draft_league_teams(league_id: int) -> dict:
    data = get_league_details(league_id)
    entries = data.get("league_entries", [])
    return {e['id']: e['entry_name'] for e in entries}


def get_league_entries(league_id: int) -> Dict[int, str]:
    """Helper for utils: Entry ID -> Entry Name"""
    return get_draft_league_teams(league_id)


def get_league_teams(league_id):
    """Wrapper for config caching of league teams."""
    if config.LEAGUE_DATA is None:
        config.LEAGUE_DATA = get_draft_league_teams(league_id)
    return config.LEAGUE_DATA


# ==============================================================================
# 2. PLAYER MAPPING & TRANSACTION REPLAY LOGIC (CRITICAL RESTORATION)
# ==============================================================================

def get_fpl_player_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Fetches FPL player data and returns dictionary linking player ids to player names.
    RESTORED: Uses Full Name (First + Second) to ensure Rotowire merge works.
    """
    try:
        player_data = get_bootstrap_static()
    except:
        player_data = requests.get(f"{FPL_DRAFT_BASE}/bootstrap-static").json()

    players = player_data['elements']
    teams = player_data.get('teams', [])

    fpl_player_map = {}

    for player in players:
        player_id = player.get('id')
        full_name = f"{player.get('first_name', '')} {player.get('second_name', '')}"
        web_name = player.get('web_name', '').strip()
        if not web_name or web_name == full_name:
            web_name = None

        team_index = player.get('team', 0) - 1
        pos_idx = player.get('element_type', 1) - 1

        team_str = teams[team_index]['short_name'] if 0 <= team_index < len(teams) else 'Unknown'
        pos_str = ['G', 'D', 'M', 'F'][pos_idx] if 0 <= pos_idx < 4 else 'Unknown'

        fpl_player_map[player_id] = {
            'Player': full_name,
            'Web_Name': web_name,
            'Team': team_str,
            'Position': pos_str
        }
    return fpl_player_map


def get_transaction_data(league_id: int) -> List[Dict[str, Any]]:
    """
    Fetches transactions. Checks global config cache first.
    RESTORED URL: /api/draft/league/{id}/transactions
    """
    if getattr(config, "TRANSACTION_DATA", None) is not None:
        return config.TRANSACTION_DATA

    url = f"{FPL_DRAFT_BASE}/draft/league/{league_id}/transactions"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get('transactions', [])
        config.TRANSACTION_DATA = data
        return data
    except Exception:
        return []


def get_waiver_transactions_up_to_gameweek(league_id, gameweek):
    """
    Fetches transactions up to a specific gameweek.
    """
    transactions = get_transaction_data(league_id)
    if gameweek is None:
        return transactions

    # Filter by event
    return [tx for tx in transactions if tx.get('event') is not None and int(tx.get('event')) <= int(gameweek)]


def get_starting_team_composition(league_id):
    """
    Fetches the ORIGINAL draft picks for each team.
    """
    draft_url = f"{FPL_DRAFT_BASE}/draft/{league_id}/choices"
    draft_data = requests.get(draft_url).json()

    player_data = get_bootstrap_static()
    player_mapping = {
        p['id']: f"{p['first_name']} {p['second_name']}"
        for p in player_data['elements']
    }

    draft_picks = {}
    for choice in draft_data.get('choices', []):
        team_id = choice['entry']
        team_name = choice['entry_name']
        player_id = choice['element']

        if team_id not in draft_picks:
            draft_picks[team_id] = {'team_name': team_name, 'players': []}

        pname = player_mapping.get(player_id, f"Unknown ({player_id})")
        draft_picks[team_id]['players'].append(pname)

    return draft_picks


def get_team_composition_for_gameweek(league_id, team_id, gameweek):
    """
    Determines team composition by replaying transactions over the draft picks.
    THIS IS THE CRITICAL FUNCTION FOR ROSTERS.
    """
    # 1. Setup
    try:
        league_id = int(league_id); team_id = int(team_id)
    except:
        return pd.DataFrame()

    if gameweek is None: gameweek = get_current_gameweek()

    player_map = get_fpl_player_mapping()
    name_to_id = {v['Player']: k for k, v in player_map.items()}

    # 2. Get Starting Squad
    starting = get_starting_team_composition(league_id)
    team_start = starting.get(team_id, {})
    team_composition = set(team_start.get('players', []))  # Set of Names

    # 3. Replay Transactions
    transactions = get_waiver_transactions_up_to_gameweek(league_id, gameweek)

    for tx in transactions:
        if tx.get('entry') == team_id and tx.get('result') == 'a':
            pid_in = tx.get('element_in')
            pid_out = tx.get('element_out')

            name_in = player_map.get(pid_in, {}).get('Player', f"Unknown {pid_in}")
            name_out = player_map.get(pid_out, {}).get('Player', f"Unknown {pid_out}")

            team_composition.discard(name_out)
            team_composition.add(name_in)

    # 4. Build DataFrame
    rows = []
    for name in sorted(team_composition):
        pid = name_to_id.get(name)
        if pid and pid in player_map:
            rows.append({
                'Player': player_map[pid]['Player'],
                'Team': player_map[pid]['Team'],
                'Position': player_map[pid]['Position']
            })
        else:
            rows.append({'Player': name, 'Team': 'Unknown', 'Position': 'Unknown'})

    return pd.DataFrame(rows)


# Alias for compatibility with new scripts
get_draft_team_composition_for_gameweek = get_team_composition_for_gameweek


def get_league_player_ownership(league_id: int) -> Dict[int, Dict[str, Any]]:
    """
    Fetch current ownership using element-status (fast) or fallback to roster replay.
    """
    element_status = get_element_status(league_id)
    league_details = get_draft_league_details(league_id)

    # Map League Entry ID -> Global Entry ID
    id_to_global = {e['id']: e['entry_id'] for e in league_details.get("league_entries", []) if 'id' in e}
    id_to_name = {e['entry_id']: e['entry_name'] for e in league_details.get("league_entries", []) if 'entry_id' in e}

    league_ownership = {
        gid: {"team_name": name, "players": {"G": [], "D": [], "M": [], "F": []}}
        for gid, name in id_to_name.items()
    }

    player_map = get_fpl_player_mapping()

    if element_status:
        for status in element_status:
            lid = status.get("owner")
            pid = status.get("element")
            if not lid: continue

            gid = id_to_global.get(lid)
            if gid and gid in league_ownership:
                pinfo = player_map.get(pid, {})
                pos = pinfo.get("Position")
                name = pinfo.get("Player")
                if pos in league_ownership[gid]["players"]:
                    league_ownership[gid]["players"][pos].append(name)
    else:
        # Fallback: Loop through every team and use get_team_composition_for_gameweek
        gw = get_current_gameweek()
        for gid in league_ownership:
            df = get_team_composition_for_gameweek(league_id, gid, gw)
            if not df.empty:
                for _, row in df.iterrows():
                    pos = row['Position']
                    if pos in league_ownership[gid]["players"]:
                        league_ownership[gid]["players"][pos].append(row['Player'])

    return league_ownership


# ==============================================================================
# 3. CLASSIC & FIXTURE ENDPOINTS (NEWER FEATURES)
# ==============================================================================

def get_classic_league_standings(league_id: int, page: int = 1) -> pd.DataFrame:
    url = f"{FPL_CLASSIC_BASE}/leagues-classic/{league_id}/standings/?page_new_entries=1&page_standings={page}"
    try:
        r = requests.get(url, timeout=10);
        r.raise_for_status();
        data = r.json()
        df = pd.DataFrame(data.get('standings', {}).get('results', []))
        if df.empty: return df
        df = df.rename(
            columns={'rank': 'Rank', 'entry_name': 'Team', 'player_name': 'Manager', 'event_total': 'GW Points',
                     'total': 'Total Points', 'entry': 'Team_ID'})
        df['Movement'] = df['last_rank'].replace(0, df['Rank']) - df['Rank']
        return df[['Rank', 'Team', 'Manager', 'GW Points', 'Total Points', 'Movement', 'Team_ID']]
    except:
        return pd.DataFrame()


def get_entry_details(team_id: int) -> Optional[Dict[str, Any]]:
    if not team_id: return None
    try:
        return requests.get(f"{FPL_CLASSIC_BASE}/entry/{team_id}/", timeout=10).json()
    except:
        return None


def get_historical_team_scores(league_id: int) -> pd.DataFrame:
    """Pull per-team weekly scores from details."""
    try:
        url = f"{FPL_DRAFT_BASE}/league/{league_id}/details"
        data = requests.get(url, timeout=30).json()
        entries = {e.get('entry_id'): e.get('entry_name') for e in data.get('league_entries', [])}
        rows = []
        for m in data.get("matches", []):
            gw = m.get("event")
            e1, e2 = m.get("league_entry_1"), m.get("league_entry_2")
            p1, p2 = m.get("league_entry_1_points"), m.get("league_entry_2_points")
            if p1 is not None and e1: rows.append({"event": gw, "entry_id": e1, "points": float(p1)})
            if p2 is not None and e2: rows.append({"event": gw, "entry_id": e2, "points": float(p2)})
        return pd.DataFrame(rows).sort_values("event")
    except:
        return pd.DataFrame()


def get_earliest_kickoff_et(gw: int) -> datetime:
    try:
        r = requests.get(config.FPL_FIXTURES_BY_EVENT.format(gw=gw), timeout=10)
        times = [datetime.fromisoformat(x["kickoff_time"].replace("Z", "+00:00")).astimezone(TZ_ET) for x in r.json() if
                 x.get("kickoff_time")]
        return min(times) if times else datetime.now(TZ_ET)
    except:
        return datetime.now(TZ_ET)


def get_fixtures_for_event(gw: int) -> List[Dict[str, Any]]:
    try:
        return requests.get("https://fantasy.premierleague.com/api/fixtures/", params={"event": int(gw)},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=20).json()
    except:
        return []


@st.cache_data(show_spinner=False)
def get_future_fixtures() -> pd.DataFrame:
    try:
        df = pd.DataFrame(requests.get("https://fantasy.premierleague.com/api/fixtures/?future=1", timeout=30).json())
        return df[["event", "kickoff_time", "team_h", "team_a", "team_h_difficulty",
                   "team_a_difficulty"]] if not df.empty else df
    except:
        return pd.DataFrame()


# ==============================================================================
# 4. ROTOWIRE SCRAPERS
# ==============================================================================

def get_rotowire_rankings_url(current_gameweek=None) -> Optional[str]:
    if current_gameweek is None: current_gameweek = get_current_gameweek()
    try:
        soup = BeautifulSoup(
            requests.get(config.ARTICLES_INDEX, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).content,
            "html.parser")
        anchors = soup.select('a[href*="fantasy-premier-league-player-rankings-gameweek-"]')
        pat = re.compile(
            r"/soccer/article/fantasy-premier-league-player-rankings-gameweek-(\d+)(?:-[a-z0-9-]+)?-(\d+)$")
        candidates = []
        for a in anchors:
            m = pat.search(a.get("href", ""))
            if m: candidates.append(
                (int(m.group(1)), int(m.group(2)), urljoin(config.ARTICLES_INDEX, a.get("href", ""))))
        return max([c for c in candidates if c[0] == current_gameweek], key=lambda x: x[1])[2] if candidates else None
    except:
        return None


def get_rotowire_player_projections(url: str, limit=None) -> pd.DataFrame:
    try:
        soup = BeautifulSoup(requests.get(url, timeout=15).content, 'html.parser')
        tbl = soup.find(class_='article-table__tablesorter article-table__standard article-table__figure')
        if not tbl: return pd.DataFrame()
        tds = tbl.find_all("td")
        if not tds: return pd.DataFrame()
        data = [[tds[i + k].text for k in range(12)] for i in range(0, len(tds), 12) if i + 11 < len(tds)]
        df = pd.DataFrame(data, columns=['Overall Rank', 'FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank', 'Player', 'Team',
                                         'Matchup', 'Position', 'Price', 'TSB %', 'Points'])
        for c in ['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank', 'Points', 'Price']: df[c] = pd.to_numeric(df[c],
                                                                                                          errors='coerce').fillna(
            0)
        df['Pos Rank'] = (df['FW Rank'] + df['MID Rank'] + df['DEF Rank'] + df['GK Rank']).astype(int)
        df.drop(columns=['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank'], inplace=True)
        if limit: df = df.head(limit)
        return df.reset_index(drop=True)
    except:
        return pd.DataFrame()


def get_rotowire_season_rankings(url: str, limit: Optional[int] = None) -> pd.DataFrame:
    try:
        soup = BeautifulSoup(requests.get(url, timeout=30).content, "html.parser")
        tbl = soup.select_one("table.article-table__tablesorter")
        if not tbl: return pd.DataFrame()
        data = [[t.get_text(strip=True) for t in tr.find_all("td")] for tr in tbl.find_all("tr") if
                len(tr.find_all("td")) == 12]
        df = pd.DataFrame(data, columns=["Overall Rank", "FW Rank", "MID Rank", "DEF Rank", "GK Rank", "Player", "Team",
                                         "Position", "Price", "TSB %", "Points", "PP/90"])
        for c in ["FW Rank", "MID Rank", "DEF Rank", "GK Rank", "Points", "PP/90", "Price", "TSB %", "Overall Rank"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df["Pos Rank"] = df[["FW Rank", "MID Rank", "DEF Rank", "GK Rank"]].fillna(0).sum(axis=1).astype(int)
        if limit: df = df.head(limit)
        return df
    except:
        return pd.DataFrame()