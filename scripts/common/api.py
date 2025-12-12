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
# GENERAL HELPERS
# ==============================================================================

def get_current_gameweek() -> int:
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        for event in data.get("events", []):
            if event.get("finished") is False:
                return int(event["id"])
        return 1
    except Exception as e:
        print(f"Error fetching current gameweek: {e}")
        return 1


# ==============================================================================
# DRAFT ENDPOINTS
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_bootstrap_static() -> Dict[str, Any]:
    url = f"{FPL_DRAFT_BASE}/bootstrap-static"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def pull_fpl_player_stats() -> pd.DataFrame:
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


# ==============================================================================
# PLAYER MAPPING & OWNERSHIP
# ==============================================================================

def get_fpl_player_mapping() -> Dict[int, Dict[str, Any]]:
    """Returns mapping of Player ID -> {Player, Position}."""
    data = get_bootstrap_static()
    elements = data.get("elements", [])
    type_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    mapping = {}
    for p in elements:
        full_name = f"{p.get('first_name')} {p.get('second_name')}"
        mapping[p['id']] = {
            "Player": full_name,
            "Position": type_map.get(p['element_type'], "F")
        }
    return mapping


def get_league_player_ownership(league_id: int) -> Dict[int, Dict[str, Any]]:
    """
    Fetch current ownership grouped by team (GLOBAL ENTRY ID).
    Fixes the mismatch between League Entry ID (from status) and Global ID (used by app).
    """
    element_status = get_element_status(league_id)
    league_details = get_draft_league_details(league_id)

    # 1. Build ID Maps
    # id_to_global: Maps the internal 'id' (used in element_status) to 'entry_id' (Global ID)
    id_to_global = {}
    global_to_name = {}

    for entry in league_details.get("league_entries", []) or []:
        league_entry_id = entry.get("id")
        global_entry_id = entry.get("entry_id")
        entry_name = entry.get("entry_name")

        if league_entry_id and global_entry_id:
            id_to_global[league_entry_id] = global_entry_id
            global_to_name[global_entry_id] = entry_name

    # 2. Initialize Ownership Dict (Keyed by Global ID)
    league_ownership = {}
    for gid, name in global_to_name.items():
        league_ownership[gid] = {
            "team_name": name,
            "players": {"G": [], "D": [], "M": [], "F": []}
        }

    # 3. Populate from Element Status
    player_map = get_fpl_player_mapping()

    if element_status:
        for status in element_status:
            league_owner_id = status.get("owner")  # This is League Entry ID
            player_id = status.get("element")

            if not league_owner_id: continue

            # Convert to Global ID
            global_id = id_to_global.get(league_owner_id)

            if global_id and global_id in league_ownership:
                pinfo = player_map.get(player_id, {})
                pos = pinfo.get("Position")
                pname = pinfo.get("Player", f"Unknown {player_id}")

                if pos in {"G", "D", "M", "F"}:
                    league_ownership[global_id]["players"][pos].append(pname)

    # 4. Fallback (Strategy B) if status empty
    else:
        current_gw = get_current_gameweek()
        for gid in league_ownership.keys():
            try:
                r = requests.get(f"{FPL_DRAFT_BASE}/entry/{gid}/event/{current_gw}", timeout=5)
                if r.status_code == 200:
                    picks = r.json().get("picks", [])
                    for p in picks:
                        pid = p['element']
                        pinfo = player_map.get(pid, {})
                        pos = pinfo.get("Position")
                        pname = pinfo.get("Player")
                        if pos: league_ownership[gid]["players"][pos].append(pname)
            except:
                pass

    return league_ownership


def get_transaction_data(league_id: int) -> List[Dict[str, Any]]:
    if getattr(config, "TRANSACTION_DATA", None) is not None:
        return config.TRANSACTION_DATA
    url = f"{FPL_DRAFT_BASE}/league/{league_id}/transactions"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get('transactions', [])
        config.TRANSACTION_DATA = data
        return data
    except Exception:
        return []


def get_draft_team_composition_for_gameweek(team_id: int, gameweek: int) -> pd.DataFrame:
    # 1. Try requested gameweek
    url = f"https://draft.premierleague.com/api/entry/{team_id}/event/{gameweek}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        picks = data.get('picks', [])
    except:
        picks = []

    # 2. Fallback
    if not picks and gameweek > 1:
        try:
            url_prev = f"https://draft.premierleague.com/api/entry/{team_id}/event/{gameweek - 1}"
            r = requests.get(url_prev, timeout=10)
            data = r.json()
            picks = data.get('picks', [])
        except:
            picks = []

    if not picks:
        return pd.DataFrame()

    player_map = get_fpl_player_mapping()
    row_list = []
    for p in picks:
        pid = p['element']
        info = player_map.get(pid, {'Player': f"Unknown {pid}", 'Position': 'F'})
        row_list.append({
            'element': pid,
            'Player': info['Player'],
            'Position': info['Position'],
            'pick_position': p['position'],
            'is_captain': p['is_captain'],
            'is_vice_captain': p['is_vice_captain']
        })
    return pd.DataFrame(row_list)


def get_historical_team_scores(league_id: int) -> pd.DataFrame:
    try:
        league_id = int(league_id)
    except:
        raise ValueError("league_id must be an integer")

    url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    try:
        data = requests.get(url, timeout=30).json()
    except:
        return pd.DataFrame()

    entries = {}
    for e in data.get("league_entries", []) or []:
        eid = e.get("entry_id", e.get("id"))
        name = e.get("entry_name")
        if eid and name: entries[int(eid)] = str(name)

    rows = []
    for m in data.get("matches", []) or []:
        gw = m.get("event")
        e1, e2 = m.get("league_entry_1"), m.get("league_entry_2")
        p1, p2 = m.get("league_entry_1_points"), m.get("league_entry_2_points")
        if isinstance(gw, int) and p1 is not None and e1:
            rows.append({"event": int(gw), "entry_id": int(e1), "entry_name": entries.get(int(e1), f"Team {e1}"),
                         "points": float(p1)})
        if isinstance(gw, int) and p2 is not None and e2:
            rows.append({"event": int(gw), "entry_id": int(e2), "entry_name": entries.get(int(e2), f"Team {e2}"),
                         "points": float(p2)})

    if not rows: return pd.DataFrame(columns=["event", "entry_id", "entry_name", "points", "total_points"])
    df = pd.DataFrame(rows)
    df["total_points"] = df["points"]
    df = df.sort_values(["event", "entry_id"]).reset_index(drop=True)
    return df


# ... (Classic/Fixture/Rotowire functions unchanged) ...
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


def get_entry_history(team_id: int) -> pd.DataFrame:
    try:
        return pd.DataFrame(
            requests.get(f"{FPL_CLASSIC_BASE}/entry/{team_id}/history/", timeout=10).json().get('current', []))
    except:
        return pd.DataFrame()


def get_element_history(player_id: int) -> pd.DataFrame:
    try:
        return pd.DataFrame(
            requests.get(f"{FPL_CLASSIC_BASE}/element-summary/{player_id}/", timeout=30).json().get("history", []))
    except:
        return pd.DataFrame()


def get_earliest_kickoff_et(gw: int) -> datetime:
    try:
        r = requests.get(config.FPL_FIXTURES_BY_EVENT.format(gw=gw), timeout=10);
        r.raise_for_status()
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
        if not candidates: return None
        exact = [c for c in candidates if c[0] == current_gameweek]
        return max(exact, key=lambda x: x[1])[2] if exact else \
        min(candidates, key=lambda x: (abs(x[0] - current_gameweek), -x[1]))[2]
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
        df['Value'] = df.apply(lambda r: r['Points'] / r['Price'] if r['Price'] > 0 else float('nan'), axis=1)
        if limit: df = df.head(limit)
        return df.reset_index(drop=True)
    except:
        return pd.DataFrame()


def get_rotowire_season_rankings(url: str, limit: Optional[int] = None) -> pd.DataFrame:
    try:
        soup = BeautifulSoup(requests.get(url, timeout=30).content, "html.parser")
        tbl = soup.select_one("table.article-table__tablesorter")
        if not tbl: return pd.DataFrame()
        data = []
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 12: data.append([t.get_text(strip=True) for t in tds])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=["Overall Rank", "FW Rank", "MID Rank", "DEF Rank", "GK Rank", "Player", "Team",
                                         "Position", "Price", "TSB %", "Points", "PP/90"])
        for c in ["FW Rank", "MID Rank", "DEF Rank", "GK Rank", "Points", "PP/90", "Price", "TSB %", "Overall Rank"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df["Pos Rank"] = df[["FW Rank", "MID Rank", "DEF Rank", "GK Rank"]].fillna(0).sum(axis=1).astype(int)
        if limit: df = df.head(limit)
        return df
    except:
        return pd.DataFrame()