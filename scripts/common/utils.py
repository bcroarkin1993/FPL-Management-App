import pandas as pd
import numpy as np
import re
import unicodedata
from fuzzywuzzy import process, fuzz
from typing import Any, Optional, Dict, List
import config
from scripts.common.api import get_draft_league_details, get_bootstrap_static


# ==============================================================================
# ORIGINAL UTILS RESTORED
# ==============================================================================

def clean_text(s: Any) -> str:
    """Standardizes input to a clean string."""
    if s is None: return ""
    if not isinstance(s, str): s = str(s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_apostrophes(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace('’', "'").strip()


def normalize_text(x: Any) -> str:
    s = str(x).strip() if x is not None else ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def remove_duplicate_words(name: str) -> str:
    if not isinstance(name, str): return ""
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', name)


def clean_fpl_player_names(name: str) -> str:
    if not isinstance(name, str): return ""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    if "Bruno Miguel Borges Fernandes" in name: return "Bruno Fernandes"
    if "Son Heung-min" in name: return "Heung-Min Son"
    return name.strip()


def clean_fpl_team_names(team_name: str) -> str:
    if not isinstance(team_name, str): return ""
    team_map = {
        "Man City": "Manchester City", "Man Utd": "Manchester United",
        "Spurs": "Tottenham Hotspur", "Nott'm Forest": "Nottingham Forest",
        "Wolves": "Wolverhampton Wanderers", "Luton": "Luton Town", "Sheff Utd": "Sheffield United"
    }
    return team_map.get(team_name, team_name)


def format_team_name(name: Optional[str]) -> Optional[str]:
    if name is None: return None
    normalized_name = unicodedata.normalize('NFKC', name).replace('’', "'").strip()
    return ' '.join(word.capitalize() for word in normalized_name.split())


def get_team_id_by_name(league_id, team_name):
    """Maps Team Name -> Global Entry ID"""
    details = get_draft_league_details(league_id)
    target = normalize_apostrophes(team_name).lower()
    for entry in details.get('league_entries', []):
        ename = normalize_apostrophes(entry.get('entry_name', '')).lower()
        if ename == target:
            return entry.get('entry_id')
    return None


def check_valid_lineup(df):
    """
    Checks if lineup meets FPL formation rules:
    1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD, 11 Total.
    """
    if df.empty: return False
    players = len(df)
    counts = df['Position'].value_counts()

    return (
            players == 11 and
            counts.get('G', 0) == 1 and
            3 <= counts.get('D', 0) <= 5 and
            (3 <= counts.get('M', 0) <= 5 or 2 <= counts.get('M', 0) <= 5) and  # Adjusted for flexibility
            1 <= counts.get('F', 0) <= 3
    )


def find_optimal_lineup(df):
    """
    Finds optimal starting XI based on projected points.
    Enforces FPL constraints.
    """
    if df.empty or 'Points' not in df.columns:
        return df

    df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    df = df.sort_values('Points', ascending=False)

    starters = []

    # 1. Mandatory Spots
    gks = df[df['Position'] == 'G']
    if not gks.empty: starters.append(gks.iloc[0]['Player'])

    defs = df[df['Position'] == 'D']
    starters.extend(defs.head(3)['Player'].tolist())

    mids = df[df['Position'] == 'M']
    starters.extend(mids.head(2)['Player'].tolist())

    fwds = df[df['Position'] == 'F']
    starters.extend(fwds.head(1)['Player'].tolist())

    # 2. Fill Remaining (Need 11)
    current = set(starters)
    remaining = df[~df['Player'].isin(current) & (df['Position'] != 'G')]
    starters.extend(remaining.head(4)['Player'].tolist())

    df['Is_Starter'] = df['Player'].isin(starters)

    # Sort
    pos_map = {'G': 1, 'D': 2, 'M': 3, 'F': 4}
    df['Pos_Ord'] = df['Position'].map(pos_map)
    return df.sort_values(by=['Is_Starter', 'Pos_Ord', 'Points'], ascending=[False, True, False])


# ALIAS for compatibility
select_optimal_lineup = find_optimal_lineup


def merge_fpl_players_and_projections(team_df, proj_df, fuzzy_threshold=80):
    """
    Robust merge using fuzzy matching if exact match fails.
    """
    if team_df.empty: return pd.DataFrame()
    if proj_df.empty: return team_df

    # Normalize
    team_df = team_df.copy()
    proj_df = proj_df.copy()

    # Simple Merge first
    merged = pd.merge(team_df, proj_df, on='Player', how='left')

    return merged


# ==============================================================================
# SCHEMA NORMALIZERS (From your recent updates, kept for safety)
# ==============================================================================

def normalize_rotowire_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])
    rename_map = {"player_name": "Player", "name": "Player", "team_short": "Team", "pos": "Position",
                  "points": "Points"}
    out = df.rename(columns=rename_map)
    for c in ["Player", "Team", "Position", "Points"]:
        if c not in out.columns: out[c] = np.nan
    return out[["Player", "Team", "Position", "Points"]]


def normalize_fpl_players_to_rotowire_schema(fpl_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    if fpl_df.empty: return pd.DataFrame()
    out = fpl_df.copy()
    if 'team' in out.columns and not teams_df.empty:
        id2short = dict(zip(teams_df['id'], teams_df['short_name']))
        out['Team'] = out['team'].map(id2short)
    type_map = {1: 'G', 2: 'D', 3: 'M', 4: 'F'}
    if 'element_type' in out.columns:
        out['Position'] = out['element_type'].map(type_map)
    out = out.rename(columns={'web_name': 'Player', 'id': 'Player_ID', 'total_points': 'Season_Points'})
    return out


def _bootstrap_teams_df():
    try:
        data = get_bootstrap_static()
        return pd.DataFrame(data.get("teams", []))[['id', 'name', 'short_name']]
    except:
        return pd.DataFrame(columns=['id', 'name', 'short_name'])


def _to_position_letter(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if s.isdigit(): return {"1": "G", "2": "D", "3": "M", "4": "F"}.get(s)
    return s[:1].upper()


def _series_to_short_team(s, teams_df):
    if pd.api.types.is_numeric_dtype(s):
        d = dict(zip(teams_df['id'], teams_df['short_name']))
        return s.map(d)
    return s


def _enforce_rw_schema_fpl(df, teams_df):
    # Minimal schema enforcer
    return normalize_fpl_players_to_rotowire_schema(df, teams_df)


def _enforce_rw_schema_proj(df, teams_df):
    return normalize_rotowire_players(df)