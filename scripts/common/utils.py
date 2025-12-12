import pandas as pd
import numpy as np
import re
import unicodedata
from typing import Any, Optional


# ==============================================================================
# STRING CLEANING & NORMALIZATION
# ==============================================================================

def clean_text(s: Any) -> str:
    """Standardizes input to a clean string."""
    if s is None: return ""
    if isinstance(s, float) and s != s: return ""
    if not isinstance(s, str): s = str(s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_apostrophes(text: str) -> str:
    """Ensures smart quotes are turned into straight quotes."""
    if not isinstance(text, str): return ""
    return text.replace("’", "'").replace("‘", "'")


def normalize_text(x: Any) -> str:
    """Aggressive normalization: strip accents, lowercase, collapse spaces."""
    s = str(x).strip() if x is not None else ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def remove_duplicate_words(name: str) -> str:
    """Removes duplicate consecutive words (e.g., 'Salah Salah' -> 'Salah')."""
    if not isinstance(name, str): return ""
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', name)


def clean_fpl_player_names(name: str) -> str:
    """
    Cleans player names by removing accents and handling common mismatches.
    Specific handle for known edge cases like 'Bruno Fernandes'.
    """
    if not isinstance(name, str):
        return ""

    # Normalize unicode characters
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

    # Specific known edge cases
    if "Bruno Miguel Borges Fernandes" in name: return "Bruno Fernandes"
    if "Son Heung-min" in name: return "Heung-Min Son"

    return name.strip()


def clean_fpl_team_names(team_name: str) -> str:
    """Standardizes team names to match RotoWire/FPL conventions."""
    if not isinstance(team_name, str): return ""
    team_map = {
        "Man City": "Manchester City",
        "Man Utd": "Manchester United",
        "Spurs": "Tottenham Hotspur",
        "Nott'm Forest": "Nottingham Forest",
        "Wolves": "Wolverhampton Wanderers",
        "Luton": "Luton Town",
        "Sheff Utd": "Sheffield United"
    }
    return team_map.get(team_name, team_name)


def format_team_name(name: Optional[str]) -> Optional[str]:
    """Formats a team name by normalizing apostrophes and capitalizing."""
    if name is None: return None
    normalized_name = unicodedata.normalize('NFKC', name).replace('’', "'").strip()
    return ' '.join(word.capitalize() for word in normalized_name.split())


# ==============================================================================
# HELPERS FOR NORMALIZATION
# ==============================================================================

_POS_NUM_TO_LETTER = {1: "G", 2: "D", 3: "M", 4: "F"}
_POS_WORD_TO_LETTER = {
    "GK": "G", "GKP": "G", "Goalkeeper": "G",
    "DEF": "D", "Defender": "D",
    "MID": "M", "Midfielder": "M",
    "FWD": "F", "FW": "F", "Forward": "F"
}


def _to_position_letter(val) -> Optional[str]:
    """Map whatever we get to G/D/M/F."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        return _POS_NUM_TO_LETTER.get(int(val))
    v = str(val).strip()
    if v in {"G", "D", "M", "F"}:
        return v
    return _POS_WORD_TO_LETTER.get(v)


def _series_to_short_team(s: pd.Series, teams_df: pd.DataFrame) -> pd.Series:
    """Convert a Series of team identifiers to short codes (e.g., 'MCI')."""
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s):
        id2short = dict(zip(teams_df["id"], teams_df["short_name"]))
        return s.map(id2short)
    s = s.astype(str)
    mask_short = s.str.len() == 3
    out = s.where(mask_short)
    # Check if config has mapping (optional)
    try:
        import config
        team_map = getattr(config, "TEAM_FULL_TO_SHORT", None)
        if isinstance(team_map, dict):
            out = out.fillna(s.map(team_map))
    except ImportError:
        pass
    return out.fillna(s)


def _bootstrap_teams_df() -> pd.DataFrame:
    """
    Fetches teams from bootstrap-static and returns a DataFrame [id, name, short_name].
    Uses api.get_bootstrap_static to utilize caching.
    """
    try:
        from scripts.common.api import get_bootstrap_static
        data = get_bootstrap_static()
        teams = data.get("teams", [])
        if not teams:
            return pd.DataFrame(columns=["id", "name", "short_name"])
        return pd.DataFrame(teams)[["id", "name", "short_name"]]
    except Exception as e:
        print(f"Error in _bootstrap_teams_df: {e}")
        return pd.DataFrame(columns=["id", "name", "short_name"])


# ==============================================================================
# SCHEMA NORMALIZERS (FPL & ROTOWIRE)
# ==============================================================================

def normalize_rotowire_players(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes Rotowire projection columns."""
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])

    # Map common column variations
    rename_map = {
        "player_name": "Player", "name": "Player",
        "team_short": "Team", "team": "Team",
        "pos": "Position", "position": "Position",
        "points": "Points", "proj_points": "Points"
    }

    out = df.rename(columns=rename_map)
    # Ensure columns exist
    for col in ["Player", "Team", "Position", "Points"]:
        if col not in out.columns:
            out[col] = np.nan

    return out[["Player", "Team", "Position", "Points"]]


def normalize_fpl_players_to_rotowire_schema(fpl_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes FPL Stats columns to match Rotowire schema."""
    if fpl_df.empty:
        return pd.DataFrame(
            columns=["Player", "Team", "Position", "Player_ID", "Season_Points", "form", "points_per_game"])

    out = fpl_df.copy()

    # map team ID to short name
    if "team" in out.columns and not teams_df.empty:
        id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))
        out["Team"] = out["team"].map(id_to_short)

    # map element_type to Position (1=G, 2=D, 3=M, 4=F)
    type_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    if "element_type" in out.columns:
        out["Position"] = out["element_type"].map(type_map)

    # Rename common columns
    rename_map = {
        "web_name": "Player",
        "id": "Player_ID",
        "total_points": "Season_Points"
    }
    out = out.rename(columns=rename_map)

    cols_needed = ["Player", "Team", "Position", "Player_ID", "Season_Points", "form", "points_per_game"]
    for c in cols_needed:
        if c not in out.columns:
            out[c] = np.nan

    return out[cols_needed]


def _enforce_rw_schema_fpl(fpl_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure FPL stats DF has: Player, Team (short), Position, Player_ID, Season_Points."""
    df = fpl_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Player_ID", "Season_Points"])

    idx = {c.lower(): c for c in df.columns}

    # --- Player ---
    if "player" in idx:
        df["Player"] = df[idx["player"]]
    elif "web_name" in idx:
        df["Player"] = df[idx["web_name"]]
    elif "name" in idx:
        df["Player"] = df[idx["name"]]
    else:
        df["Player"] = df.index.astype(str)

    # --- Team ---
    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    else:
        df["Team"] = np.nan

    # --- Position ---
    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "element_type" in idx:
        df["Position"] = df[idx["element_type"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    # --- Player_ID ---
    if "player_id" in idx:
        df["Player_ID"] = pd.to_numeric(df[idx["player_id"]], errors="coerce")
    elif "id" in idx:
        df["Player_ID"] = pd.to_numeric(df[idx["id"]], errors="coerce")
    else:
        df["Player_ID"] = np.nan

    # --- Season_Points ---
    if "season_points" in idx:
        df["Season_Points"] = pd.to_numeric(df[idx["season_points"]], errors="coerce")
    elif "total_points" in idx:
        df["Season_Points"] = pd.to_numeric(df[idx["total_points"]], errors="coerce")
    else:
        df["Season_Points"] = 0

    df["Team"] = df["Team"].astype(str)
    df["Position"] = df["Position"].astype(str)

    keep = ["Player", "Team", "Position", "Player_ID", "Season_Points"]
    extras = [c for c in df.columns if c not in keep]
    return df[keep + extras]


def _enforce_rw_schema_proj(proj_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure projections DF has Player, Team, Position, Points."""
    df = proj_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])

    idx = {c.lower(): c for c in df.columns}

    df["Player"] = df[idx["player"]] if "player" in idx else df.index.astype(str)

    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    else:
        df["Team"] = np.nan

    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "pos" in idx:
        df["Position"] = df[idx["pos"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    df["Points"] = pd.to_numeric(df[idx["points"]], errors="coerce") if "points" in idx else np.nan

    return df


# ==============================================================================
# DATA PROCESSING & MERGING
# ==============================================================================

def merge_fpl_players_and_projections(team_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """Merges team composition DataFrame with Projections DataFrame on 'Player'."""
    if team_df.empty: return pd.DataFrame()
    if projections_df.empty: return team_df
    return pd.merge(team_df, projections_df, on="Player", how="left")


# ==============================================================================
# LEAGUE & STATS LOGIC
# ==============================================================================

def get_team_id_by_name(league_id: int, team_name: str) -> int:
    """Given a team name, returns the corresponding team_id."""
    from scripts.common.api import get_draft_league_teams
    teams = get_draft_league_teams(league_id)
    name_to_id = {name: id for id, name in teams.items()}
    return name_to_id.get(team_name)


def compute_team_record(team_id: int, details: dict, up_to_gw: int = None):
    """Return (W, D, L, Pts) using league details."""
    standings = details.get("standings", []) or []
    row = next((s for s in standings if s.get("league_entry") == team_id), None)

    if row is not None and up_to_gw is None:
        w = int(row.get("matches_won", 0) or 0)
        d = int(row.get("matches_drawn", 0) or 0)
        l = int(row.get("matches_lost", 0) or 0)
        pts = w * 3 + d
        return w, d, l, pts

    # Fallback: compute from matches
    w = d = l = 0
    matches = details.get("matches", []) or []
    for m in matches:
        if up_to_gw is not None and int(m.get("event", 0) or 0) > int(up_to_gw): continue
        if team_id not in (m.get("league_entry_1"), m.get("league_entry_2")): continue
        s1 = int(m.get("league_entry_1_points", 0) or 0)
        s2 = int(m.get("league_entry_2_points", 0) or 0)
        t1 = m.get("league_entry_1")
        if not m.get("finished"): continue
        if s1 == s2:
            d += 1
        else:
            won = (team_id == t1 and s1 > s2) or (team_id != t1 and s2 > s1)
            if won:
                w += 1
            else:
                l += 1

    pts = w * 3 + d
    return w, d, l, pts


def compute_h2h_breakdown(team_id: int, matches: list, id_to_name: dict,
                          up_to_gw: Optional[int] = None) -> pd.DataFrame:
    """Per-opponent W/D/L and percentages vs each opponent."""
    rows = []
    opp_ids = set()
    for m in matches:
        if team_id in (m.get("league_entry_1"), m.get("league_entry_2")):
            opp_ids.add(m["league_entry_2"] if m["league_entry_1"] == team_id else m["league_entry_1"])

    for opp in opp_ids:
        w = d = l = 0
        total = 0
        for m in matches:
            if up_to_gw is not None and int(m.get("event", 0)) > int(up_to_gw): continue
            if not m.get("finished"): continue
            t1, t2 = m.get("league_entry_1"), m.get("league_entry_2")
            if {t1, t2} != {team_id, opp}: continue
            s1 = m.get("league_entry_1_points", 0) or 0
            s2 = m.get("league_entry_2_points", 0) or 0
            total += 1
            if s1 == s2:
                d += 1
            else:
                our_win = (team_id == t1 and s1 > s2) or (team_id == t2 and s2 > s1)
                if our_win:
                    w += 1
                else:
                    l += 1
        if total == 0: continue
        rows.append({
            "Opponent": id_to_name.get(opp, f"Team {opp}"),
            "W": w, "D": d, "L": l, "GP": total,
            "Win%": w / total, "Draw%": d / total, "Loss%": l / total
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Win%", ascending=False).reset_index(drop=True)
    return df


def select_optimal_lineup(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a roster DataFrame with columns ['Player', 'Position', 'Points'],
    Selects the best Starting XI (11) and Bench (4) respecting FPL formation rules:
    - 1 GK
    - Min 3 DEF
    - Min 1 FWD
    - (Min 2 MID is standard but usually covered by filling 10 outfielders)

    Returns DataFrame with 'Is_Starter' boolean column.
    """
    df = roster_df.copy()
    if 'Points' not in df.columns:
        df['Points'] = 0.0

    # Fill NaN points with -1 to prioritize playing players
    df['Points'] = df['Points'].fillna(-1)

    # Sort by Points descending to pick best players easily
    df = df.sort_values('Points', ascending=False)

    starters = []

    # 1. Mandatory Spots
    # Best GK
    gks = df[df['Position'] == 'G']
    if not gks.empty:
        starters.append(gks.iloc[0]['Player'])

    # Best 3 DEFs
    defs = df[df['Position'] == 'D']
    starters.extend(defs.head(3)['Player'].tolist())

    # Best 1 FWD
    fwds = df[df['Position'] == 'F']
    starters.extend(fwds.head(1)['Player'].tolist())

    # Best 2 MIDs (Standard FPL rule: min 2 mids)
    mids = df[df['Position'] == 'M']
    starters.extend(mids.head(2)['Player'].tolist())

    # 2. Fill remaining spots (need 11 total, we have 1+3+1+2 = 7 so far)
    # We need 4 more outfielders (D, M, F)
    # Get pool of players not yet chosen
    current_ids = set(starters)

    # Filter for outfielders only (No 2nd GK in XI) who aren't picked
    remaining = df[
        (df['Position'].isin(['D', 'M', 'F'])) &
        (~df['Player'].isin(current_ids))
        ]

    # Pick top 4
    flex_picks = remaining.head(4)['Player'].tolist()
    starters.extend(flex_picks)

    # 3. Mark in DataFrame
    df['Is_Starter'] = df['Player'].isin(starters)

    # 4. Sort for display (Starters first, ordered by Pos G-D-M-F, then Points)
    pos_map = {'G': 1, 'D': 2, 'M': 3, 'F': 4}
    df['Pos_Ord'] = df['Position'].map(pos_map)
    df = df.sort_values(by=['Is_Starter', 'Pos_Ord', 'Points'], ascending=[False, True, False])

    return df