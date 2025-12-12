# scripts/common/data.py

import numpy as np
import pandas as pd
from fuzzywuzzy import process, fuzz
from typing import Any, Dict, List, Optional, Tuple

from scripts.common.utils import normalize_apostrophes
from scripts.common.api import (
    get_bootstrap_static, get_draft_league_details, get_element_status,
    get_draft_picks_raw, get_transaction_data, get_current_gameweek,
    get_fixtures_for_event, get_future_fixtures, get_entry_details
)

# ==============================================================================
# FPL CONSTANTS / MAPS
# ==============================================================================

TEAM_FULL_TO_SHORT = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Crystal Palace": "CRY", "Everton": "EVE",
    "Fulham": "FUL", "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man Utd": "MUN", "Newcastle": "NEW", "Nott'm Forest": "NFO",
    "Southampton": "SOU", "Spurs": "TOT", "Tottenham": "TOT", "West Ham": "WHU",
    "Wolves": "WOL"
}

POS_MAP = {"1": "G", "2": "D", "3": "M", "4": "F", "GK": "G", "DEF": "D", "MID": "M", "FWD": "F"}


# ==============================================================================
# LEAGUE DATA AGGREGATION
# ==============================================================================

def get_draft_league_entries(league_id: int) -> Dict[int, str]:
    """Returns {entry_id: team_name}."""
    data = get_draft_league_details(league_id)
    return {e['entry_id']: e['entry_name'] for e in data.get('league_entries', [])}


def get_team_id_by_name(league_id: int, team_name: str) -> int:
    """Converts a team name to its corresponding ID."""
    team_map = get_draft_league_entries(league_id)
    target = normalize_apostrophes(team_name).lower()

    for eid, name in team_map.items():
        if normalize_apostrophes(name).lower() == target:
            return eid
    raise ValueError(f"Team '{team_name}' not found.")


def get_gameweek_fixtures_text(league_id: int, gameweek: int) -> List[str]:
    """Returns list of formatted strings: 'Team A (Manager) vs Team B (Manager)'."""
    data = get_draft_league_details(league_id)
    fixtures = data.get('matches', [])
    entries = data.get('league_entries', [])

    # Map entry ID -> (Team Name, Manager Name)
    info = {
        e['id']: (e['entry_name'], f"{e['player_first_name']} {e['player_last_name']}")
        for e in entries
    }

    out = []
    for f in fixtures:
        if f['event'] == gameweek:
            t1 = info.get(f['league_entry_1'], ("Unknown", "Unknown"))
            t2 = info.get(f['league_entry_2'], ("Unknown", "Unknown"))
            out.append(f"{t1[0]} ({t1[1]}) vs {t2[0]} ({t2[1]})")
    return out


def get_draft_picks(league_id: int) -> Dict[int, Dict[str, Any]]:
    """Returns {team_id: {team_name: str, players: [names]}}."""
    raw = get_draft_picks_raw(league_id)
    static = get_bootstrap_static()

    pmap = {p['id']: f"{p['first_name']} {p['second_name']}" for p in static['elements']}

    picks = {}
    for c in raw.get('choices', []):
        tid = c['entry']
        tname = c['entry_name']
        pid = c['element']

        if tid not in picks:
            picks[tid] = {'team_name': tname, 'players': []}
        picks[tid]['players'].append(pmap.get(pid, f"Unknown ({pid})"))

    return picks


def get_league_player_ownership(league_id: int) -> Dict[int, Any]:
    """Returns {team_id: {team_name: str, players: {G:[], D:[]...}}}."""
    status = get_element_status(league_id)
    owner_map = get_draft_league_entries(league_id)
    static = get_bootstrap_static()

    pmap = {}
    for p in static['elements']:
        pos = POS_MAP.get(str(p['element_type']), '?')
        pmap[p['id']] = {'Name': f"{p['first_name']} {p['second_name']}", 'Pos': pos}

    ownership = {}
    for s in status:
        owner = s.get("owner")
        if not owner: continue

        if owner not in ownership:
            ownership[owner] = {
                "team_name": owner_map.get(owner, str(owner)),
                "players": {"G": [], "D": [], "M": [], "F": []}
            }

        pinfo = pmap.get(s['element'])
        if pinfo and pinfo['Pos'] in ownership[owner]['players']:
            ownership[owner]['players'][pinfo['Pos']].append(pinfo['Name'])

    return ownership


def get_historical_team_scores(league_id: int) -> pd.DataFrame:
    """Returns DataFrame of ['event', 'entry_id', 'entry_name', 'points']."""
    data = get_draft_league_details(league_id)
    entries = {e['id']: e['entry_name'] for e in data.get('league_entries', [])}

    rows = []
    for m in data.get('matches', []):
        gw = m.get('event')
        if not gw: continue

        for i in [1, 2]:
            eid = m.get(f'league_entry_{i}')
            pts = m.get(f'league_entry_{i}_points')
            if eid and pts is not None:
                rows.append({
                    "event": int(gw),
                    "entry_id": int(eid),
                    "entry_name": entries.get(eid, str(eid)),
                    "points": float(pts),
                    "total_points": float(pts)
                })

    return pd.DataFrame(rows).sort_values(['event', 'entry_id']).reset_index(drop=True)


def get_classic_leagues_for_team(team_id: int) -> Dict[int, str]:
    """Returns {league_id: league_name} for a classic team."""
    data = get_entry_details(team_id)
    if not data or 'leagues' not in data: return {}
    return {l['id']: l['name'] for l in data['leagues'].get('classic', [])}


# ==============================================================================
# TEAM COMPOSITION & ROSTERS
# ==============================================================================

def get_draft_team_composition_for_gameweek(league_id: int, team_id: int, gameweek: int = None) -> pd.DataFrame:
    """
    Reconstructs a team's roster for a specific GW by applying transactions
    to the initial draft.
    """
    if not gameweek: gameweek = get_current_gameweek()

    static = get_bootstrap_static()
    teams = {t['id']: t['short_name'] for t in static['teams']}
    pmap = {}
    for p in static['elements']:
        pmap[p['id']] = {
            'Player': f"{p['first_name']} {p['second_name']}",
            'Team': teams.get(p['team']),
            'Position': POS_MAP.get(str(p['element_type']))
        }

    picks = get_draft_picks_raw(league_id)
    roster_ids = set()
    for c in picks.get('choices', []):
        if c['entry'] == int(team_id):
            roster_ids.add(c['element'])

    tx_data = sorted(get_transaction_data(league_id), key=lambda x: x['added'])

    for tx in tx_data:
        if tx['event'] > int(gameweek): continue
        if tx['entry'] == int(team_id) and tx['result'] == 'a':
            roster_ids.discard(tx['element_out'])
            roster_ids.add(tx['element_in'])

    rows = []
    for pid in roster_ids:
        info = pmap.get(pid)
        if info:
            rows.append(info)
        else:
            rows.append({'Player': f"Unknown ({pid})", 'Team': '?', 'Position': '?'})

    return pd.DataFrame(rows)


def get_available_players(projections_df: pd.DataFrame, league_ownership: Dict) -> pd.DataFrame:
    """Anti-join: Returns rows from projections_df that are NOT in league_ownership."""
    # Flatten owned players
    owned_names = []
    for team_blob in league_ownership.values():
        for pos_list in team_blob.get('players', {}).values():
            owned_names.extend(pos_list)

    # Simple exclusion
    return projections_df[~projections_df['Player'].isin(owned_names)].copy()


# ==============================================================================
# LINEUP OPTIMIZATION & VALIDATION
# ==============================================================================

def find_optimal_lineup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy algorithm to find best XI (1G, 3-5D, 3-5M, 1-3F).
    Expects columns: ['Position', 'Points', 'Player'].
    """
    work = df.copy()
    work['Position'] = work['Position'].replace({'GK': 'G', 'DEF': 'D', 'MID': 'M', 'FWD': 'F'})

    # Ensure Points are numeric
    work['Points'] = pd.to_numeric(work['Points'], errors='coerce').fillna(0)

    top_gk = work[work['Position'] == 'G'].nlargest(1, 'Points')
    top_def = work[work['Position'] == 'D'].nlargest(3, 'Points')
    top_mid = work[work['Position'] == 'M'].nlargest(3, 'Points')
    top_fwd = work[work['Position'] == 'F'].nlargest(1, 'Points')

    selected = pd.concat([top_gk, top_def, top_mid, top_fwd])

    remaining = work[~work['Player'].isin(selected['Player'])]
    top_rem = remaining.nlargest(3, 'Points')

    final = pd.concat([selected, top_rem])

    final['PosVal'] = final['Position'].map({'G': 0, 'D': 1, 'M': 2, 'F': 3})
    final = final.sort_values(['PosVal', 'Points'], ascending=[True, False]).drop(columns=['PosVal'])

    return final.reset_index(drop=True)


def check_valid_lineup(df: pd.DataFrame) -> bool:
    """Validates formation constraints."""
    if 'Position' in df.columns:
        col = 'Position'
    elif 'position' in df.columns:
        col = 'position'
    else:
        return False

    counts = df[col].replace({'GK': 'G', 'DEF': 'D', 'MID': 'M', 'FWD': 'F'}).value_counts()

    return (
            len(df) == 11 and
            counts.get('G', 0) == 1 and
            3 <= counts.get('D', 0) <= 5 and
            3 <= counts.get('M', 0) <= 5 and
            1 <= counts.get('F', 0) <= 3
    )


# ==============================================================================
# MERGING & NORMALIZATION
# ==============================================================================

def normalize_rotowire_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes Rotowire columns."""
    df = df.copy()
    rename = {}
    for c in df.columns:
        l = c.lower()
        if 'player' in l:
            rename[c] = 'Player'
        elif 'team' in l:
            rename[c] = 'Team'
        elif 'pos' in l and 'rank' not in l:
            rename[c] = 'Position'
        elif 'points' in l:
            rename[c] = 'Points'
    df.rename(columns=rename, inplace=True)

    # Map Teams and Positions
    df['Team'] = df['Team'].map(lambda x: TEAM_FULL_TO_SHORT.get(x, str(x).upper()[:3]))
    df['Position'] = df['Position'].map(lambda x: POS_MAP.get(str(x).upper(), str(x)[0]))
    return df


def merge_fpl_players_and_projections(fpl_df: pd.DataFrame, proj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuzzy matches FPL roster with Rotowire projections.
    Prioritizes matches where Team + Position are identical.
    """
    proj_norm = normalize_rotowire_data(proj_df)
    candidates = proj_norm['Player'].dropna().unique().tolist()

    out = []

    for _, row in fpl_df.iterrows():
        name = row['Player']
        team = row['Team']
        pos = row['Position']

        # 1. Fuzzy Match Name
        best = process.extractOne(name, candidates)
        matched_row = None

        if best:
            match_name, score = best

            cand_row = proj_norm[proj_norm['Player'] == match_name].iloc[0]

            # 2. Logic: Lower threshold if Team+Pos match
            if cand_row['Team'] == team and cand_row['Position'] == pos:
                if score >= 60: matched_row = cand_row
            else:
                if score >= 80: matched_row = cand_row

        # 3. Assemble Result
        res = row.to_dict()  # Keep original FPL data
        if matched_row is not None:
            res['Points'] = matched_row['Points']
            res['Price'] = matched_row.get('Price', 0)
            res['TSB %'] = matched_row.get('TSB %', 0)
        else:
            res['Points'] = 0.0

        out.append(res)

    return pd.DataFrame(out)


# ==============================================================================
# STATS ENRICHMENT & SCORING
# ==============================================================================

def _min_max_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)


def calculate_waiver_scores(df: pd.DataFrame, w_proj=0.5, w_form=0.3, w_fdr=0.2) -> pd.DataFrame:
    """Calculates 'Waiver Score' based on weighted factors."""
    tmp = df.copy()
    tmp['Proj_norm'] = _min_max_norm(tmp['Points']).fillna(0.5)
    tmp['Form_norm'] = _min_max_norm(tmp['Form']).fillna(0.5)

    # Invert FDR (Lower is better, so 6 - FDR gives higher score to easy games)
    tmp['FDREase'] = 6 - pd.to_numeric(tmp['AvgFDRNextN'], errors='coerce')
    tmp['FDREase_norm'] = _min_max_norm(tmp['FDREase']).fillna(0.5)

    tmp['Waiver Score'] = (
            w_proj * tmp['Proj_norm'] +
            w_form * tmp['Form_norm'] +
            w_fdr * tmp['FDREase_norm']
    )
    return tmp


def add_enriched_stats(df: pd.DataFrame, fpl_stats: pd.DataFrame, current_gw: int, weeks=3) -> pd.DataFrame:
    """
    Adds Form and AvgFDRNextN to the dataframe.
    Calculates REAL Fixture Difficulty using get_future_fixtures.
    """
    # 1. Merge generic stats (Form, PPG)
    base = df.merge(fpl_stats[['Player', 'Team', 'Position', 'form', 'points_per_game', 'id']],
                    on=['Player', 'Team', 'Position'], how='left')

    if 'id' in base.columns: base.rename(columns={'id': 'Player_ID'}, inplace=True)

    # 2. Add Form (Fallback chain: form -> ppg -> 0)
    base['Form'] = pd.to_numeric(base['form'], errors='coerce').fillna(
        pd.to_numeric(base['points_per_game'], errors='coerce')
    ).fillna(0.0)

    # 3. Calculate FDR (Fixture Difficulty)
    # We need to map the 'Team' column (short code like 'ARS') to 'Team ID' to query the fixtures DF
    # We can get this map from fpl_stats
    team_map_df = fpl_stats[['team', 'team_name_abbrv']].drop_duplicates()
    short_to_id = dict(zip(team_map_df['team_name_abbrv'], team_map_df['team']))

    fixtures = get_future_fixtures()

    # Pre-filter fixtures to the relevant window
    fixtures = fixtures[
        (fixtures['event'] >= current_gw) &
        (fixtures['event'] < current_gw + weeks)
        ]

    def get_avg_fdr(team_short_code):
        tid = short_to_id.get(team_short_code)
        if not tid: return 3.0  # Default neutral

        # Find home and away games for this team
        home_games = fixtures[fixtures['team_h'] == tid]['team_h_difficulty']
        away_games = fixtures[fixtures['team_a'] == tid]['team_a_difficulty']

        all_diffs = pd.concat([home_games, away_games])
        if all_diffs.empty: return 3.0

        return float(all_diffs.mean())

    base['AvgFDRNextN'] = base['Team'].apply(get_avg_fdr)

    return base


# ==============================================================================
# LUCK ADJUSTED STANDINGS
# ==============================================================================

def calculate_luck_adjusted_standings(league_id: int) -> pd.DataFrame:
    """Calculates Fair Rank based on 'All-Play' logic."""
    data = get_draft_league_details(league_id)
    fixtures = pd.DataFrame(data.get('matches', []))
    entries = {e['id']: e['entry_name'] for e in data.get('league_entries', [])}

    # Filter played matches
    played = fixtures[(fixtures['league_entry_1_points'] > 0) | (fixtures['league_entry_2_points'] > 0)].copy()

    if played.empty:
        return pd.DataFrame()

    home = played[['event', 'league_entry_1', 'league_entry_1_points']].rename(
        columns={'league_entry_1': 'id', 'league_entry_1_points': 'score'})
    away = played[['event', 'league_entry_2', 'league_entry_2_points']].rename(
        columns={'league_entry_2': 'id', 'league_entry_2_points': 'score'})

    long_df = pd.concat([home, away])
    long_df['Team'] = long_df['id'].map(entries)

    long_df['GW_Rank'] = long_df.groupby('event')['score'].rank(ascending=False, method='min')

    stats = long_df.groupby('Team').agg(
        Avg_GW_Rank=('GW_Rank', 'mean'),
        Total_Points=('score', 'sum'),
        Avg_Score=('score', 'mean')
    ).reset_index()

    stats['Fair_Rank'] = stats['Avg_GW_Rank'].rank(ascending=True, method='min').astype(int)
    return stats.sort_values('Fair_Rank').set_index('Fair_Rank')


# ==============================================================================
# FIXTURE DIFFICULTY GRID
# ==============================================================================

def get_fixture_difficulty_grid(weeks: int = 6):
    """
    Generates the FDR grid.
    Returns: (display_df, diff_numeric_df, avg_series)
    """
    current_gw = get_current_gameweek()

    static = get_bootstrap_static()
    id2short = {t['id']: t['short_name'] for t in static['teams']}
    team_list = sorted(id2short.values())

    cols = [f"GW{gw}" for gw in range(current_gw, current_gw + weeks)]
    disp = pd.DataFrame("—", index=team_list, columns=cols)
    diffs = pd.DataFrame(np.nan, index=team_list, columns=cols)

    for i, gw in enumerate(range(current_gw, current_gw + weeks)):
        fixtures = get_fixtures_for_event(gw)
        for fx in fixtures:
            h_id, a_id = fx['team_h'], fx['team_a']
            h_diff, a_diff = fx['team_h_difficulty'], fx['team_a_difficulty']

            h_name = id2short.get(h_id)
            a_name = id2short.get(a_id)

            if not h_name or not a_name: continue

            col_name = cols[i]

            # Update Home
            prev = disp.at[h_name, col_name]
            disp.at[h_name, col_name] = f"{a_name} (H)" if prev == "—" else f"{prev}\n{a_name} (H)"
            diffs.at[h_name, col_name] = np.nanmean([diffs.at[h_name, col_name], float(h_diff)])

            # Update Away
            prev = disp.at[a_name, col_name]
            disp.at[a_name, col_name] = f"{h_name} (A)" if prev == "—" else f"{prev}\n{h_name} (A)"
            diffs.at[a_name, col_name] = np.nanmean([diffs.at[a_name, col_name], float(a_diff)])

    avg = diffs.fillna(3).mean(axis=1)

    order = avg.sort_values().index
    disp = disp.loc[order]
    diffs = diffs.loc[order]
    disp.insert(0, "Team", disp.index)
    disp["Avg FDR"] = avg.loc[order].round(2)

    return disp, diffs, avg