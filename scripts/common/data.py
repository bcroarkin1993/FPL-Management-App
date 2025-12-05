# scripts/common/data.py

import config
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process
from typing import Any, Dict, List, Optional

from scripts.common.utils import (
    clean_text,
    normalize_text,
    remove_duplicate_words
)


# ==============================================================================
# FPL DATA PROCESSING (ETL)
# ==============================================================================

def process_fpl_static_data(json_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the raw JSON from 'bootstrap-static' into the clean, rich DataFrame
    used throughout the app.

    Args:
        json_data: The JSON response from api.pull_fpl_player_stats()

    Returns:
        pd.DataFrame: A processed dataframe of all players.
    """
    # 1. Parse Element Types (Positions)
    pos_df = pd.DataFrame.from_records(json_data['element_types'])
    pos_df = pos_df[['id', 'singular_name', 'singular_name_short']]
    pos_df.columns = ['position_id', 'position_name', 'position_abbrv']

    # 2. Parse Teams
    team_df = pd.DataFrame.from_records(json_data['teams'])
    team_df = team_df[['id', 'name', 'short_name']]
    team_df.columns = ['team_id', 'team_name', 'team_name_abbrv']

    # 3. Parse Players (Elements)
    player_df = pd.DataFrame.from_records(json_data['elements'])

    # 4. Create Full Name & Clean
    player_df['player'] = player_df['first_name'] + ' ' + player_df['second_name']
    player_df['player'] = player_df['player'].apply(remove_duplicate_words)

    # 5. Merge Metadata
    player_df = pd.merge(player_df, team_df, left_on='team', right_on='team_id')
    player_df = pd.merge(player_df, pos_df, left_on='element_type', right_on='position_id')

    # 6. Select & Rename Columns
    cols = [
        'id', 'player', 'position_abbrv', 'team_name_abbrv', 'clean_sheets', 'saves',
        'goals_scored', 'assists', 'minutes', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'yellow_cards', 'starts', 'expected_goals',
        'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
        'creativity', 'influence', 'bonus', 'bps', 'form', 'points_per_game',
        'total_points', 'clearances_blocks_interceptions', 'recoveries', 'tackles',
        'defensive_contribution', 'chance_of_playing_this_round',
        'chance_of_playing_next_round', 'status', 'now_cost', 'added'
    ]
    # Filter strictly for columns that exist to prevent crashes if API changes
    existing_cols = [c for c in cols if c in player_df.columns]
    player_df = player_df[existing_cols]

    # Standardize key column names
    rename_map = {
        'id': 'Player_ID',
        'player': 'Player',
        'position_abbrv': 'Position',
        'team_name_abbrv': 'Team'
    }
    player_df = player_df.rename(columns=rename_map)

    # 7. Numeric Conversions & Calculated Stats
    player_df["expected_goal_involvements"] = pd.to_numeric(
        player_df["expected_goal_involvements"], errors="coerce"
    )

    player_df["actual_goal_involvements"] = (
            pd.to_numeric(player_df["goals_scored"], errors="coerce") +
            pd.to_numeric(player_df["assists"], errors="coerce")
    )

    # Sort by total points by default
    player_df = player_df.sort_values(by='total_points', ascending=False)

    return player_df


# ==============================================================================
# LINEUP OPTIMIZATION & VALIDATION
# ==============================================================================

def check_valid_lineup(df: pd.DataFrame) -> bool:
    """
    Validates if a lineup meets FPL formation rules.

    Requirements:
    - 11 Players total
    - 1 GK
    - 3-5 DEF
    - 3-5 MID
    - 1-3 FWD
    """
    # Normalize position column casing just in case
    if 'Position' in df.columns:
        pos_col = 'Position'
    elif 'position' in df.columns:
        pos_col = 'position'
    else:
        return False

    players = len(df)
    counts = df[pos_col].value_counts()

    # Safely get counts, defaulting to 0
    g = counts.get('G', counts.get('GK', 0))
    d = counts.get('D', counts.get('DEF', 0))
    m = counts.get('M', counts.get('MID', 0))
    f = counts.get('F', counts.get('FWD', 0))

    player_check = players == 11
    gk_check = g == 1
    def_check = 3 <= d <= 5
    mid_check = 3 <= m <= 5
    fwd_check = 1 <= f <= 3

    return bool(player_check & gk_check & def_check & mid_check & fwd_check)


def find_optimal_lineup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy algorithm to find the best XI from a squad based on 'Projected_Points'.
    Ensures valid formation logic (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD).
    """
    # Ensure we have a standard 'Position' column (G, D, M, F)
    work_df = df.copy()
    work_df['Position'] = work_df['Position'].replace({'GK': 'G', 'DEF': 'D', 'MID': 'M', 'FWD': 'F'})

    # 1. Core Requirements (The "spine")
    top_gk = work_df[work_df['Position'] == 'G'].sort_values(by='Projected_Points', ascending=False).head(1)
    top_def = work_df[work_df['Position'] == 'D'].sort_values(by='Projected_Points', ascending=False).head(3)
    top_mid = work_df[work_df['Position'] == 'M'].sort_values(by='Projected_Points', ascending=False).head(3)
    top_fwd = work_df[work_df['Position'] == 'F'].sort_values(by='Projected_Points', ascending=False).head(1)

    selected_players = pd.concat([top_gk, top_def, top_mid, top_fwd])

    # 2. Fill remaining 3 spots with best available (regardless of pos)
    remaining_players = work_df[~work_df['Player'].isin(selected_players['Player'])]
    top_remaining = remaining_players.sort_values(by='Projected_Points', ascending=False).head(3)

    # 3. Combine
    final_selection = pd.concat([selected_players, top_remaining])

    # 4. Sort for display (G -> D -> M -> F)
    final_selection = final_selection.sort_values(
        by=['Position', 'Projected_Points'],
        key=lambda x: x.map({'G': 0, 'D': 1, 'M': 2, 'F': 3}),
        ascending=[True, False]
    ).reset_index(drop=True)

    return final_selection


# ==============================================================================
# MERGING & NORMALIZATION
# ==============================================================================

def backfill_player_ids(roster_df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where Player_ID is NaN, fill via fuzzy match against fpl_stats.
    Constrained by Team and Position to reduce false positives.
    """
    df = roster_df.copy()

    # Create temp normalized columns for matching
    fpl_stats = fpl_stats.copy()
    fpl_stats["__name_norm"] = fpl_stats["Player"].apply(normalize_text)
    df["__name_norm"] = df["Player"].apply(normalize_text)

    if "Player_ID" not in df.columns:
        df["Player_ID"] = np.nan

    missing_idx = df[df["Player_ID"].isna()].index

    for i in missing_idx:
        name_norm = df.at[i, "__name_norm"]
        team = df.at[i, "Team"]
        pos = df.at[i, "Position"]

        # Filter candidates by Team + Position first
        scope = fpl_stats[(fpl_stats["Team"] == team) & (fpl_stats["Position"] == pos)]

        # Fallback 1: Just Position (if team name mismatch)
        if scope.empty:
            scope = fpl_stats[fpl_stats["Position"] == pos]

        # Fallback 2: All players
        if scope.empty:
            scope = fpl_stats

        if scope.empty:
            continue

        match = process.extractOne(name_norm, scope["__name_norm"].dropna().tolist(), scorer=fuzz.WRatio)

        if match:
            m_name, score = match[0], match[1]
            if score >= 85:
                pid = scope.loc[scope["__name_norm"] == m_name, "Player_ID"]
                if not pid.empty:
                    df.at[i, "Player_ID"] = float(pid.iloc[0])

    return df.drop(columns=["__name_norm"], errors="ignore")


def merge_fpl_players_and_projections(fpl_df: pd.DataFrame, roto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges FPL official data with Rotowire projections.
    1. Normalizes Rotowire data.
    2. Exact match on Name + Team.
    3. Fuzzy match on Name (within Team) for misses.
    """
    fpl = fpl_df.copy()
    roto = normalize_rotowire_data(roto_df)

    if '__name_norm' not in fpl.columns:
        fpl['__name_norm'] = fpl['Player'].apply(normalize_text)

    # Pass 1: Exact Merge
    merged = pd.merge(
        fpl,
        roto[['__name_norm', 'Rotowire_Team', 'Projected_Points', 'Rotowire_Price']],
        left_on=['__name_norm', 'Team'],
        right_on=['__name_norm', 'Rotowire_Team'],
        how='left'
    )

    # Pass 2: Fuzzy Match for missing projections
    missing_mask = merged['Projected_Points'].isna()

    for idx, row in merged[missing_mask].iterrows():
        fpl_name = row['__name_norm']
        team = row['Team']

        # Limit scope to team
        team_roto_subset = roto[roto['Rotowire_Team'] == team]

        if team_roto_subset.empty:
            continue

        match = process.extractOne(fpl_name, team_roto_subset['__name_norm'].tolist())

        if match and match[1] >= 85:
            matched_name = match[0]
            match_row = team_roto_subset[team_roto_subset['__name_norm'] == matched_name].iloc[0]

            merged.at[idx, 'Projected_Points'] = match_row['Projected_Points']
            merged.at[idx, 'Rotowire_Price'] = match_row['Rotowire_Price']

    # Cleanup
    merged['Projected_Points'] = pd.to_numeric(merged['Projected_Points'], errors='coerce').fillna(0.0)
    return merged.drop(columns=['__name_norm', 'Rotowire_Team'], errors='ignore')


def normalize_rotowire_data(roto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes Rotowire columns to match FPL schema.
    """
    df = roto_df.copy()

    # Map Columns
    # Input expected: 'Player', 'Team', 'Position', 'Points', 'Price'
    df = df.rename(columns={
        "Player": "Rotowire_Name",
        "Team": "Rotowire_Team",
        "Points": "Projected_Points",
        "Price": "Rotowire_Price"
    })

    # Map Positions
    df['Position'] = df['Position'].replace({'GK': 'G', 'FW': 'F'})

    # Aliases (Hardcoded fixes for known mismatches)
    aliases = {
        "Heung-Min Son": "Son Heung-min",
        "Matty Cash": "Matthew Cash",
        # Add more as discovered
    }
    df['Rotowire_Name'] = df['Rotowire_Name'].replace(aliases)

    # Normalize text
    df['__name_norm'] = df['Rotowire_Name'].apply(normalize_text)

    return df


# ==============================================================================
# STATS & SCORING ENRICHMENT
# ==============================================================================

def add_fdr_and_form(
        df: pd.DataFrame,
        fpl_stats: pd.DataFrame,
        current_gw: int,
        weeks_lookahead: int = 3
) -> pd.DataFrame:
    """
    Enriches a dataframe with 'AvgFDRNextN' and 'Form'.
    Requires Player/Team/Position columns.
    """
    base = df.copy()

    # Merge generic stats first (including Player_ID if missing)
    cols_to_merge = ["Player", "Team", "Position", "Player_ID", "form", "points_per_game"]
    # Safety: only merge cols that actually exist in fpl_stats
    cols_available = [c for c in cols_to_merge if c in fpl_stats.columns]

    base = base.merge(
        fpl_stats[cols_available],
        on=["Player", "Team", "Position"],
        how="left",
        suffixes=("", "_fpl")
    )

    # Coalesce Player_ID if it appeared from merge
    if "Player_ID_fpl" in base.columns:
        base["Player_ID"] = base["Player_ID"].fillna(base["Player_ID_fpl"])
        base.drop(columns=["Player_ID_fpl"], inplace=True)

    # Calculate Avg FDR
    # Note: Requires a helper or lookup table for fixtures.
    # Assuming simple calculation logic here or placeholders if specific fixture data isn't passed.
    base["AvgFDRNextN"] = 0.0  # Placeholder if fixture dict not available in this scope

    # Calculate Form (Fallback: FPL Form -> PPG -> 0)
    base["Form"] = pd.to_numeric(base["form"], errors="coerce")
    base["Form"] = base["Form"].fillna(pd.to_numeric(base["points_per_game"], errors="coerce"))
    base["Form"] = base["Form"].fillna(0.0)

    return base


def apply_availability_penalty(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """
    Multiplies a score column by (PlayPct / 100).
    Used to down-weight injured players in projections.
    """
    out = df.copy()
    pct = pd.to_numeric(out.get("PlayPct", 100), errors="coerce").fillna(100.0)
    score = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)

    out[out_col] = score * (pct / 100.0)
    return out


def attach_availability(df: pd.DataFrame, avail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges PlayPct, StatusBucket, and News onto the main dataframe.
    Tries Player+Team match first, then Player_ID.
    """
    base = df.copy()
    cols_keep = ["PlayPct", "StatusBucket", "News"]

    # 1. Merge on Name + Team
    base = base.merge(
        avail_df[["Player", "Team", "PlayPct", "StatusBucket", "News"]],
        on=["Player", "Team"],
        how="left",
        suffixes=("", "_new")
    )

    # 2. Backfill with ID if needed
    if "Player_ID" in base.columns and "Player_ID" in avail_df.columns:
        mask = base["PlayPct"].isna()
        if mask.any():
            # distinct dataframe for ID lookup to avoid duplicates
            id_lookup = avail_df[["Player_ID"] + cols_keep].drop_duplicates(subset=["Player_ID"])

            merged_ids = base.loc[mask, ["Player_ID"]].merge(id_lookup, on="Player_ID", how="left")

            base.loc[mask, "PlayPct"] = merged_ids["PlayPct"].values
            base.loc[mask, "StatusBucket"] = merged_ids["StatusBucket"].values
            base.loc[mask, "News"] = merged_ids["News"].values

    # Defaults
    base["PlayPct"] = pd.to_numeric(base["PlayPct"], errors="coerce").fillna(100.0)
    base["StatusBucket"] = base["StatusBucket"].fillna("Available")
    base["News"] = base["News"].fillna("")

    return base


# ==============================================================================
# TRANSFERS (CLASSIC & DRAFT)
# ==============================================================================

def get_transfer_recommendations(
        current_squad_df: pd.DataFrame,
        all_players_df: pd.DataFrame,
        bank_budget: float,
        sort_metric: str = 'Projected_Points'
) -> pd.DataFrame:
    """
    Generates 'Buy' recommendations for Classic mode.

    Logic:
    1. Iterates through current squad.
    2. Simulates selling player -> calculates available budget.
    3. Finds replacements in same position within budget.
    4. Returns Top 3 upgrades per position.
    """
    recommendations = []

    # Ensure we have IDs to prevent buying players we already own
    if 'Player_ID' in current_squad_df.columns:
        my_ids = set(current_squad_df['Player_ID'].dropna().tolist())
    else:
        my_ids = set()

    for idx, player in current_squad_df.iterrows():
        current_pos = player['Position']
        # Classic prices are often 10x in API (e.g. 100 = 10.0). Adjust accordingly if needed.
        sell_price = player.get('now_cost', 0)

        available_funds = bank_budget + sell_price

        # Candidates: Same Pos, Affordable, Not Owned
        candidates = all_players_df[
            (all_players_df['Position'] == current_pos) &
            (all_players_df['now_cost'] <= available_funds) &
            (~all_players_df['Player_ID'].isin(my_ids))
            ].copy()

        if candidates.empty:
            continue

        # Calculate Gain
        current_pts = player.get(sort_metric, 0)
        candidates['Point_Diff'] = candidates[sort_metric] - current_pts

        # Get Top 3 Better Options
        better_options = candidates[candidates['Point_Diff'] > 0].sort_values(
            by='Point_Diff', ascending=False
        ).head(3)

        for _, option in better_options.iterrows():
            recommendations.append({
                'Sell': player['Player'],
                'Buy': option['Player'],
                'Cost_Diff': option['now_cost'] - sell_price,
                'Projected_Gain': option['Point_Diff'],
                'New_Player_FDR': option.get('AvgFDRNextN', 0)
            })

    if not recommendations:
        return pd.DataFrame()

    return pd.DataFrame(recommendations).sort_values(by='Projected_Gain', ascending=False)