# scripts/common/luck_analysis.py
"""
Shared All-Play Record calculations for luck-adjusted standings.

Provides:
- extract_draft_gw_scores: Convert Draft API match data to per-team GW scores
- extract_classic_h2h_gw_scores: Convert Classic H2H matches to per-team GW scores
- calculate_all_play_standings: Core all-play algorithm comparing every team vs every other
"""

import pandas as pd
import streamlit as st
from typing import Optional
from scripts.common.styled_tables import render_styled_table


def extract_draft_gw_scores(league_response: dict) -> pd.DataFrame:
    """
    Convert Draft API match data to long format: (gameweek, team, score).

    Skips matches where BOTH teams scored 0 (unplayed matches).
    A match where one team scored 0 is a valid result.

    Parameters:
    - league_response: Full league details dict with 'matches' and 'league_entries'.

    Returns:
    - DataFrame with columns: gameweek, team, score
    """
    matches = league_response.get('matches', [])
    league_entries = league_response.get('league_entries', [])
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    rows = []
    for match in matches:
        team1_score = match['league_entry_1_points']
        team2_score = match['league_entry_2_points']

        # Only skip if BOTH teams scored 0 (unplayed)
        if team1_score == 0 and team2_score == 0:
            continue

        gw = match['event']
        team1_name = team_names.get(match['league_entry_1'], 'Unknown')
        team2_name = team_names.get(match['league_entry_2'], 'Unknown')

        rows.append({'gameweek': gw, 'team': team1_name, 'score': team1_score})
        rows.append({'gameweek': gw, 'team': team2_name, 'score': team2_score})

    return pd.DataFrame(rows, columns=['gameweek', 'team', 'score'])


def extract_classic_h2h_gw_scores(matches: list) -> pd.DataFrame:
    """
    Convert Classic H2H match list to long format: (gameweek, team, team_id, score).

    Skips unplayed matches (finished == False and both scores 0).

    Parameters:
    - matches: List of match dicts from get_all_h2h_league_matches().

    Returns:
    - DataFrame with columns: gameweek, team, team_id, score
    """
    rows = []
    for match in matches:
        score1 = match.get('entry_1_points', 0)
        score2 = match.get('entry_2_points', 0)

        # Skip unplayed matches
        if not match.get('finished', False) and score1 == 0 and score2 == 0:
            continue

        gw = match.get('event', 0)
        rows.append({
            'gameweek': gw,
            'team': match.get('entry_1_name', 'Unknown'),
            'team_id': match.get('entry_1_entry', 0),
            'score': score1,
        })
        rows.append({
            'gameweek': gw,
            'team': match.get('entry_2_name', 'Unknown'),
            'team_id': match.get('entry_2_entry', 0),
            'score': score2,
        })

    return pd.DataFrame(rows, columns=['gameweek', 'team', 'team_id', 'score'])


def calculate_all_play_standings(
    gw_scores: pd.DataFrame,
    actual_standings: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculate All-Play Record standings.

    For each gameweek, every team is compared against every other team.
    A team earns an all-play "win" for each opponent it outscored, a "draw"
    for ties, and a "loss" for each opponent that outscored it.

    Parameters:
    - gw_scores: DataFrame with columns: gameweek, team, score.
                 May also contain 'team_id'.
    - actual_standings: Optional DataFrame with columns:
        - 'team' (and optionally 'team_id')
        - 'actual_rank'
        - 'actual_pts'
      If provided, merges in actual rank/pts and computes Luck Delta.

    Returns:
    - DataFrame with columns: Fair Rank, Team, AP W, AP D, AP L, AP Win%,
      Avg Score, Avg GW Rank, and optionally Actual Rank, Actual Pts, Luck +/-.
      Sorted by Fair Rank.
    """
    if gw_scores.empty:
        return pd.DataFrame()

    gameweeks = gw_scores['gameweek'].unique()
    has_team_id = 'team_id' in gw_scores.columns

    # Accumulate all-play results per team
    ap_results = {}  # team -> {'w': 0, 'd': 0, 'l': 0}

    for gw in gameweeks:
        gw_data = gw_scores[gw_scores['gameweek'] == gw]
        teams = gw_data['team'].tolist()
        scores = gw_data['score'].tolist()

        for i, (team_i, score_i) in enumerate(zip(teams, scores)):
            if team_i not in ap_results:
                ap_results[team_i] = {'w': 0, 'd': 0, 'l': 0}

            for j, (team_j, score_j) in enumerate(zip(teams, scores)):
                if i == j:
                    continue
                if score_i > score_j:
                    ap_results[team_i]['w'] += 1
                elif score_i == score_j:
                    ap_results[team_i]['d'] += 1
                else:
                    ap_results[team_i]['l'] += 1

    # Build per-team aggregates
    team_stats = []
    for team, record in ap_results.items():
        total_matches = record['w'] + record['d'] + record['l']
        win_pct = (record['w'] + 0.5 * record['d']) / total_matches * 100 if total_matches > 0 else 0.0

        # Calculate avg score and avg GW rank
        team_gw = gw_scores[gw_scores['team'] == team]
        avg_score = team_gw['score'].mean()

        # Rank within each GW (lower = better)
        ranks = []
        for gw in team_gw['gameweek'].values:
            gw_data = gw_scores[gw_scores['gameweek'] == gw]
            rank = (gw_data['score'] > team_gw[team_gw['gameweek'] == gw]['score'].values[0]).sum() + 1
            ranks.append(rank)
        avg_gw_rank = sum(ranks) / len(ranks) if ranks else 0

        row = {
            'Team': team,
            'AP W': record['w'],
            'AP D': record['d'],
            'AP L': record['l'],
            'AP Win%': round(win_pct, 1),
            'Avg Score': round(avg_score, 1),
            'Avg GW Rank': round(avg_gw_rank, 2),
        }

        # Carry team_id if available
        if has_team_id:
            team_ids = gw_scores[gw_scores['team'] == team]['team_id'].unique()
            if len(team_ids) > 0:
                row['team_id'] = team_ids[0]

        team_stats.append(row)

    result_df = pd.DataFrame(team_stats)

    if result_df.empty:
        return result_df

    # Fair Rank by AP Win% (descending) then Avg Score (descending)
    result_df = result_df.sort_values(['AP Win%', 'Avg Score'], ascending=[False, False])
    result_df['Fair Rank'] = range(1, len(result_df) + 1)

    # Merge actual standings if provided
    if actual_standings is not None and not actual_standings.empty:
        # Determine merge key
        if 'team_id' in actual_standings.columns and 'team_id' in result_df.columns:
            merge_key = 'team_id'
        else:
            merge_key = None

        if merge_key:
            result_df = result_df.merge(
                actual_standings[['team_id', 'actual_rank', 'actual_pts']],
                on='team_id',
                how='left',
            )
        else:
            # Merge on team name
            result_df = result_df.merge(
                actual_standings[['team', 'actual_rank', 'actual_pts']].rename(columns={'team': 'Team'}),
                on='Team',
                how='left',
            )

        result_df.rename(columns={
            'actual_rank': 'Actual Rank',
            'actual_pts': 'Actual Pts',
        }, inplace=True)

        result_df['Luck +/-'] = result_df['Actual Rank'] - result_df['Fair Rank']

    # Select final columns
    columns = ['Fair Rank', 'Team', 'AP W', 'AP D', 'AP L', 'AP Win%', 'Avg Score', 'Avg GW Rank']
    if 'Actual Rank' in result_df.columns:
        columns += ['Actual Rank', 'Actual Pts', 'Luck +/-']

    result_df = result_df[columns]
    result_df = result_df.set_index('Fair Rank')

    return result_df


def render_luck_adjusted_table(df: pd.DataFrame):
    """
    Render a luck-adjusted standings DataFrame with text color styling.

    Applies text color gradients:
    - Green for higher AP Win%, Avg Score, AP W (positive)
    - Red for higher Avg GW Rank, AP L (negative — lower is better)
    - Diverging on Luck +/- (positive = unlucky = red, negative = lucky = green)
    """
    if df.empty:
        st.warning("Not enough match data to calculate luck-adjusted standings.")
        return

    # Reset index to avoid issues
    if df.index.name:
        df = df.reset_index()

    positive_cols = [c for c in ['AP W', 'AP Win%', 'Avg Score'] if c in df.columns]
    negative_cols = [c for c in ['AP L', 'Avg GW Rank'] if c in df.columns]

    # Luck +/- : positive = unlucky (red), negative = lucky (green) → negative_color_cols
    if 'Luck +/-' in df.columns:
        negative_cols.append('Luck +/-')

    col_fmts = {}
    if 'AP Win%' in df.columns:
        col_fmts['AP Win%'] = '{:.1f}%'
    if 'Avg Score' in df.columns:
        col_fmts['Avg Score'] = '{:.1f}'
    if 'Avg GW Rank' in df.columns:
        col_fmts['Avg GW Rank'] = '{:.2f}'

    render_styled_table(
        df,
        col_formats=col_fmts,
        positive_color_cols=positive_cols,
        negative_color_cols=negative_cols,
    )


def render_standings_table(df: pd.DataFrame, is_h2h: bool = True):
    """
    Render a regular league standings DataFrame with text color styling.

    Parameters:
    - df: Standings DataFrame (already formatted for display, no internal IDs).
    - is_h2h: True for H2H standings (W/D/L/PF/PA/Pts), False for Classic (GW Pts/Total Pts).
    """
    if df.empty:
        st.info("No standings data available yet.")
        return

    # Reset index to avoid non-unique index issues
    if df.index.name:
        df = df.reset_index()

    if is_h2h:
        positive_cols = [c for c in ['W', 'PF', 'Pts'] if c in df.columns]
        negative_cols = [c for c in ['L'] if c in df.columns]
    else:
        positive_cols = [c for c in ['Total Pts', 'GW Pts'] if c in df.columns]
        negative_cols = []

    render_styled_table(
        df,
        positive_color_cols=positive_cols,
        negative_color_cols=negative_cols,
    )
