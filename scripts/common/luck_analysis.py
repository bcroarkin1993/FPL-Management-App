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
    - DataFrame with columns: gameweek, team, team_id, score

    The id is carried so `calculate_all_play_standings` can key on it: two
    squads sharing a name would otherwise be merged into one row. See the
    note there.
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

    return pd.DataFrame(rows, columns=['gameweek', 'team', 'team_id', 'score'])


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

    # **Keyed on the team id where there is one, never on the name.** Two
    # managers who have not renamed their squad both show as "Change Name", and
    # keying on that merged them into one row: their all-play records were
    # summed, they were counted against *each other*, and the id carried
    # through was whichever appeared first. Reported from the app, and it is the
    # same "a name is not a join key" failure recorded all over this codebase.
    #
    # The Draft extractors publish no id, so the name remains the fallback --
    # which preserves their behaviour exactly, including that latent bug if two
    # Draft squads ever share a name.
    key_col = 'team_id' if has_team_id else 'team'
    names = {}
    for key, name in zip(gw_scores[key_col], gw_scores['team']):
        names.setdefault(key, name)

    # Accumulate all-play results per team
    ap_results = {}  # key -> {'w': 0, 'd': 0, 'l': 0}

    for gw in gameweeks:
        gw_data = gw_scores[gw_scores['gameweek'] == gw]
        keys = gw_data[key_col].tolist()
        scores = gw_data['score'].tolist()

        for i, (key_i, score_i) in enumerate(zip(keys, scores)):
            if key_i not in ap_results:
                ap_results[key_i] = {'w': 0, 'd': 0, 'l': 0}

            for j, (key_j, score_j) in enumerate(zip(keys, scores)):
                if i == j:
                    continue
                if score_i > score_j:
                    ap_results[key_i]['w'] += 1
                elif score_i == score_j:
                    ap_results[key_i]['d'] += 1
                else:
                    ap_results[key_i]['l'] += 1

    # Build per-team aggregates
    team_stats = []
    for key, record in ap_results.items():
        total_matches = record['w'] + record['d'] + record['l']
        win_pct = (record['w'] + 0.5 * record['d']) / total_matches * 100 if total_matches > 0 else 0.0

        team_gw = gw_scores[gw_scores[key_col] == key]
        avg_score = team_gw['score'].mean()

        # Rank within each GW (lower = better)
        ranks = []
        for gw in team_gw['gameweek'].values:
            gw_data = gw_scores[gw_scores['gameweek'] == gw]
            own = team_gw[team_gw['gameweek'] == gw]['score'].values[0]
            ranks.append(int((gw_data['score'] > own).sum()) + 1)
        avg_gw_rank = sum(ranks) / len(ranks) if ranks else 0

        row = {
            'Team': names.get(key, key),
            'AP W': record['w'],
            'AP D': record['d'],
            'AP L': record['l'],
            'AP Win%': round(win_pct, 1),
            'Avg Score': round(avg_score, 1),
            'Avg GW Rank': round(avg_gw_rank, 2),
        }
        if has_team_id:
            row['team_id'] = key

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


def calculate_h2h_streaks(weekly_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate W/D/L streaks for all teams from H2H match data.

    Parameters:
    - weekly_scores: DataFrame with columns Team, Gameweek, Points, Opp_Points

    Returns:
    - DataFrame with columns: Team, Team_ID, Current_Streak, Longest_Win_Streak,
      Longest_Unbeaten_Streak, Longest_Loss_Streak. Sorted by Longest_Win_Streak
      desc. ``Team_ID`` is the key the rows were grouped on, published so a
      caller can join on it rather than on the name -- it is the name where the
      input carried no id.
    """
    if weekly_scores.empty:
        return pd.DataFrame()

    streak_data = []

    # Grouped on the id where the frame carries one -- see
    # `build_classic_h2h_weekly_scores`. The Draft callers pass names only, so
    # the name remains the fallback.
    key_col = "Team_ID" if "Team_ID" in weekly_scores.columns else "Team"
    names = {}
    for key, name in zip(weekly_scores[key_col], weekly_scores["Team"]):
        names.setdefault(key, name)

    for key in weekly_scores[key_col].unique():
        team = names.get(key, key)
        team_df = weekly_scores[weekly_scores[key_col] == key].sort_values("Gameweek")

        results = []
        for _, row in team_df.iterrows():
            if row["Points"] > row["Opp_Points"]:
                results.append("W")
            elif row["Points"] < row["Opp_Points"]:
                results.append("L")
            else:
                results.append("D")

        if not results:
            streak_data.append({
                "Team": team,
                "Team_ID": key,
                "Current_Streak": "-",
                "Longest_Win_Streak": 0,
                "Longest_Unbeaten_Streak": 0,
                "Longest_Loss_Streak": 0,
            })
            continue

        # Current streak: count matching tail results
        current_type = results[-1]
        current_streak = 1
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current_type:
                current_streak += 1
            else:
                break

        # Longest win / loss streaks
        longest_win = 0
        longest_loss = 0
        streak = 1
        for i in range(1, len(results)):
            if results[i] == results[i - 1]:
                streak += 1
            else:
                if results[i - 1] == "W":
                    longest_win = max(longest_win, streak)
                elif results[i - 1] == "L":
                    longest_loss = max(longest_loss, streak)
                streak = 1
        if results[-1] == "W":
            longest_win = max(longest_win, streak)
        elif results[-1] == "L":
            longest_loss = max(longest_loss, streak)

        # Longest unbeaten run (W or D)
        cur_unbeaten = 0
        longest_unbeaten = 0
        for r in results:
            if r in ("W", "D"):
                cur_unbeaten += 1
                longest_unbeaten = max(longest_unbeaten, cur_unbeaten)
            else:
                cur_unbeaten = 0

        streak_data.append({
            "Team": team,
            "Team_ID": key,
            "Current_Streak": f"{current_streak}{current_type}",
            "Longest_Win_Streak": longest_win,
            "Longest_Unbeaten_Streak": longest_unbeaten,
            "Longest_Loss_Streak": longest_loss,
        })

    return pd.DataFrame(streak_data).sort_values("Longest_Win_Streak", ascending=False)


def build_classic_h2h_weekly_scores(matches: list) -> pd.DataFrame:
    """
    Convert Classic H2H match list to weekly scores format.

    Parameters:
    - matches: List of match dicts from get_all_h2h_league_matches()

    Returns:
    - DataFrame with columns: Team, Team_ID, Gameweek, Points, Opp_Points

    ``Team_ID`` is what `calculate_h2h_streaks` groups on: two managers who have
    not renamed their squad share the name "Change Name", and grouping on that
    interleaved their results into a single streak row.
    """
    rows = []
    for m in matches:
        p1 = m.get("entry_1_points", 0)
        p2 = m.get("entry_2_points", 0)
        if not m.get("finished", False) and p1 == 0 and p2 == 0:
            continue
        gw = m.get("event", 0)
        t1 = m.get("entry_1_name", "Unknown")
        t2 = m.get("entry_2_name", "Unknown")
        rows.append({"Team": t1, "Team_ID": m.get("entry_1_entry"),
                     "Gameweek": gw, "Points": p1, "Opp_Points": p2})
        rows.append({"Team": t2, "Team_ID": m.get("entry_2_entry"),
                     "Gameweek": gw, "Points": p2, "Opp_Points": p1})

    if not rows:
        return pd.DataFrame(columns=["Team", "Team_ID", "Gameweek", "Points",
                                     "Opp_Points"])
    return pd.DataFrame(rows).sort_values(["Team", "Gameweek"])


def format_streak_html(val: str) -> str:
    """Wrap a streak string (e.g. '3W', '2L', '1D') in a colored HTML span."""
    if not val or val == "-":
        return val
    last = val[-1]
    if last == "W":
        color = "#4CAF50"
    elif last == "L":
        color = "#f44336"
    else:
        color = "#9e9e9e"
    return f'<span style="color:{color};font-weight:600">{val}</span>'


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
