import config
import math
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scripts.common.utils import (
    find_optimal_lineup, format_team_name, get_current_gameweek, get_gameweek_fixtures,
    get_team_id_by_name, get_rotowire_player_projections, get_team_composition_for_gameweek,
    merge_fpl_players_and_projections, normalize_apostrophes, get_historical_team_scores,
    get_draft_h2h_record, get_live_gameweek_stats, is_gameweek_live, get_fpl_player_mapping,
    get_team_actual_lineup
)

def _blend_live_with_projections(team_df: pd.DataFrame, live_stats: dict, player_mapping: dict) -> pd.DataFrame:
    """
    Blend live points with projections for players who haven't played yet.

    For each player:
    - If they have played (minutes > 0): use actual points
    - If they haven't played yet: use projected points

    Returns DataFrame with additional columns:
    - 'Live_Points': actual points scored (0 if not played)
    - 'Has_Played': bool indicating if player has played
    - 'Blended_Points': live points if played, projected if not
    """
    from scripts.common.player_matching import canonical_normalize

    result = team_df.copy()

    # Create multiple lookups for matching: name -> element_id
    name_to_id = {}
    norm_to_id = {}  # normalized name -> element_id
    last_name_to_ids = {}  # last name -> list of (element_id, team)

    for eid, pdata in player_mapping.items():
        if isinstance(pdata, dict):
            web_name = pdata.get('Web_Name', '')
            full_name = pdata.get('Player', '')
            team = pdata.get('Team', '')

            if web_name:
                name_to_id[web_name] = eid
                norm_to_id[canonical_normalize(web_name)] = eid
            if full_name:
                name_to_id[full_name] = eid
                norm_to_id[canonical_normalize(full_name)] = eid
                # Store first + last name combo (e.g., "Robert Sánchez" from "Robert Lynch Sánchez")
                parts = full_name.split()
                if len(parts) >= 2:
                    first_last = f"{parts[0]} {parts[-1]}"
                    norm_to_id[canonical_normalize(first_last)] = eid
                    # Also store last name only for fallback matching
                    last_norm = canonical_normalize(parts[-1])
                    if last_norm not in last_name_to_ids:
                        last_name_to_ids[last_norm] = []
                    last_name_to_ids[last_norm].append((eid, team))

    result['Live_Points'] = 0
    result['Has_Played'] = False
    result['Blended_Points'] = result['Points'].fillna(0)

    # Get team info from DataFrame if available
    team_col = 'Team' if 'Team' in result.columns else None

    # Player name is in the index for this DataFrame
    for idx in result.index:
        player_name = idx  # Index is the player name
        player_team = result.at[idx, team_col] if team_col else None

        # Try multiple matching strategies
        element_id = None

        # Strategy 1: Direct match
        if player_name in name_to_id:
            element_id = name_to_id[player_name]

        # Strategy 2: Normalized match
        if element_id is None:
            norm_name = canonical_normalize(player_name)
            if norm_name in norm_to_id:
                element_id = norm_to_id[norm_name]

        # Strategy 3: First + last name match (handles middle names)
        if element_id is None:
            parts = player_name.split()
            if len(parts) >= 2:
                first_last_norm = canonical_normalize(f"{parts[0]} {parts[-1]}")
                if first_last_norm in norm_to_id:
                    element_id = norm_to_id[first_last_norm]

        # Strategy 4: Last name only with team disambiguation
        if element_id is None:
            parts = player_name.split()
            if parts:
                last_norm = canonical_normalize(parts[-1])
                if last_norm in last_name_to_ids:
                    candidates = last_name_to_ids[last_norm]
                    if len(candidates) == 1:
                        # Only one player with this last name
                        element_id = candidates[0][0]
                    elif player_team:
                        # Try to match by team
                        for eid, team in candidates:
                            if team == player_team:
                                element_id = eid
                                break

        if element_id and element_id in live_stats:
            stats = live_stats[element_id]
            result.at[idx, 'Live_Points'] = stats.get('points', 0)
            result.at[idx, 'Has_Played'] = stats.get('has_played', False)

            if stats.get('has_played', False):
                result.at[idx, 'Blended_Points'] = stats.get('points', 0)

    return result


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _estimate_score_std(league_id: int) -> tuple[float, int]:  # <<< ADD
    """
    Returns (std, n) for historical single-team weekly scores if available.
    Tries: scripts.common.utils.get_historical_team_scores(league_id) -> DataFrame with 'total_points' or 'score'.
    Fallback: CSV path in config.HISTORICAL_SCORES_CSV (or 'data/historical_team_scores.csv').
    Final fallback: (15.0, 0) — a reasonable league-wide prior.
    """
    # Try utils function if it exists
    try:
        hist = get_historical_team_scores(league_id)
    except Exception:
        hist = None
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        col = 'total_points' if 'total_points' in hist.columns else ('score' if 'score' in hist.columns else None)
        if col:
            s = pd.to_numeric(hist[col], errors='coerce').dropna()
            if len(s) >= 2:
                return float(s.std(ddof=1)), int(len(s))
    # Try CSV from config or default path
    try:
        csv_path = getattr(config, 'HISTORICAL_SCORES_CSV', 'data/historical_team_scores.csv')
        df = pd.read_csv(csv_path)
        col = 'total_points' if 'total_points' in df.columns else ('score' if 'score' in df.columns else None)
        if col:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) >= 2:
                return float(s.std(ddof=1)), int(len(s))
    except Exception:
        pass
    return 15.0, 0  # conservative default if nothing available


def _render_team_lineup(team_df: pd.DataFrame, team_name: str, is_live: bool = False):
    """
    Render a styled team lineup with player cards grouped by position.
    Shows live points, projected points, and performance indicators.
    """
    # Position display order and colors
    pos_config = {
        'G': {'name': 'Goalkeeper', 'color': '#f39c12', 'short': 'GK'},
        'D': {'name': 'Defenders', 'color': '#3498db', 'short': 'DEF'},
        'M': {'name': 'Midfielders', 'color': '#2ecc71', 'short': 'MID'},
        'F': {'name': 'Forwards', 'color': '#e74c3c', 'short': 'FWD'},
    }

    html = f"""
    <style>
        .lineup-container {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .pos-group {{ margin-bottom: 12px; }}
        .pos-header {{
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; padding: 6px 10px; border-radius: 4px;
            margin-bottom: 6px; color: white;
        }}
        .player-card {{
            display: flex; align-items: center; justify-content: space-between;
            background: #f8f9fa; border-radius: 6px; padding: 8px 12px;
            margin-bottom: 4px; border-left: 3px solid #ddd;
        }}
        .player-card.played {{ border-left-color: #28a745; background: #f0fff4; }}
        .player-card.upcoming {{ border-left-color: #6c757d; }}
        .player-info {{ flex: 1; min-width: 0; }}
        .player-name {{ font-weight: 600; font-size: 13px; color: #1a1a2e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .player-team {{ font-size: 10px; color: #888; text-transform: uppercase; }}
        .player-matchup {{ font-size: 10px; color: #666; }}
        .player-points {{ text-align: right; min-width: 70px; }}
        .live-pts {{ font-size: 18px; font-weight: 700; color: #28a745; }}
        .proj-pts {{ font-size: 12px; color: #666; }}
        .proj-only {{ font-size: 16px; font-weight: 600; color: #555; }}
        .perf-indicator {{ font-size: 10px; margin-top: 2px; }}
        .perf-up {{ color: #28a745; }}
        .perf-down {{ color: #dc3545; }}
        .status-badge {{
            font-size: 9px; padding: 2px 6px; border-radius: 3px;
            text-transform: uppercase; font-weight: 600; margin-left: 8px;
        }}
        .status-played {{ background: #d4edda; color: #155724; }}
        .status-upcoming {{ background: #e2e3e5; color: #383d41; }}
    </style>
    <div class="lineup-container">
    """

    # Group players by position
    for pos_code in ['G', 'D', 'M', 'F']:
        pos_info = pos_config.get(pos_code, {'name': pos_code, 'color': '#888', 'short': pos_code})

        # Filter players for this position
        if 'Position' in team_df.columns:
            pos_players = team_df[team_df['Position'] == pos_code]
        else:
            pos_players = pd.DataFrame()

        if pos_players.empty:
            continue

        html += f"""
        <div class="pos-group">
            <div class="pos-header" style="background: {pos_info['color']};">{pos_info['name']}</div>
        """

        for player_name in pos_players.index:
            row = pos_players.loc[player_name]
            team = row.get('Team', '')
            matchup = row.get('Matchup', '')
            proj_pts = row.get('Points', 0) or 0

            if is_live:
                live_pts = row.get('Live_Points', 0) or 0
                has_played = row.get('Has_Played', False)
                blended_pts = row.get('Blended_Points', 0) or 0

                if has_played:
                    # Player has finished - show actual vs projected
                    diff = live_pts - proj_pts
                    diff_sign = "+" if diff > 0 else ""
                    diff_class = "perf-up" if diff > 0 else "perf-down" if diff < 0 else ""

                    points_html = f"""
                        <div class="live-pts">{live_pts:.0f}</div>
                        <div class="perf-indicator {diff_class}">proj: {proj_pts:.1f} ({diff_sign}{diff:.1f})</div>
                    """
                    card_class = "player-card played"
                    status_html = '<span class="status-badge status-played">Played</span>'
                else:
                    # Player yet to play - show projected
                    points_html = f"""
                        <div class="proj-only">{proj_pts:.1f}</div>
                        <div class="proj-pts">projected</div>
                    """
                    card_class = "player-card upcoming"
                    status_html = '<span class="status-badge status-upcoming">Upcoming</span>'
            else:
                # Pre-match: just show projections
                points_html = f'<div class="proj-only">{proj_pts:.1f}</div>'
                card_class = "player-card"
                status_html = ""

            html += f"""
            <div class="{card_class}">
                <div class="player-info">
                    <div class="player-name">{player_name}{status_html}</div>
                    <div class="player-team">{team}</div>
                    <div class="player-matchup">{matchup}</div>
                </div>
                <div class="player-points">{points_html}</div>
            </div>
            """

        html += "</div>"

    html += "</div>"

    # Calculate total height based on player count
    player_count = len(team_df)
    # Count actual position groups present
    pos_groups = team_df['Position'].nunique() if 'Position' in team_df.columns else 4
    height = 40 + (player_count * 56) + (pos_groups * 32)
    components.html(html, height=height, scrolling=False)


# --- Win % bar (two-color) ---
def _render_winprob_bar(team1_name: str, team2_name: str, p_team1: float):
    p1 = max(0.0, min(100.0, round(p_team1 * 100, 1)))
    p2 = round(100.0 - p1, 1)
    html = f"""
    <style>
      .wpb-wrap {{
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
      }}
      .wpb-labels, .wpb-bar {{
        display: grid;
        grid-template-columns: {p1}% {p2}%;
        gap: 0;
        width: 100%;
      }}
      .wpb-labels div {{
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.2;
        white-space: nowrap;
      }}
      .wpb-bar {{
        height: 36px;                  /* thicker bar */
        border-radius: 9999px;
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
      }}
      .wpb-left  {{ background: #2563eb; }}  /* blue  */
      .wpb-right {{ background: #dc2626; }}  /* red   */
      .wpb-subtle {{ color: rgba(0,0,0,0.65); }}
    </style>
    <div class="wpb-wrap">
      <div class="wpb-labels">
        <div class="wpb-subtle">{team1_name} {p1}%</div>
        <div class="wpb-subtle">{p2}% {team2_name}</div>
      </div>
      <div class="wpb-bar" role="img" aria-label="Win probability: {team1_name} {p1} percent, {team2_name} {p2} percent.">
        <div class="wpb-left"></div>
        <div class="wpb-right"></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def analyze_fixture_projections(fixture, league_id, projections_df, use_actual_lineup: bool = False):
    """
    Returns two DataFrames representing lineups and points for each team in a fixture,
    sorted by position (GK, DEF, MID, FWD) and then by descending projected points within each position.

    Parameters:
    - fixture (str): The selected fixture, formatted as "Team1 (Player1) vs Team2 (Player2)".
    - league_id (int): The ID of the FPL Draft league.
    - projections_df (DataFrame): DataFrame containing player projections from Rotowire.
    - use_actual_lineup (bool): If True, use the manager's actual starting 11 picks.
                                If False, calculate the optimal lineup by projections.

    Returns:
    - Tuple of two DataFrames: (team1_df, team2_df, team1_name, team2_name)
    """
    # Normalize the apostrophes in the fixture string
    fixture = normalize_apostrophes(fixture)

    # Extract the team names only (ignore player names inside parentheses)
    team1_name = fixture.split(' vs ')[0].split(' (')[0].strip()
    team2_name = fixture.split(' vs ')[1].split(' (')[0].strip()

    # Get the team ids based on the team names
    team1_id = get_team_id_by_name(league_id, team1_name)
    team2_id = get_team_id_by_name(league_id, team2_name)

    # Get the current gameweek
    gameweek = get_current_gameweek()

    if use_actual_lineup:
        # Use actual picks from the FPL Draft API
        team1_actual = get_team_actual_lineup(team1_id, gameweek)
        team2_actual = get_team_actual_lineup(team2_id, gameweek)

        # Filter to starters only (positions 1-11)
        team1_starters = team1_actual[team1_actual['Is_Starter'] == True].copy()
        team2_starters = team2_actual[team2_actual['Is_Starter'] == True].copy()

        # Merge with projections
        team1_df = merge_fpl_players_and_projections(
            team1_starters[['Player', 'Team', 'Position']],
            projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
        )
        team2_df = merge_fpl_players_and_projections(
            team2_starters[['Player', 'Team', 'Position']],
            projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
        )
    else:
        # Use optimal lineup calculation (original behavior)
        team1_composition = get_team_composition_for_gameweek(league_id, team1_id, gameweek)
        team2_composition = get_team_composition_for_gameweek(league_id, team2_id, gameweek)

        # Merge FPL players with projections for both teams
        team1_df = merge_fpl_players_and_projections(
            team1_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
        )
        team2_df = merge_fpl_players_and_projections(
            team2_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
        )

    # Check if 'Points' column exists
    if 'Points' not in team1_df or 'Points' not in team2_df:
        print("Error: 'Points' column not found in one or both dataframes.")
        return None

    # Fill NaN values in 'Points' column with 0.0
    team1_df['Points'] = pd.to_numeric(team1_df['Points'], errors='coerce').fillna(0.0)
    team2_df['Points'] = pd.to_numeric(team2_df['Points'], errors='coerce').fillna(0.0)

    if not use_actual_lineup:
        # Find the optimal lineup (top 11 players) for each team
        team1_df = find_optimal_lineup(team1_df)
        team2_df = find_optimal_lineup(team2_df)

    # Define the position order for sorting
    position_order = ['G', 'D', 'M', 'F']
    for df in [team1_df, team2_df]:
        df['Position'] = pd.Categorical(df['Position'], categories=position_order, ordered=True)
        df.sort_values(by=['Position', 'Points'], ascending=[True, False], inplace=True)

    # Select the final columns to use
    team1_df = team1_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    team2_df = team2_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]

    # Format team DataFrames to use player names as the index
    team1_df.set_index('Player', inplace=True)
    team2_df.set_index('Player', inplace=True)

    # Return the final DataFrames and team names
    return team1_df, team2_df, team1_name, team2_name

def _get_win_pct_color(pct: float) -> str:
    """
    Returns a color on a red-to-green gradient based on win percentage.
    Uses a compressed scale so colors diverge more quickly from 50%.

    0-35% = strong red
    35-45% = red to yellow
    45-55% = yellow (narrow band)
    55-65% = yellow to green
    65-100% = strong green
    """
    if pct <= 35:
        # Strong red
        return "rgb(220, 53, 69)"  # Bootstrap danger red
    elif pct <= 45:
        # Red to Yellow (35-45%)
        ratio = (pct - 35) / 10
        r = 220 + int((255 - 220) * ratio)  # 220 to 255
        g = 53 + int((193 - 53) * ratio)    # 53 to 193
        b = 69 - int((69 - 7) * ratio)      # 69 to 7
        return f"rgb({r}, {g}, {b})"
    elif pct <= 55:
        # Yellow zone (45-55%) - narrow band
        ratio = (pct - 45) / 10
        r = 255 - int((255 - 200) * ratio)  # 255 to 200
        g = 193 + int((200 - 193) * ratio)  # 193 to 200
        b = 7 + int((80 - 7) * ratio)       # 7 to 80
        return f"rgb({r}, {g}, {b})"
    elif pct <= 65:
        # Yellow to Green (55-65%)
        ratio = (pct - 55) / 10
        r = 200 - int((200 - 40) * ratio)   # 200 to 40
        g = 200 - int((200 - 167) * ratio)  # 200 to 167
        b = 80 - int((80 - 69) * ratio)     # 80 to 69
        return f"rgb({r}, {g}, {b})"
    else:
        # Strong green (65%+)
        return "rgb(40, 167, 69)"  # Bootstrap success green


def _render_fixtures_overview(fixtures: list, league_id: int, projections_df: pd.DataFrame, sigma: float,
                              live_stats: dict = None, player_mapping: dict = None, gw_is_live: bool = False):
    """
    Render an overview table showing all fixtures with projected scores and win probabilities.
    If gw_is_live, blends actual points with projections for remaining players.
    """
    if not fixtures:
        return

    live_stats = live_stats or {}
    player_mapping = player_mapping or {}

    overview_data = []
    denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0

    spinner_msg = "Calculating live scores..." if gw_is_live else "Calculating projections for all fixtures..."
    with st.spinner(spinner_msg):
        for fixture in fixtures:
            try:
                # Use actual lineups for live gameweeks, optimal projections otherwise
                result = analyze_fixture_projections(fixture, league_id, projections_df, use_actual_lineup=gw_is_live)
                if result is None:
                    continue

                team1_df, team2_df, team1_name, team2_name = result

                # Store original projections before blending
                team1_orig_proj = team1_df['Points'].sum()
                team2_orig_proj = team2_df['Points'].sum()

                # Blend live points with projections if gameweek is live
                if gw_is_live and live_stats:
                    team1_df = _blend_live_with_projections(team1_df, live_stats, player_mapping)
                    team2_df = _blend_live_with_projections(team2_df, live_stats, player_mapping)
                    team1_blended = team1_df['Blended_Points'].sum()
                    team2_blended = team2_df['Blended_Points'].sum()
                    team1_live = team1_df['Live_Points'].sum()
                    team2_live = team2_df['Live_Points'].sum()
                else:
                    team1_blended = team1_df['Points'].sum()
                    team2_blended = team2_df['Points'].sum()
                    team1_live = 0
                    team2_live = 0

                # Calculate win probability based on blended/projected scores
                z = (team1_blended - team2_blended) / denom
                p_team1 = _normal_cdf(z)
                p_team2 = 1.0 - p_team1

                overview_data.append({
                    "team1": format_team_name(team1_name),
                    "blended1": team1_blended,
                    "live1": team1_live if gw_is_live else None,
                    "orig1": team1_orig_proj,
                    "pct1": p_team1 * 100,
                    "pct2": p_team2 * 100,
                    "blended2": team2_blended,
                    "live2": team2_live if gw_is_live else None,
                    "orig2": team2_orig_proj,
                    "team2": format_team_name(team2_name),
                })
            except Exception:
                continue

    if not overview_data:
        st.warning("Could not calculate projections for fixtures.")
        return

    # Build HTML table with fancy styling - different layout for live vs pre-match
    if gw_is_live:
        # Live layout: Live Score | Updated Proj | Win % | vs | Win % | Updated Proj | Live Score
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: transparent; }
            .fixtures-table { width: 100%; border-collapse: separate; border-spacing: 0; margin: 10px 0; }
            .fixtures-table th {
                background: linear-gradient(135deg, #37003c 0%, #5a0050 100%);
                color: white; padding: 12px 8px; text-align: center;
                font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
            }
            .fixtures-table th:first-child { border-radius: 10px 0 0 0; }
            .fixtures-table th:last-child { border-radius: 0 10px 0 0; }
            .fixtures-table td { padding: 12px 8px; text-align: center; border-bottom: 1px solid #e0e0e0; font-size: 13px; vertical-align: middle; }
            .fixtures-table tr:last-child td:first-child { border-radius: 0 0 0 10px; }
            .fixtures-table tr:last-child td:last-child { border-radius: 0 0 10px 0; }
            .fixtures-table tr:hover td { background-color: #f8f4f9; }
            .team-name { font-weight: 600; color: #1a1a2e; min-width: 120px; }
            .team-left { text-align: right !important; padding-right: 12px !important; }
            .team-right { text-align: left !important; padding-left: 12px !important; }
            .score-cell { min-width: 90px; }
            .live-score { font-size: 22px; font-weight: 700; color: #e74c3c; }
            .blended-score { font-size: 13px; color: #555; margin-top: 2px; }
            .orig-proj { font-size: 10px; color: #999; margin-top: 1px; }
            .perf-up { color: #28a745; }
            .perf-down { color: #dc3545; }
            .win-pct { font-weight: 700; font-size: 14px; min-width: 55px; padding: 6px 10px !important; border-radius: 6px; }
            .vs-cell { color: #888; font-weight: 500; font-size: 12px; min-width: 30px; }
        </style>
        </head>
        <body>
        <table class="fixtures-table">
        <thead>
            <tr>
                <th>Team</th>
                <th>Live / Proj</th>
                <th>Win %</th>
                <th></th>
                <th>Win %</th>
                <th>Live / Proj</th>
                <th>Team</th>
            </tr>
        </thead>
        <tbody>
        """

        for row in overview_data:
            color1 = _get_win_pct_color(row["pct1"])
            color2 = _get_win_pct_color(row["pct2"])

            # Calculate performance vs original projection
            perf1 = row["live1"] - (row["orig1"] * (row["live1"] / row["blended1"])) if row["blended1"] > 0 else 0
            perf2 = row["live2"] - (row["orig2"] * (row["live2"] / row["blended2"])) if row["blended2"] > 0 else 0

            # Format: Live pts on top, blended proj below, original proj as reference
            diff1 = row["blended1"] - row["orig1"]
            diff2 = row["blended2"] - row["orig2"]
            diff1_class = "perf-up" if diff1 > 0 else "perf-down" if diff1 < 0 else ""
            diff2_class = "perf-up" if diff2 > 0 else "perf-down" if diff2 < 0 else ""
            diff1_sign = "+" if diff1 > 0 else ""
            diff2_sign = "+" if diff2 > 0 else ""

            score1_html = f'''
                <div class="live-score">{row["live1"]:.0f}</div>
                <div class="blended-score">→ {row["blended1"]:.1f} proj</div>
                <div class="orig-proj">orig: {row["orig1"]:.1f} <span class="{diff1_class}">({diff1_sign}{diff1:.1f})</span></div>
            '''
            score2_html = f'''
                <div class="live-score">{row["live2"]:.0f}</div>
                <div class="blended-score">→ {row["blended2"]:.1f} proj</div>
                <div class="orig-proj">orig: {row["orig2"]:.1f} <span class="{diff2_class}">({diff2_sign}{diff2:.1f})</span></div>
            '''

            html += f"""
            <tr>
                <td class="team-name team-left">{row["team1"]}</td>
                <td class="score-cell">{score1_html}</td>
                <td class="win-pct" style="background: {color1}; color: white;">{row["pct1"]:.0f}%</td>
                <td class="vs-cell">vs</td>
                <td class="win-pct" style="background: {color2}; color: white;">{row["pct2"]:.0f}%</td>
                <td class="score-cell">{score2_html}</td>
                <td class="team-name team-right">{row["team2"]}</td>
            </tr>
            """
    else:
        # Pre-match layout: Team | Proj | Win % | vs | Win % | Proj | Team
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: transparent; }
            .fixtures-table { width: 100%; border-collapse: separate; border-spacing: 0; margin: 10px 0; }
            .fixtures-table th {
                background: linear-gradient(135deg, #37003c 0%, #5a0050 100%);
                color: white; padding: 14px 12px; text-align: center;
                font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;
            }
            .fixtures-table th:first-child { border-radius: 10px 0 0 0; }
            .fixtures-table th:last-child { border-radius: 0 10px 0 0; }
            .fixtures-table td { padding: 14px 12px; text-align: center; border-bottom: 1px solid #e0e0e0; font-size: 14px; }
            .fixtures-table tr:last-child td:first-child { border-radius: 0 0 0 10px; }
            .fixtures-table tr:last-child td:last-child { border-radius: 0 0 10px 0; }
            .fixtures-table tr:hover td { background-color: #f8f4f9; }
            .team-name { font-weight: 600; color: #1a1a2e; min-width: 140px; }
            .team-left { text-align: right !important; padding-right: 20px !important; }
            .team-right { text-align: left !important; padding-left: 20px !important; }
            .proj-score { font-weight: 500; color: #444; min-width: 55px; }
            .win-pct { font-weight: 700; font-size: 15px; min-width: 65px; padding: 8px 12px !important; border-radius: 6px; }
            .vs-cell { color: #888; font-weight: 500; font-size: 12px; min-width: 40px; }
        </style>
        </head>
        <body>
        <table class="fixtures-table">
        <thead>
            <tr>
                <th>Team</th>
                <th>Proj</th>
                <th>Win %</th>
                <th></th>
                <th>Win %</th>
                <th>Proj</th>
                <th>Team</th>
            </tr>
        </thead>
        <tbody>
        """

        for row in overview_data:
            color1 = _get_win_pct_color(row["pct1"])
            color2 = _get_win_pct_color(row["pct2"])

            html += f"""
            <tr>
                <td class="team-name team-left">{row["team1"]}</td>
                <td class="proj-score">{row["blended1"]:.1f}</td>
                <td class="win-pct" style="background: {color1}; color: white;">{row["pct1"]:.0f}%</td>
                <td class="vs-cell">vs</td>
                <td class="win-pct" style="background: {color2}; color: white;">{row["pct2"]:.0f}%</td>
                <td class="proj-score">{row["blended2"]:.1f}</td>
                <td class="team-name team-right">{row["team2"]}</td>
            </tr>
            """

    html += """
    </tbody>
    </table>
    </body>
    </html>
    """

    # Calculate height based on number of fixtures (taller rows for live view with 3 lines of text)
    row_height = 90 if gw_is_live else 52
    table_height = 70 + (len(overview_data) * row_height)
    components.html(html, height=table_height, scrolling=False)


def show_fixtures_page():
    st.title("Upcoming Fixtures & Projections")

    current_gw = config.CURRENT_GAMEWEEK
    gw_is_live = is_gameweek_live(current_gw)

    # Header with refresh button and live indicator
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        if gw_is_live:
            st.subheader(f"🔴 LIVE - Gameweek {current_gw}")
        else:
            st.subheader(f"Gameweek {current_gw} Fixtures Overview")
    with col2:
        if gw_is_live:
            # Auto-refresh toggle for live games
            auto_refresh = st.checkbox("Auto", value=False, help="Auto-refresh every 60s")
            if auto_refresh:
                import time
                time.sleep(0.1)  # Small delay to prevent infinite loop
                st.rerun()
    with col3:
        if st.button("🔄", help="Refresh live data"):
            # Clear cached live stats
            get_live_gameweek_stats.clear()
            is_gameweek_live.clear()
            config.refresh_gameweek()
            st.rerun()

    # Get live stats if gameweek is live
    live_stats = get_live_gameweek_stats(current_gw) if gw_is_live else {}
    player_mapping = get_fpl_player_mapping() if gw_is_live else {}

    # Find the fixtures for the current gameweek
    gameweek_fixtures = get_gameweek_fixtures(config.FPL_DRAFT_LEAGUE_ID, current_gw)

    if not gameweek_fixtures:
        st.warning("No fixtures found for the current gameweek.")
        return

    # Pull FPL player projections from Rotowire
    fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    if fpl_player_projections is None or fpl_player_projections.empty:
        st.warning("Rotowire projections unavailable.")
        # Still show fixtures list
        for fixture in gameweek_fixtures:
            st.text(fixture)
        return

    # Get sigma for win probability calculations
    sigma, n_hist = _estimate_score_std(config.FPL_DRAFT_LEAGUE_ID)

    # Render the fixtures overview table (with live data if available)
    _render_fixtures_overview(gameweek_fixtures, config.FPL_DRAFT_LEAGUE_ID, fpl_player_projections, sigma,
                              live_stats=live_stats, player_mapping=player_mapping, gw_is_live=gw_is_live)

    if gw_is_live:
        st.caption("🔴 **LIVE**: Scores update as players finish. Projected points shown for players yet to play.")
    else:
        hist_note = f"σ≈{sigma:.2f} from {n_hist} historical scores" if n_hist > 0 else f"σ≈{sigma:.2f} (default)"
        st.caption(f"Win probability model: P(A>B) = Φ((μA−μB)/√(2σ²)). {hist_note}")

    # Divider before detailed view
    st.divider()

    # Detailed view section
    st.subheader("Detailed Match Analysis")

    # Create a dropdown to choose a fixture
    fixture_selection = st.selectbox("Select a fixture to analyze:", gameweek_fixtures)

    # Create the Streamlit visuals
    if fixture_selection:
        # Analyze fixture projections - use actual lineups for live gameweeks
        result = analyze_fixture_projections(fixture_selection, config.FPL_DRAFT_LEAGUE_ID, fpl_player_projections, use_actual_lineup=gw_is_live)

        if result is None:
            st.error(
                "**Could not analyze this fixture.** Player projections may be unavailable "
                "or team rosters could not be resolved. Try selecting a different fixture."
            )
            return

        team1_df, team2_df, team1_name, team2_name = result

        # Blend live points if gameweek is live
        if gw_is_live and live_stats:
            team1_df = _blend_live_with_projections(team1_df, live_stats, player_mapping)
            team2_df = _blend_live_with_projections(team2_df, live_stats, player_mapping)
            team1_score = team1_df['Blended_Points'].sum()
            team2_score = team2_df['Blended_Points'].sum()
            team1_live = team1_df['Live_Points'].sum()
            team2_live = team2_df['Live_Points'].sum()
        else:
            team1_score = team1_df['Points'].sum()
            team2_score = team2_df['Points'].sum()
            team1_live = None
            team2_live = None

        # --- Win Probability (Normal model) ---
        denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0
        z = (team1_score - team2_score) / denom
        p_team1 = _normal_cdf(z)

        st.subheader("Win Probability")
        _render_winprob_bar(format_team_name(team1_name), format_team_name(team2_name), p_team1)

        # Team Lineups section
        st.subheader("Team Lineups")

        # Create columns for side-by-side lineup display
        col1, col2 = st.columns(2)

        with col1:
            # Team 1 header with prominent score
            orig_proj1 = team1_df['Points'].sum()
            if gw_is_live and team1_live is not None:
                diff1 = team1_score - orig_proj1
                diff1_color = "green" if diff1 > 0 else "red" if diff1 < 0 else "gray"
                diff1_sign = "+" if diff1 > 0 else ""
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #37003c 0%, #5a0050 100%); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                    <div style="color: white; font-size: 14px; font-weight: 500; margin-bottom: 4px;">{format_team_name(team1_name)}</div>
                    <div style="display: flex; align-items: baseline; gap: 12px;">
                        <span style="color: #00ff87; font-size: 32px; font-weight: 700;">{team1_live:.0f}</span>
                        <span style="color: rgba(255,255,255,0.7); font-size: 16px;">→ {team1_score:.1f} proj</span>
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 4px;">
                        Pre-match: {orig_proj1:.1f} <span style="color: {'#00ff87' if diff1 > 0 else '#ff6b6b' if diff1 < 0 else 'rgba(255,255,255,0.6)'};">({diff1_sign}{diff1:.1f})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #37003c 0%, #5a0050 100%); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                    <div style="color: white; font-size: 14px; font-weight: 500; margin-bottom: 4px;">{format_team_name(team1_name)}</div>
                    <div style="color: #00ff87; font-size: 32px; font-weight: 700;">{team1_score:.1f}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px;">Projected Points</div>
                </div>
                """, unsafe_allow_html=True)

            _render_team_lineup(team1_df, team1_name, is_live=gw_is_live)

        with col2:
            # Team 2 header with prominent score
            orig_proj2 = team2_df['Points'].sum()
            if gw_is_live and team2_live is not None:
                diff2 = team2_score - orig_proj2
                diff2_color = "green" if diff2 > 0 else "red" if diff2 < 0 else "gray"
                diff2_sign = "+" if diff2 > 0 else ""
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #37003c 0%, #5a0050 100%); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                    <div style="color: white; font-size: 14px; font-weight: 500; margin-bottom: 4px;">{format_team_name(team2_name)}</div>
                    <div style="display: flex; align-items: baseline; gap: 12px;">
                        <span style="color: #00ff87; font-size: 32px; font-weight: 700;">{team2_live:.0f}</span>
                        <span style="color: rgba(255,255,255,0.7); font-size: 16px;">→ {team2_score:.1f} proj</span>
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 4px;">
                        Pre-match: {orig_proj2:.1f} <span style="color: {'#00ff87' if diff2 > 0 else '#ff6b6b' if diff2 < 0 else 'rgba(255,255,255,0.6)'};">({diff2_sign}{diff2:.1f})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #37003c 0%, #5a0050 100%); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                    <div style="color: white; font-size: 14px; font-weight: 500; margin-bottom: 4px;">{format_team_name(team2_name)}</div>
                    <div style="color: #00ff87; font-size: 32px; font-weight: 700;">{team2_score:.1f}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px;">Projected Points</div>
                </div>
                """, unsafe_allow_html=True)

            _render_team_lineup(team2_df, team2_name, is_live=gw_is_live)

        # --- Head-to-Head History (below lineups) ---
        team1_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, team1_name)
        team2_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, team2_name)

        if team1_id and team2_id:
            h2h = get_draft_h2h_record(config.FPL_DRAFT_LEAGUE_ID, team1_id, team2_id)

            if h2h["wins"] + h2h["draws"] + h2h["losses"] > 0:
                st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
                st.subheader("Head-to-Head History")

                # Styled H2H record display
                total_matches = h2h["wins"] + h2h["draws"] + h2h["losses"]
                t1_pct = (h2h["wins"] / total_matches * 100) if total_matches > 0 else 0
                t2_pct = (h2h["losses"] / total_matches * 100) if total_matches > 0 else 0

                st.markdown(f"""
                <div style="background: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 36px; font-weight: 700; color: #28a745;">{h2h["wins"]}</div>
                            <div style="font-size: 12px; color: #666; text-transform: uppercase;">{format_team_name(team1_name)} Wins</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 36px; font-weight: 700; color: #6c757d;">{h2h["draws"]}</div>
                            <div style="font-size: 12px; color: #666; text-transform: uppercase;">Draws</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 36px; font-weight: 700; color: #dc3545;">{h2h["losses"]}</div>
                            <div style="font-size: 12px; color: #666; text-transform: uppercase;">{format_team_name(team2_name)} Wins</div>
                        </div>
                    </div>
                    <div style="height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; display: flex;">
                        <div style="width: {t1_pct}%; background: #28a745;"></div>
                        <div style="width: {100 - t1_pct - t2_pct}%; background: #6c757d;"></div>
                        <div style="width: {t2_pct}%; background: #dc3545;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show recent matchups if available
                if h2h["matches"]:
                    with st.expander("View Past Matchups"):
                        match_data = []
                        for m in reversed(h2h["matches"]):  # Most recent first
                            match_data.append({
                                "Gameweek": f"GW{m['gameweek']}",
                                format_team_name(team1_name): m["my_pts"],
                                format_team_name(team2_name): m["opp_pts"],
                                "Result": m["outcome"]
                            })
                        st.dataframe(
                            pd.DataFrame(match_data),
                            use_container_width=True,
                            hide_index=True
                        )