import config
import math
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional
from scripts.common.analytics import simulate_auto_subs, blend_fixture_projections, numeric_col
from scripts.common.error_helpers import get_logger
from scripts.common.scraping import get_ffp_feed, render_ffp_status
from scripts.common.utils import (
    find_optimal_lineup, format_team_name, get_current_gameweek, get_gameweek_fixtures,
    get_team_id_by_name, get_rotowire_player_projections, get_team_composition_for_gameweek,
    merge_fpl_players_and_projections, normalize_apostrophes, get_historical_team_scores,
    get_draft_h2h_record, get_live_gameweek_stats, is_gameweek_live, get_fpl_player_mapping,
    get_team_actual_lineup, get_gw_finished_teams, get_classic_bootstrap_static,
)
from scripts.common.fixture_helpers import (
    compute_key_differentials, live_player_status, render_key_differentials,
)
from scripts.common.styled_tables import render_styled_table

_logger = get_logger("fpl_app.draft.fixture_projections")


def _blend_live_with_projections(team_df: pd.DataFrame, live_stats: dict, player_mapping: dict) -> pd.DataFrame:
    """
    Blend live points with projections for players who haven't played yet.

    For each player:
    - If they have played (minutes > 0): use actual points
    - If their match is over and they never came on: use their actual (0) points --
      an unused sub cannot score any more this week, so holding his projection in
      the team total quietly overstates the score for the rest of the gameweek
    - Otherwise: use projected points

    Matching prefers 'Player_ID' (the FPL element id) when the frame carries it.
    Name matching is a fallback only: this frame's names come from the projection
    source, so "Igor Thiago" never reached the bootstrap's "Igor Thiago Nascimento
    Rodrigues" and he showed as Upcoming through a full 90 minutes.

    Returns DataFrame with additional columns:
    - 'Live_Points': actual points scored (0 if not played)
    - 'Has_Played': bool indicating if player has played
    - 'Fixture_Started' / 'Fixture_Finished': state of the player's own match
    - 'Blended_Points': live points if their match is done, projected if not
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
    result['Fixture_Started'] = False
    result['Fixture_Finished'] = False
    result['Blended_Points'] = result['Points'].fillna(0)

    id_col = 'Player_ID' if 'Player_ID' in result.columns else (
        'element_id' if 'element_id' in result.columns else None
    )

    # Get team info from DataFrame if available
    team_col = 'Team' if 'Team' in result.columns else None

    # Player name is in the index for this DataFrame
    for idx in result.index:
        player_name = idx  # Index is the player name
        player_team = result.at[idx, team_col] if team_col else None

        # Try multiple matching strategies
        element_id = None

        # Strategy 0: element id carried through the merge (exact, never ambiguous)
        if id_col is not None:
            raw_id = result.at[idx, id_col]
            if pd.notna(raw_id):
                try:
                    element_id = int(raw_id)
                except (TypeError, ValueError):
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
            has_played = bool(stats.get('has_played', False))
            fixture_finished = bool(stats.get('fixture_finished', False))
            result.at[idx, 'Live_Points'] = stats.get('points', 0)
            result.at[idx, 'Has_Played'] = has_played
            result.at[idx, 'Fixture_Started'] = bool(stats.get('fixture_started', False))
            result.at[idx, 'Fixture_Finished'] = fixture_finished

            if has_played or fixture_finished:
                result.at[idx, 'Blended_Points'] = stats.get('points', 0)

    return result


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# League-wide prior for the spread of single-gameweek team scores, used whenever
# real history can't produce a usable estimate. Never let sigma reach 0: the win
# probability is Phi(diff / sqrt(2*sigma^2)), so a near-zero sigma turns a 1-point
# projection edge into a ~100% win call.
_DEFAULT_SCORE_STD = 15.0


def _weekly_score_std(df: pd.DataFrame) -> Optional[float]:
    """Std of single-gameweek team scores from a historical scores frame.

    Prefers the per-gameweek column ('points'/'score') over the cumulative
    season running total ('total_points') -- the model wants the spread of one
    week's scores, and the std of a monotonically increasing season total is a
    completely different (and much larger) number.

    Unplayed gameweeks are dropped: the Draft API returns a full 38-gameweek
    grid from day one, so preseason every row is 0 and the std collapses to 0.
    A gameweek is treated as unplayed only when *every* team scored 0 -- an
    individual 0 is left in, since it could in principle be a real result.

    Returns None when no usable estimate can be made.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    col = next((c for c in ('points', 'score', 'total_points') if c in df.columns), None)
    if col is None:
        return None
    if col == 'total_points' and 'event' in df.columns:
        # Cumulative totals -- recover per-gameweek scores by differencing per team.
        key = 'entry_id' if 'entry_id' in df.columns else ('entry_name' if 'entry_name' in df.columns else None)
        if key:
            tmp = df[[key, 'event', 'total_points']].copy()
            tmp['total_points'] = pd.to_numeric(tmp['total_points'], errors='coerce')
            tmp = tmp.sort_values([key, 'event'])
            tmp['__weekly'] = tmp.groupby(key)['total_points'].diff().fillna(tmp['total_points'])
            df, col = tmp, '__weekly'

    scores = df[[col] + (['event'] if 'event' in df.columns else [])].copy()
    scores[col] = pd.to_numeric(scores[col], errors='coerce')
    scores = scores.dropna(subset=[col])

    # Drop gameweeks nobody has played yet (every team on 0).
    if 'event' in scores.columns:
        played = scores.groupby('event')[col].transform(lambda g: (g != 0).any())
        scores = scores[played]
    elif not (scores[col] != 0).any():
        return None

    s = scores[col]
    if len(s) < 2:
        return None
    std = float(s.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    return std


def _estimate_score_std(league_id: int) -> tuple[float, int]:
    """
    Returns (std, n) for historical single-team weekly scores if available.
    Tries: scripts.common.utils.get_historical_team_scores(league_id).
    Fallback: CSV path in config.HISTORICAL_SCORES_CSV (or 'data/historical_team_scores.csv').
    Final fallback: (_DEFAULT_SCORE_STD, 0) — a reasonable league-wide prior.

    n is 0 whenever the returned std is the prior rather than a real estimate, so
    callers can label it as a default in the UI.
    """
    # Try utils function if it exists
    try:
        hist = get_historical_team_scores(league_id)
    except Exception:
        hist = None
    std = _weekly_score_std(hist)
    if std is not None:
        return std, int(len(hist))

    # Try CSV from config or default path
    try:
        csv_path = getattr(config, 'HISTORICAL_SCORES_CSV', 'data/historical_team_scores.csv')
        df = pd.read_csv(csv_path)
        std = _weekly_score_std(df)
        if std is not None:
            return std, int(len(df))
    except Exception:
        pass
    return _DEFAULT_SCORE_STD, 0  # conservative default if nothing available


def _winprob_denom(sigma: float) -> float:
    """sqrt(2*sigma^2) for P(A>B) = Phi((muA - muB) / denom).

    Falls back to the league prior rather than 1.0 when sigma is missing or
    degenerate -- a denominator of 1.0 makes a 1-point projection edge read as
    an ~85% win, which is what this guard exists to prevent.
    """
    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        sigma = _DEFAULT_SCORE_STD
    return math.sqrt(2.0 * (sigma ** 2))


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
        .player-card.dnp {{ border-left-color: #b0b3b8; background: #f1f2f3; opacity: 0.75; }}
        .player-card.upcoming {{ border-left-color: #6c757d; }}
        .player-info {{ flex: 1; min-width: 0; }}
        .player-name {{ font-weight: 600; font-size: 13px; line-height: 18px; color: #1a1a2e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .player-team {{ font-size: 10px; line-height: 14px; color: #888; text-transform: uppercase; }}
        .player-matchup {{ font-size: 10px; line-height: 14px; color: #666; }}
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
        .status-dnp {{ background: #e6d9d9; color: #7a1f1f; }}
        .status-live {{ background: #fff3cd; color: #7a5b00; }}
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

        for player_name, row in pos_players.iterrows():
            team = row.get('Team', '')
            matchup = row.get('Matchup', '')
            proj_pts = pd.to_numeric(row.get('Points', 0), errors='coerce')
            proj_pts = 0.0 if pd.isna(proj_pts) else float(proj_pts)
            display_pts = pd.to_numeric(row.get('Proj_Blended'), errors='coerce')
            display_pts = proj_pts if pd.isna(display_pts) else float(display_pts)

            if is_live:
                live_pts = row.get('Live_Points', 0) or 0
                status = live_player_status(
                    bool(row.get('Has_Played', False)),
                    bool(row.get('Fixture_Finished', False)),
                    bool(row.get('Fixture_Started', False)),
                )

                if status == 'played':
                    # Player has finished - show actual vs projected
                    diff = live_pts - display_pts
                    diff_sign = "+" if diff > 0 else ""
                    diff_class = "perf-up" if diff > 0 else "perf-down" if diff < 0 else ""

                    points_html = f"""
                        <div class="live-pts">{live_pts:.0f}</div>
                        <div class="perf-indicator {diff_class}">proj: {display_pts:.1f} ({diff_sign}{diff:.1f})</div>
                    """
                    card_class = "player-card played"
                    status_html = '<span class="status-badge status-played">Played</span>'
                elif status == 'dnp':
                    # Match over, never came on - the projection can no longer arrive
                    points_html = f"""
                        <div class="live-pts">0</div>
                        <div class="perf-indicator perf-down">proj: {display_pts:.1f} (-{display_pts:.1f})</div>
                    """
                    card_class = "player-card dnp"
                    status_html = '<span class="status-badge status-dnp">Did not play</span>'
                else:
                    # Yet to play - show blended projection (Rotowire + FFP) when available
                    points_html = f"""
                        <div class="proj-only">{display_pts:.1f}</div>
                        <div class="proj-pts">projected</div>
                    """
                    if status == 'live':
                        card_class = "player-card upcoming"
                        status_html = '<span class="status-badge status-live">In play</span>'
                    else:
                        card_class = "player-card upcoming"
                        status_html = '<span class="status-badge status-upcoming">Upcoming</span>'
            else:
                # Pre-match: show blended projection (Rotowire + FFP) when available
                points_html = f'<div class="proj-only">{display_pts:.1f}</div>'
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

    # Calculate total height based on player count. Every term is measured from the
    # CSS above -- the iframe does not scroll, so an underestimate silently clips the
    # last card off the bottom of the lineup.
    #   card    = 8+8 padding + 18 name + 14 team + 14 matchup + 4 margin = 66
    #   heading = 6+6 padding + 14 text + 6 margin                       = 32
    #   group   = 12 margin-bottom
    player_count = len(team_df)
    pos_groups = team_df['Position'].nunique() if 'Position' in team_df.columns else 4
    height = 8 + (player_count * 66) + (pos_groups * 44)
    components.html(html, height=height, scrolling=False)


def _render_draft_bench_section(bench_df: pd.DataFrame, is_live: bool = False):
    """
    Render bench slots below the starting XI. GK is shown first (fixed sub), then the
    3 outfield players sorted by projected points descending (optimal auto-sub order).
    """
    if bench_df.empty:
        return

    bench_df = bench_df.copy()
    bench_df['Points'] = numeric_col(bench_df, 'Points', 0.0)
    bench_df['Proj_Blended'] = numeric_col(bench_df, 'Proj_Blended', float('nan')).fillna(bench_df['Points'])

    sort_col = 'Proj_Blended' if bench_df['Proj_Blended'].gt(0).any() else 'Points'

    # Split GK and outfield
    pos_col = 'Position' if 'Position' in bench_df.columns else None
    if pos_col:
        gk_rows = bench_df[bench_df[pos_col] == 'G']
        out_rows = bench_df[bench_df[pos_col] != 'G'].sort_values(sort_col, ascending=False)
        ordered = pd.concat([gk_rows, out_rows])
        labels = ['GK Sub'] + [f'{i+1}{"st" if i==0 else "nd" if i==1 else "rd"} Sub' for i in range(len(out_rows))]
    else:
        ordered = bench_df
        labels = ['GK Sub', '1st Sub', '2nd Sub', '3rd Sub']

    html = """
    <style>
        .bench-section { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin-top: 8px; }
        .bench-header {
            font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
            padding: 6px 10px; border-radius: 4px; margin-bottom: 6px; color: #555;
            background: #e8e8e8;
        }
        .bench-card {
            display: flex; align-items: center; justify-content: space-between;
            background: #f4f4f4; border-radius: 6px; padding: 7px 12px;
            margin-bottom: 4px; border-left: 3px dashed #bbb;
        }
        .bench-card.played { border-left-color: #28a745; background: #f0fff4; }
        .bench-card.dnp { border-left-color: #b0b3b8; background: #f1f2f3; opacity: 0.75; }
        .sub-label { font-size: 9px; font-weight: 700; color: #888; text-transform: uppercase;
                     min-width: 48px; }
        .bench-player-info { flex: 1; min-width: 0; }
        .bench-player-name { font-weight: 500; font-size: 12px; line-height: 17px; color: #333;
                             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .bench-player-meta { font-size: 10px; line-height: 14px; color: #999; text-transform: uppercase; }
        .bench-pts { text-align: right; min-width: 55px; }
        .bench-live-pts { font-size: 15px; font-weight: 700; color: #28a745; }
        .bench-proj-pts { font-size: 13px; font-weight: 500; color: #777; }
        .bench-proj-label { font-size: 10px; color: #aaa; }
    </style>
    <div class="bench-section">
        <div class="bench-header">Bench</div>
    """

    for i, (player_name, row) in enumerate(ordered.iterrows()):
        label = labels[i] if i < len(labels) else f'{i}th Sub'
        team = row.get('Team', '')
        position = row.get('Position', '')
        proj_pts = pd.to_numeric(row.get('Points', 0), errors='coerce')
        proj_pts = 0.0 if pd.isna(proj_pts) else float(proj_pts)
        display_pts = pd.to_numeric(row.get('Proj_Blended'), errors='coerce')
        display_pts = proj_pts if pd.isna(display_pts) else float(display_pts)
        meta = f"{team} · {position}" if team and position else team or position

        if is_live and 'Has_Played' in row.index:
            live_pts = row.get('Live_Points', 0) or 0
            status = live_player_status(
                bool(row.get('Has_Played', False)),
                bool(row.get('Fixture_Finished', False)),
                bool(row.get('Fixture_Started', False)),
            )
            if status == 'played':
                points_html = f'<div class="bench-live-pts">{live_pts:.0f}</div><div class="bench-proj-label">proj: {display_pts:.1f}</div>'
                card_class = "bench-card played"
            elif status == 'dnp':
                points_html = f'<div class="bench-live-pts">0</div><div class="bench-proj-label">did not play</div>'
                card_class = "bench-card dnp"
            else:
                points_html = f'<div class="bench-proj-pts">{display_pts:.1f}</div><div class="bench-proj-label">projected</div>'
                card_class = "bench-card"
        else:
            points_html = f'<div class="bench-proj-pts">{display_pts:.1f}</div>'
            card_class = "bench-card"

        html += f"""
        <div class="{card_class}">
            <div class="sub-label">{label}</div>
            <div class="bench-player-info">
                <div class="bench-player-name">{player_name}</div>
                <div class="bench-player-meta">{meta}</div>
            </div>
            <div class="bench-pts">{points_html}</div>
        </div>
        """

    html += "</div>"
    # 7+7 padding + 17 name + 14 meta + 4 margin = 49 per card, + header (32) + margin (8)
    height = 40 + (len(ordered) * 49) + 8
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

def analyze_fixture_projections(fixture, league_id, projections_df, use_actual_lineup: bool = False,
                                live_stats: dict = None, ffp_df=None):
    """
    Returns two DataFrames representing lineups and points for each team in a fixture,
    sorted by position (GK, DEF, MID, FWD) and then by descending projected points within each position.

    Parameters:
    - fixture (str): The selected fixture, formatted as "Team1 (Player1) vs Team2 (Player2)".
    - league_id (int): The ID of the FPL Draft league.
    - projections_df (DataFrame): DataFrame containing player projections from Rotowire.
    - use_actual_lineup (bool): If True, use the manager's actual starting 11 picks.
                                If False, calculate the optimal lineup by projections.
    - live_stats (dict): Live gameweek stats for auto-sub simulation. If provided with
                         use_actual_lineup=True, auto-subs are applied before filtering to starters.

    Returns:
    - Tuple: (team1_df, team2_df, team1_name, team2_name, bench1_df, bench2_df) or
             (team1_df, team2_df, team1_name, team2_name, bench1_df, bench2_df, subs1, subs2) when live_stats provided.
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

    subs1, subs2 = [], []
    bench1_df = bench2_df = pd.DataFrame()

    if use_actual_lineup:
        # Use actual picks from the FPL Draft API (full 15-player squad)
        team1_actual = get_team_actual_lineup(team1_id, gameweek)
        team2_actual = get_team_actual_lineup(team2_id, gameweek)

        # Apply auto-subs if live stats provided
        if live_stats and not team1_actual.empty and not team2_actual.empty:
            # Rename Player_ID to element_id for simulate_auto_subs compatibility
            for df in [team1_actual, team2_actual]:
                if 'Player_ID' in df.columns and 'element_id' not in df.columns:
                    df['element_id'] = df['Player_ID']

            finished_teams = get_gw_finished_teams(gameweek)
            if finished_teams:
                bootstrap = get_classic_bootstrap_static()
                if bootstrap:
                    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
                    element_to_team = {eid: p.get("team") for eid, p in elements.items()}

                    team1_actual, subs1 = simulate_auto_subs(
                        team1_actual, live_stats, element_to_team, finished_teams
                    )
                    team2_actual, subs2 = simulate_auto_subs(
                        team2_actual, live_stats, element_to_team, finished_teams
                    )

        # Filter to starters and extract bench (squad_position 12-15)
        if 'squad_position' in team1_actual.columns:
            team1_starters = team1_actual[team1_actual['squad_position'] <= 11].copy()
            team2_starters = team2_actual[team2_actual['squad_position'] <= 11].copy()
            team1_bench_raw = team1_actual[team1_actual['squad_position'] > 11].copy()
            team2_bench_raw = team2_actual[team2_actual['squad_position'] > 11].copy()
        else:
            team1_starters = team1_actual[team1_actual['Is_Starter'] == True].copy()
            team2_starters = team2_actual[team2_actual['Is_Starter'] == True].copy()
            team1_bench_raw = team1_actual[team1_actual['Is_Starter'] == False].copy()
            team2_bench_raw = team2_actual[team2_actual['Is_Starter'] == False].copy()

        # Merge starters with projections. Carry Player_ID through: the merge takes
        # matched names from the projection source, and live stats are keyed on the
        # FPL element id -- without it the live lookup falls back to name matching.
        team1_df = merge_fpl_players_and_projections(
            team1_starters[['Player', 'Team', 'Position', 'Player_ID']],
            projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']],
            carry_cols=['Player_ID'],
        )
        team2_df = merge_fpl_players_and_projections(
            team2_starters[['Player', 'Team', 'Position', 'Player_ID']],
            projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']],
            carry_cols=['Player_ID'],
        )

        # Blend starters with FFP
        team1_df = blend_fixture_projections(team1_df, ffp_df)
        team2_df = blend_fixture_projections(team2_df, ffp_df)

        # Merge bench players with projections
        if not team1_bench_raw.empty:
            _bench1_carry = [c for c in ('squad_position', 'Player_ID') if c in team1_bench_raw.columns]
            bench1_merged = merge_fpl_players_and_projections(
                team1_bench_raw[['Player', 'Team', 'Position'] + _bench1_carry],
                projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']],
                carry_cols=_bench1_carry,
            )
            bench1_merged = blend_fixture_projections(bench1_merged, ffp_df)
            bench1_df = bench1_merged.set_index('Player') if 'Player' in bench1_merged.columns else bench1_merged
        if not team2_bench_raw.empty:
            _bench2_carry = [c for c in ('squad_position', 'Player_ID') if c in team2_bench_raw.columns]
            bench2_merged = merge_fpl_players_and_projections(
                team2_bench_raw[['Player', 'Team', 'Position'] + _bench2_carry],
                projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']],
                carry_cols=_bench2_carry,
            )
            bench2_merged = blend_fixture_projections(bench2_merged, ffp_df)
            bench2_df = bench2_merged.set_index('Player') if 'Player' in bench2_merged.columns else bench2_merged
    else:
        # Use optimal lineup calculation (original behavior)
        team1_composition = get_team_composition_for_gameweek(league_id, team1_id, gameweek)
        team2_composition = get_team_composition_for_gameweek(league_id, team2_id, gameweek)

        # Merge full squad with projections, then split into optimal XI + bench
        team1_full = merge_fpl_players_and_projections(
            team1_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
        )
        team2_full = merge_fpl_players_and_projections(
            team2_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
        )
        team1_full = blend_fixture_projections(team1_full, ffp_df)
        team2_full = blend_fixture_projections(team2_full, ffp_df)
        team1_df = team1_full.copy()
        team2_df = team2_full.copy()

    # Check if 'Points' column exists
    if 'Points' not in team1_df or 'Points' not in team2_df:
        print("Error: 'Points' column not found in one or both dataframes.")
        return None

    # Fill NaN values in 'Points' column with 0.0
    team1_df['Points'] = pd.to_numeric(team1_df['Points'], errors='coerce').fillna(0.0)
    team2_df['Points'] = pd.to_numeric(team2_df['Points'], errors='coerce').fillna(0.0)

    if not use_actual_lineup:
        # Find the optimal lineup (top 11 players) for each team. Rank on the same
        # column the team total is reported on, otherwise the XI is chosen on raw
        # Rotowire points while the score shown comes from the blend.
        _xi_col = 'Proj_Blended' if 'Proj_Blended' in team1_full.columns else 'Points'
        team1_df = find_optimal_lineup(team1_df, points_col=_xi_col)
        team2_df = find_optimal_lineup(team2_df, points_col=_xi_col)

        # Bench = players not in the optimal XI, sorted by blended proj pts desc (optimal auto-sub order)
        _bench_sort = 'Proj_Blended' if 'Proj_Blended' in team1_full.columns else 'Points'
        if 'Player' in team1_df.columns and 'Player' in team1_full.columns:
            selected1 = set(team1_df['Player'].tolist())
            bench1_raw = team1_full[~team1_full['Player'].isin(selected1)].sort_values(_bench_sort, ascending=False)
            bench1_df = bench1_raw.set_index('Player')
        if 'Player' in team2_df.columns and 'Player' in team2_full.columns:
            selected2 = set(team2_df['Player'].tolist())
            bench2_raw = team2_full[~team2_full['Player'].isin(selected2)].sort_values(_bench_sort, ascending=False)
            bench2_df = bench2_raw.set_index('Player')

    # Define the position order for sorting; prefer Blended_Points within each position
    position_order = ['G', 'D', 'M', 'F']
    _sort_col = 'Proj_Blended' if 'Proj_Blended' in team1_df.columns else 'Points'
    for df in [team1_df, team2_df]:
        df['Position'] = pd.Categorical(df['Position'], categories=position_order, ordered=True)
        df.sort_values(by=['Position', _sort_col], ascending=[True, False], inplace=True)

    # Select the final columns to use (include blend columns when present).
    # Player_ID rides along so the live blend can join on the element id.
    _extra = [c for c in ('Proj_Blended', '_proj_source', 'Player_ID') if c in team1_df.columns]
    team1_df = team1_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank'] + _extra]
    team2_df = team2_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank'] + _extra]

    # Format team DataFrames to use player names as the index
    team1_df.set_index('Player', inplace=True)
    team2_df.set_index('Player', inplace=True)

    # Return the final DataFrames, team names, bench, and subs (if live)
    if live_stats is not None:
        return team1_df, team2_df, team1_name, team2_name, bench1_df, bench2_df, subs1, subs2
    return team1_df, team2_df, team1_name, team2_name, bench1_df, bench2_df

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
                              live_stats: dict = None, player_mapping: dict = None, gw_is_live: bool = False,
                              ffp_df: pd.DataFrame = None):
    """
    Render an overview table showing all fixtures with projected scores and win probabilities.
    If gw_is_live, blends actual points with projections for remaining players.

    ffp_df must be the same frame the Detailed Match Analysis below uses -- without it
    this table falls back to raw Rotowire points and reports a different projected score
    (and therefore a different win probability) for the very same fixture.
    """
    if not fixtures:
        return

    live_stats = live_stats or {}
    player_mapping = player_mapping or {}

    overview_data = []
    denom = _winprob_denom(sigma)

    spinner_msg = "Calculating live scores..." if gw_is_live else "Calculating projections for all fixtures..."
    with st.spinner(spinner_msg):
        for fixture in fixtures:
            try:
                # Use actual lineups for live gameweeks, optimal projections otherwise
                result = analyze_fixture_projections(
                    fixture, league_id, projections_df,
                    use_actual_lineup=gw_is_live,
                    live_stats=live_stats if gw_is_live else None,
                    ffp_df=ffp_df,
                )
                if result is None:
                    continue

                if gw_is_live and live_stats:
                    team1_df, team2_df, team1_name, team2_name, _, _, _, _ = result
                else:
                    team1_df, team2_df, team1_name, team2_name, _, _ = result

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
                    # Same column precedence as the Detailed Match Analysis below, so the
                    # two sections can never disagree about a fixture's projected score.
                    _proj_col = 'Proj_Blended' if 'Proj_Blended' in team1_df.columns else 'Points'
                    team1_blended = team1_df[_proj_col].sum()
                    team2_blended = team2_df[_proj_col].sum()
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
                _logger.warning("Fixtures overview: skipping fixture %r", fixture, exc_info=True)
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

    # Pull FPL player projections from Rotowire and FFP (cached, zero extra cost)
    fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)
    ffp_feed_result = get_ffp_feed()
    ffp_df = ffp_feed_result.df
    render_ffp_status(ffp_feed_result, config.CURRENT_GAMEWEEK)

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
                              live_stats=live_stats, player_mapping=player_mapping, gw_is_live=gw_is_live,
                              ffp_df=ffp_df)

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
        result = analyze_fixture_projections(
            fixture_selection, config.FPL_DRAFT_LEAGUE_ID, fpl_player_projections,
            use_actual_lineup=gw_is_live,
            live_stats=live_stats if gw_is_live else None,
            ffp_df=ffp_df,
        )

        if result is None:
            st.error(
                "**Could not analyze this fixture.** Player projections may be unavailable "
                "or team rosters could not be resolved. Try selecting a different fixture."
            )
            return

        bench1_df = bench2_df = pd.DataFrame()
        subs1, subs2 = [], []
        if gw_is_live and live_stats:
            team1_df, team2_df, team1_name, team2_name, bench1_df, bench2_df, subs1, subs2 = result
        else:
            team1_df, team2_df, team1_name, team2_name, bench1_df, bench2_df = result

        # Blend live points if gameweek is live
        if gw_is_live and live_stats:
            team1_df = _blend_live_with_projections(team1_df, live_stats, player_mapping)
            team2_df = _blend_live_with_projections(team2_df, live_stats, player_mapping)
            team1_score = team1_df['Blended_Points'].sum()
            team2_score = team2_df['Blended_Points'].sum()
            team1_live = team1_df['Live_Points'].sum()
            team2_live = team2_df['Live_Points'].sum()
        else:
            _use_blended = 'Proj_Blended' in team1_df.columns
            team1_score = team1_df['Proj_Blended'].sum() if _use_blended else team1_df['Points'].sum()
            team2_score = team2_df['Proj_Blended'].sum() if _use_blended else team2_df['Points'].sum()
            team1_live = None
            team2_live = None

        # --- Win Probability (Normal model) ---
        denom = _winprob_denom(sigma)
        z = (team1_score - team2_score) / denom
        p_team1 = _normal_cdf(z)

        st.subheader("Win Probability")
        _render_winprob_bar(format_team_name(team1_name), format_team_name(team2_name), p_team1)

        # Auto-sub info banners
        if subs1 or subs2:
            sub_msgs = []
            for out_name, in_name in subs1:
                sub_msgs.append(f"{format_team_name(team1_name)}: {out_name} -> {in_name}")
            for out_name, in_name in subs2:
                sub_msgs.append(f"{format_team_name(team2_name)}: {out_name} -> {in_name}")
            st.info("**Auto-subs:** " + " | ".join(sub_msgs))

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
                _proj_label1 = "Projected Points (Rotowire + FFP)" if _use_blended else "Projected Points (Rotowire)"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #37003c 0%, #5a0050 100%); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                    <div style="color: white; font-size: 14px; font-weight: 500; margin-bottom: 4px;">{format_team_name(team1_name)}</div>
                    <div style="color: #00ff87; font-size: 32px; font-weight: 700;">{team1_score:.1f}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px;">{_proj_label1}</div>
                </div>
                """, unsafe_allow_html=True)

            _render_team_lineup(team1_df, team1_name, is_live=gw_is_live)
            if not bench1_df.empty:
                if gw_is_live and live_stats:
                    bench1_df = _blend_live_with_projections(bench1_df, live_stats, player_mapping)
                _render_draft_bench_section(bench1_df, is_live=gw_is_live)

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
                _proj_label2 = "Projected Points (Rotowire + FFP)" if _use_blended else "Projected Points (Rotowire)"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #37003c 0%, #5a0050 100%); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                    <div style="color: white; font-size: 14px; font-weight: 500; margin-bottom: 4px;">{format_team_name(team2_name)}</div>
                    <div style="color: #00ff87; font-size: 32px; font-weight: 700;">{team2_score:.1f}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px;">{_proj_label2}</div>
                </div>
                """, unsafe_allow_html=True)

            _render_team_lineup(team2_df, team2_name, is_live=gw_is_live)
            if not bench2_df.empty:
                if gw_is_live and live_stats:
                    bench2_df = _blend_live_with_projections(bench2_df, live_stats, player_mapping)
                _render_draft_bench_section(bench2_df, is_live=gw_is_live)

        # --- Key Differentials ---
        points_col = 'Blended_Points' if (gw_is_live and 'Blended_Points' in team1_df.columns) else 'Points'
        team1_diffs, team2_diffs = compute_key_differentials(
            team1_df, team2_df,
            format_team_name(team1_name), format_team_name(team2_name),
            points_col=points_col,
        )
        if team1_diffs or team2_diffs:
            render_key_differentials(
                team1_diffs, team2_diffs,
                format_team_name(team1_name), format_team_name(team2_name),
                is_draft=True,
            )

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
                        render_styled_table(
                            pd.DataFrame(match_data),
                            text_align={"Gameweek": "center", "Result": "center"},
                        )