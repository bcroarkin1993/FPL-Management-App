# scripts/common/team_analysis_helpers.py
"""
Shared helpers for Team Analysis pages (Draft and Classic).

Provides:
- Best Clubs: Top contributing EPL clubs by points
- Season Best 11: Valid soccer lineup of top performers
- Team MVP: Best scoring player with detailed stats
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any

from scripts.common.utils import get_classic_bootstrap_static


def get_best_clubs(player_data: List[Dict], top_n: int = 3) -> pd.DataFrame:
    """
    Aggregate points by EPL club and return top N clubs.

    Args:
        player_data: List of dicts with keys: player, position, total_points, team
        top_n: Number of top clubs to return

    Returns:
        DataFrame with columns: Rank, Club, Points, Players
    """
    if not player_data:
        return pd.DataFrame(columns=['Rank', 'Club', 'Points', 'Players'])

    # Aggregate points by club
    club_points = {}
    club_players = {}

    for p in player_data:
        club = p.get('team', 'Unknown')
        pts = p.get('total_points', 0)
        player_name = p.get('player', 'Unknown')

        if club not in club_points:
            club_points[club] = 0
            club_players[club] = []

        club_points[club] += pts
        club_players[club].append(player_name)

    # Create sorted list
    sorted_clubs = sorted(club_points.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for i, (club, pts) in enumerate(sorted_clubs, 1):
        players = club_players[club]
        rows.append({
            'Rank': i,
            'Club': club,
            'Points': pts,
            'Players': ', '.join(players),
        })

    return pd.DataFrame(rows)


def get_season_best_11(player_data: List[Dict]) -> Dict[str, Any]:
    """
    Select the best 11 players forming a valid soccer formation.

    Valid formations: 1 GK + 3-5 DEF + 3-5 MID + 1-3 FWD = 11

    Args:
        player_data: List of dicts with keys: player, position, total_points, team

    Returns:
        Dict with:
            - formation: str (e.g., "4-4-2")
            - players: List of player dicts sorted by position
            - total_points: int
    """
    if not player_data:
        return {'formation': 'N/A', 'players': [], 'total_points': 0}

    # Map position names to standard
    pos_map = {'GK': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'}

    # Group by position and sort by points
    by_position = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}

    for p in player_data:
        pos = p.get('position', '').upper()
        pos = pos_map.get(pos, pos)
        if pos in by_position:
            by_position[pos].append(p)

    for pos in by_position:
        by_position[pos] = sorted(by_position[pos], key=lambda x: x.get('total_points', 0), reverse=True)

    # Try different valid formations and pick the one with max points
    # Formations: DEF-MID-FWD
    formations = [
        (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
        (5, 3, 2), (5, 4, 1), (5, 2, 3),
    ]

    best_formation = None
    best_lineup = []
    best_total = -1

    for num_def, num_mid, num_fwd in formations:
        # Check if we have enough players at each position
        if (len(by_position['GK']) < 1 or
            len(by_position['DEF']) < num_def or
            len(by_position['MID']) < num_mid or
            len(by_position['FWD']) < num_fwd):
            continue

        # Build lineup
        lineup = (
            by_position['GK'][:1] +
            by_position['DEF'][:num_def] +
            by_position['MID'][:num_mid] +
            by_position['FWD'][:num_fwd]
        )

        total = sum(p.get('total_points', 0) for p in lineup)

        if total > best_total:
            best_total = total
            best_lineup = lineup
            best_formation = f"{num_def}-{num_mid}-{num_fwd}"

    if not best_lineup:
        # Fallback: just return top 11 by points
        all_players = sorted(player_data, key=lambda x: x.get('total_points', 0), reverse=True)[:11]
        return {
            'formation': 'Best Available',
            'players': all_players,
            'total_points': sum(p.get('total_points', 0) for p in all_players)
        }

    return {
        'formation': best_formation,
        'players': best_lineup,
        'total_points': best_total
    }


def get_team_mvp(player_data: List[Dict], bootstrap_data: Optional[Dict] = None) -> Optional[Dict]:
    """
    Find the MVP (best scoring player) with detailed stats.

    Args:
        player_data: List of dicts with keys: player, position, total_points, team
        bootstrap_data: Optional bootstrap-static data for extra stats

    Returns:
        Dict with: player, team, position, total_points, goals, assists, starts
        or None if no players
    """
    if not player_data:
        return None

    # Find top scorer
    mvp = max(player_data, key=lambda x: x.get('total_points', 0))

    result = {
        'player': mvp.get('player', 'Unknown'),
        'team': mvp.get('team', '???'),
        'position': mvp.get('position', '???'),
        'total_points': mvp.get('total_points', 0),
        'goals': 0,
        'assists': 0,
        'starts': 0,
    }

    # Try to enrich with bootstrap data
    if bootstrap_data:
        elements = bootstrap_data.get('elements', [])
        # Find matching player by web_name
        for elem in elements:
            if elem.get('web_name') == mvp.get('player'):
                result['goals'] = elem.get('goals_scored', 0)
                result['assists'] = elem.get('assists', 0)
                # Use 'starts' field from FPL API (games with 60+ minutes)
                result['starts'] = elem.get('starts', 0)
                break

    return result


def get_classic_mvp_with_captain_stats(
    team_id: int,
    player_data: List[Dict],
    bootstrap_data: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Find MVP for Classic FPL with captain stats.

    Returns MVP dict plus captain_points and times_captained fields.
    """
    from scripts.common.utils import get_classic_team_picks, get_current_gameweek

    mvp = get_team_mvp(player_data, bootstrap_data)
    if not mvp:
        return None

    # Add captain stats
    mvp['captain_points'] = 0
    mvp['times_captained'] = 0
    mvp['captain_appearances'] = 0

    # Find the element_id for the MVP
    if not bootstrap_data:
        return mvp

    mvp_element_id = None
    for elem in bootstrap_data.get('elements', []):
        if elem.get('web_name') == mvp['player']:
            mvp_element_id = elem.get('id')
            break

    if not mvp_element_id:
        return mvp

    # Check each gameweek for captain picks
    current_gw = get_current_gameweek()
    captain_gws = []
    total_captain_pts = 0

    for gw in range(1, current_gw + 1):
        try:
            picks_data = get_classic_team_picks(team_id, gw)
            if not picks_data:
                continue

            picks = picks_data.get('picks', [])
            for pick in picks:
                if pick.get('element') == mvp_element_id and pick.get('is_captain'):
                    captain_gws.append(gw)
                    # Get the points scored that GW (with captain multiplier)
                    multiplier = pick.get('multiplier', 2)
                    entry_history = picks_data.get('entry_history', {})
                    # We'd need live points to get exact captain contribution
                    break
        except Exception:
            continue

    mvp['times_captained'] = len(captain_gws)
    mvp['captain_appearances'] = len(captain_gws)

    return mvp


def render_best_clubs_section(player_data: List[Dict], top_n: int = 3, min_height: int = 0):
    """Render the Best Clubs section with styled cards."""
    st.markdown("##### 🏆 Best Clubs")

    clubs_df = get_best_clubs(player_data, top_n)

    if clubs_df.empty:
        st.info("No club data available yet.")
        return

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    colors = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}

    # Build all clubs as single HTML block with more padding for height
    clubs_html = ""
    for _, row in clubs_df.iterrows():
        rank = row['Rank']
        medal = medals.get(rank, "")
        color = colors.get(rank, "#666")

        clubs_html += f'''<div style="display:flex;align-items:center;padding:10px 15px;margin-bottom:9px;background:linear-gradient(90deg,rgba(26,26,46,0.9) 0%,rgba(22,33,62,0.7) 100%);border-left:4px solid {color};border-radius:6px;">
<span style="font-size:1.8em;margin-right:15px;">{medal}</span>
<div style="flex:1;">
<div style="color:#e0e0e0;font-weight:bold;font-size:1.1em;">{row['Club']}</div>
<div style="color:#888;font-size:0.9em;margin-top:4px;">{row['Players']}</div>
</div>
<div style="color:{color};font-weight:bold;font-size:1.4em;">{row['Points']}</div>
</div>'''

    # Wrap in container with optional min-height
    min_height_style = f"min-height:{min_height}px;" if min_height > 0 else ""
    html = f'''<div style="border:1px solid #333;border-radius:10px;padding:15px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);{min_height_style}">
{clubs_html}
</div>'''

    st.markdown(html, unsafe_allow_html=True)


def render_season_best_11(player_data: List[Dict]):
    """Render the Season Best 11 section with formation display."""
    st.markdown("##### ⭐ Season Best XI")

    best_11 = get_season_best_11(player_data)

    if not best_11['players']:
        st.info("Not enough player data to form a Best XI.")
        return

    # Position colors
    pos_colors = {'GK': '#f39c12', 'DEF': '#3498db', 'MID': '#2ecc71', 'FWD': '#e74c3c'}
    pos_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}

    # Sort players by position
    sorted_players = sorted(best_11['players'], key=lambda x: (pos_order.get(x.get('position', 'FWD'), 4), -x.get('total_points', 0)))

    # Build all player rows - centered name with good spacing
    player_rows = ""
    for p in sorted_players:
        pos = p.get('position', '?')
        color = pos_colors.get(pos, '#666')
        player_name = p.get('player', 'Unknown')
        team = p.get('team', '???')
        pts = p.get('total_points', 0)

        player_rows += f'''<div style="display:flex;align-items:center;padding:8px 12px;margin-bottom:6px;background:rgba(255,255,255,0.05);border-radius:5px;">
<span style="background:{color};color:#fff;padding:3px 8px;border-radius:4px;font-size:0.85em;font-weight:bold;min-width:36px;text-align:center;">{pos}</span>
<span style="color:#e0e0e0;font-size:1.05em;flex:1;text-align:center;padding:0 10px;">{player_name}</span>
<span style="color:#888;font-size:0.9em;min-width:40px;text-align:center;">{team}</span>
<span style="color:#4ecca3;font-weight:bold;font-size:1.05em;min-width:40px;text-align:right;">{pts}</span>
</div>'''

    # Complete card HTML with min-height to match right column
    html = f'''<div style="border:1px solid #333;border-radius:10px;padding:12px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);min-height:450px;">
<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;margin-bottom:10px;background:rgba(0,0,0,0.3);border-radius:6px;">
<div><div style="color:#888;font-size:0.8em;">Formation</div><div style="color:#4ecca3;font-weight:bold;font-size:1.3em;">{best_11['formation']}</div></div>
<div style="text-align:right;"><div style="color:#888;font-size:0.8em;">Total Points</div><div style="color:#ffd700;font-weight:bold;font-size:1.3em;">{best_11['total_points']}</div></div>
</div>
{player_rows}
</div>'''

    st.markdown(html, unsafe_allow_html=True)


def render_team_mvp(
    player_data: List[Dict],
    bootstrap_data: Optional[Dict] = None,
    team_id: Optional[int] = None,
    is_classic: bool = False,
    min_height: int = 0
):
    """Render the Team MVP section with stats card."""
    st.markdown("##### 👑 Team MVP")

    if is_classic and team_id:
        mvp = get_classic_mvp_with_captain_stats(team_id, player_data, bootstrap_data)
    else:
        mvp = get_team_mvp(player_data, bootstrap_data)

    if not mvp:
        st.info("No MVP data available yet.")
        return

    # Build stats grid - order: Starts, Goals, Assists, Points (+ Captained for Classic)
    stats_html = f'''<div style="text-align:center;padding:8px 10px;"><div style="color:#9b59b6;font-size:1.5em;font-weight:bold;">{mvp['starts']}</div><div style="color:#888;font-size:0.8em;">Starts*</div></div>'''
    stats_html += f'''<div style="text-align:center;padding:8px 10px;"><div style="color:#e74c3c;font-size:1.5em;font-weight:bold;">{mvp['goals']}</div><div style="color:#888;font-size:0.8em;">Goals</div></div>'''
    stats_html += f'''<div style="text-align:center;padding:8px 10px;"><div style="color:#3498db;font-size:1.5em;font-weight:bold;">{mvp['assists']}</div><div style="color:#888;font-size:0.8em;">Assists</div></div>'''
    stats_html += f'''<div style="text-align:center;padding:8px 10px;"><div style="color:#4ecca3;font-size:1.5em;font-weight:bold;">{mvp['total_points']}</div><div style="color:#888;font-size:0.8em;">Points</div></div>'''

    if is_classic and 'times_captained' in mvp:
        stats_html += f'''<div style="text-align:center;padding:8px 10px;"><div style="color:#f39c12;font-size:1.5em;font-weight:bold;">{mvp['times_captained']}</div><div style="color:#888;font-size:0.8em;">Captained</div></div>'''

    # MVP Card as single HTML string with optional min-height
    min_height_style = f"min-height:{min_height}px;" if min_height > 0 else ""
    html = f'''<div style="border:2px solid #ffd700;border-radius:12px;padding:20px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;{min_height_style}">
<div style="font-size:2.2em;margin-bottom:8px;">👑</div>
<div style="color:#ffd700;font-size:1.4em;font-weight:bold;">{mvp['player']}</div>
<div style="color:#aaa;font-size:0.95em;margin-bottom:15px;">{mvp['team']} • {mvp['position']}</div>
<div style="display:flex;justify-content:space-around;flex-wrap:wrap;">{stats_html}</div>
<div style="color:#666;font-size:0.75em;margin-top:12px;font-style:italic;">*Starts = games with 60+ minutes played</div>
</div>'''

    st.markdown(html, unsafe_allow_html=True)


def render_season_highlights(
    player_data: List[Dict],
    bootstrap_data: Optional[Dict] = None,
    team_id: Optional[int] = None,
    is_classic: bool = False
):
    """
    Render the complete Season Highlights section with proper layout.

    Layout: Best XI (left, tall) | MVP + Best Clubs stacked (right)
    """
    if not player_data:
        st.info("Season highlights will appear once you have gameweek data.")
        return

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_season_best_11(player_data)

    with col_right:
        # MVP gets ~40% of height, Best Clubs gets ~60%
        render_team_mvp(player_data, bootstrap_data, team_id, is_classic, min_height=180)
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        render_best_clubs_section(player_data, top_n=3, min_height=220)
