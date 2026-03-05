"""
Bench Analysis — shared computation + rendering for Draft and Classic Team Analysis.

Computes per-GW bench points, optimal hindsight lineups, and points lost
due to suboptimal lineup decisions. Renders summary cards, a bar chart,
and a per-GW breakdown table.
"""

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from scripts.common.error_helpers import get_logger
from scripts.common.fpl_classic_api import (
    get_classic_bootstrap_static,
    get_classic_team_picks,
    _get_classic_gw_live_points,
)
from scripts.common.fpl_draft_api import (
    get_fpl_player_mapping,
    _get_draft_gw_live_points,
)
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import position_converter

_logger = get_logger("fpl_app.bench_analysis")

# Position constraints for a valid 11-player lineup
_POS_MIN = {"GK": 1, "DEF": 3, "MID": 3, "FWD": 1}  # total = 8
_POS_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}


# =============================================================================
# OPTIMAL LINEUP ALGORITHM
# =============================================================================

def find_optimal_gw_lineup(players):
    """
    Given a list of up to 15 player dicts, return the best 11 respecting
    formation rules (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD).

    Each player dict must have keys: 'element_id', 'position' (GK/DEF/MID/FWD), 'points'.

    Returns a list of the 11 selected player dicts.
    """
    if not players:
        return []

    # Group players by position
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in players:
        pos = p.get("position", "")
        if pos in by_pos:
            by_pos[pos].append(p)

    # Sort each position group by points descending (tie-break by element_id for determinism)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: (-x.get("points", 0), x.get("element_id", 0)))

    # Step 1: Pick required minimums
    selected = []
    remaining = {}
    for pos, minimum in _POS_MIN.items():
        picked = by_pos[pos][:minimum]
        selected.extend(picked)
        remaining[pos] = by_pos[pos][minimum:]

    # Step 2: Fill remaining 3 slots from best available, respecting maximums
    slots_left = 11 - len(selected)
    pos_counts = {pos: _POS_MIN[pos] for pos in _POS_MIN}

    # Pool all remaining candidates
    candidates = []
    for pos, pool in remaining.items():
        for p in pool:
            candidates.append(p)
    candidates.sort(key=lambda x: (-x.get("points", 0), x.get("element_id", 0)))

    for p in candidates:
        if slots_left <= 0:
            break
        pos = p.get("position", "")
        if pos in pos_counts and pos_counts[pos] < _POS_MAX.get(pos, 0):
            selected.append(p)
            pos_counts[pos] += 1
            slots_left -= 1

    return selected


# =============================================================================
# CLASSIC BENCH DATA
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def compute_classic_bench_data(team_id, max_gw):
    """
    Compute per-GW bench analysis for a Classic FPL team.

    Returns dict with 'per_gw' list, totals, etc.
    Bench Boost GWs are included in the table but excluded from points_lost total.
    Free Hit GWs are excluded entirely.
    """
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        return None

    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    POS_DISPLAY = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    per_gw = []
    total_bench_pts = 0
    total_actual = 0
    total_optimal = 0
    total_points_lost = 0

    for gw in range(1, max_gw + 1):
        picks_data = get_classic_team_picks(team_id, gw)
        if not picks_data or not picks_data.get("picks"):
            continue

        picks = picks_data["picks"]
        active_chip = picks_data.get("active_chip")
        live_points = _get_classic_gw_live_points(gw)

        # Build player list with raw points and position
        all_players = []
        for pick in picks:
            eid = pick["element"]
            elem = elements.get(eid, {})
            raw_pts = live_points.get(eid, 0)
            pos = POS_DISPLAY.get(elem.get("element_type", 0), "MID")
            all_players.append({
                "element_id": eid,
                "position": pos,
                "points": raw_pts,
                "squad_position": pick["position"],
                "multiplier": pick.get("multiplier", 1),
                "is_captain": pick.get("is_captain", False),
                "web_name": elem.get("web_name", "Unknown"),
            })

        starters = [p for p in all_players if p["squad_position"] <= 11]
        bench = [p for p in all_players if p["squad_position"] > 11]

        # Actual score: sum(raw_pts * multiplier) for starters
        # With Bench Boost, all 15 play (bench positions still marked 12-15 but multiplier=1)
        if active_chip == "bboost":
            actual_score = sum(p["points"] * p["multiplier"] for p in all_players)
            bench_pts = 0
        else:
            actual_score = sum(p["points"] * p["multiplier"] for p in starters)
            bench_pts = sum(p["points"] for p in bench)

        # Optimal score: best 11, captain = highest scorer gets 2x (or 3x if 3xc active)
        optimal_11 = find_optimal_gw_lineup(all_players)
        captain_mult = 3 if active_chip == "3xc" else 2
        if optimal_11:
            best_scorer = max(optimal_11, key=lambda p: p["points"])
            optimal_score = sum(p["points"] for p in optimal_11) + best_scorer["points"] * (captain_mult - 1)
        else:
            optimal_score = actual_score

        # Top bench player
        if bench:
            top_bench_player = max(bench, key=lambda p: p["points"])
            top_bench_str = f"{top_bench_player['web_name']} ({top_bench_player['points']})"
        else:
            top_bench_str = "-"

        points_lost = max(0, optimal_score - actual_score)

        gw_data = {
            "gw": gw,
            "actual": actual_score,
            "bench_pts": bench_pts,
            "optimal": optimal_score,
            "points_lost": points_lost,
            "top_bench": top_bench_str,
            "active_chip": active_chip,
        }
        per_gw.append(gw_data)

        total_actual += actual_score
        total_bench_pts += bench_pts

        # Exclude BB and FH from points_lost total (they distort the analysis)
        if active_chip not in ("bboost", "freehit"):
            total_points_lost += points_lost
            total_optimal += optimal_score
        else:
            total_optimal += actual_score  # treat as "no loss" for these GWs

    return {
        "per_gw": per_gw,
        "total_bench_pts": total_bench_pts,
        "total_actual": total_actual,
        "total_optimal": total_optimal,
        "total_points_lost": total_points_lost,
    }


# =============================================================================
# DRAFT BENCH DATA
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def compute_draft_bench_data(entry_id, max_gw):
    """
    Compute per-GW bench analysis for a Draft FPL team.

    No captaincy in Draft — actual = sum(pts for starters), optimal = best 11.
    """
    player_map = get_fpl_player_mapping()
    if not player_map:
        return None

    per_gw = []
    total_bench_pts = 0
    total_actual = 0
    total_optimal = 0
    total_points_lost = 0

    for gw in range(1, max_gw + 1):
        # Fetch picks for this GW
        try:
            url = f"https://draft.premierleague.com/api/entry/{entry_id}/event/{gw}"
            resp = requests.get(url, timeout=30)
            data = resp.json()
            picks = data.get("picks", [])
        except Exception:
            _logger.warning("Failed to fetch draft picks for entry %s GW %s", entry_id, gw, exc_info=True)
            continue

        if not picks:
            continue

        live_points = _get_draft_gw_live_points(gw)

        POS_DISPLAY = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

        all_players = []
        for pick in picks:
            eid = pick["element"]
            raw_pts = live_points.get(eid, 0)
            pinfo = player_map.get(eid, {})
            pos_short = pinfo.get("Position", "M")
            pos = POS_DISPLAY.get(pos_short, "MID")
            web_name = pinfo.get("Web_Name") or pinfo.get("Player", "Unknown")
            all_players.append({
                "element_id": eid,
                "position": pos,
                "points": raw_pts,
                "squad_position": pick["position"],
                "web_name": web_name,
            })

        starters = [p for p in all_players if p["squad_position"] <= 11]
        bench = [p for p in all_players if p["squad_position"] > 11]

        actual_score = sum(p["points"] for p in starters)
        bench_pts = sum(p["points"] for p in bench)

        # Optimal: best 11 (no captain in Draft)
        optimal_11 = find_optimal_gw_lineup(all_players)
        optimal_score = sum(p["points"] for p in optimal_11) if optimal_11 else actual_score

        # Top bench player
        if bench:
            top_bench_player = max(bench, key=lambda p: p["points"])
            top_bench_str = f"{top_bench_player['web_name']} ({top_bench_player['points']})"
        else:
            top_bench_str = "-"

        points_lost = max(0, optimal_score - actual_score)

        per_gw.append({
            "gw": gw,
            "actual": actual_score,
            "bench_pts": bench_pts,
            "optimal": optimal_score,
            "points_lost": points_lost,
            "top_bench": top_bench_str,
            "active_chip": None,
        })

        total_actual += actual_score
        total_bench_pts += bench_pts
        total_optimal += optimal_score
        total_points_lost += points_lost

    return {
        "per_gw": per_gw,
        "total_bench_pts": total_bench_pts,
        "total_actual": total_actual,
        "total_optimal": total_optimal,
        "total_points_lost": total_points_lost,
    }


# =============================================================================
# RENDERING
# =============================================================================

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title_font=dict(size=20, color="#ffffff"),
    title_x=0.5,
    title_xanchor="center",
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
)


def _stat_card(label, value, accent="#00ff87"):
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'</div>'
    )


def render_bench_analysis(bench_data, is_classic=True):
    """
    Render bench analysis section with summary cards, bar chart, and per-GW table.

    Parameters:
    - bench_data: dict from compute_classic_bench_data or compute_draft_bench_data
    - is_classic: True for Classic (shows captain info), False for Draft
    """
    per_gw = bench_data["per_gw"]
    if not per_gw:
        st.info("No bench data available.")
        return

    num_gws = len([g for g in per_gw if g.get("active_chip") not in ("bboost", "freehit")])

    # --- Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(_stat_card("Total Bench Pts", str(bench_data["total_bench_pts"])), unsafe_allow_html=True)
    with col2:
        avg = bench_data["total_bench_pts"] / num_gws if num_gws > 0 else 0
        st.markdown(_stat_card("Avg Bench Pts/GW", f"{avg:.1f}"), unsafe_allow_html=True)
    with col3:
        st.markdown(_stat_card("Total Pts Lost", str(bench_data["total_points_lost"]), accent="#f87171"), unsafe_allow_html=True)
    with col4:
        # Worst decision GW (excluding BB/FH)
        eligible = [g for g in per_gw if g.get("active_chip") not in ("bboost", "freehit")]
        if eligible:
            worst = max(eligible, key=lambda g: g["points_lost"])
            worst_str = f"GW{worst['gw']}: {worst['points_lost']} pts"
        else:
            worst_str = "N/A"
        st.markdown(_stat_card("Worst Decision", worst_str, accent="#f87171"), unsafe_allow_html=True)

    st.markdown("")  # spacer

    # --- Bar Chart: Points Lost per GW ---
    chart_data = [g for g in per_gw if g.get("active_chip") not in ("freehit",)]  # show BB but not FH
    if chart_data:
        gws = [g["gw"] for g in chart_data]
        pts_lost = [g["points_lost"] for g in chart_data]
        chips = [g.get("active_chip") for g in chart_data]

        # Color bars by magnitude (higher = redder), muted for BB
        colors = []
        max_lost = max(pts_lost) if pts_lost else 1
        for val, chip in zip(pts_lost, chips):
            if chip == "bboost":
                colors.append("rgba(100,100,100,0.5)")
            elif max_lost > 0:
                ratio = val / max_lost
                r = int(60 + 160 * ratio)
                g = int(200 - 140 * ratio)
                colors.append(f"rgb({r},{g},60)")
            else:
                colors.append("rgb(60,200,60)")

        fig = go.Figure(data=[
            go.Bar(
                x=gws,
                y=pts_lost,
                marker_color=colors,
                hovertemplate="GW %{x}: %{y} pts lost<extra></extra>",
            )
        ])
        fig.add_hline(y=0, line_dash="dash", line_color="#666")
        fig.update_layout(
            **_DARK_CHART_LAYOUT,
            title="Points Lost per Gameweek",
            height=350,
            showlegend=False,
        )
        fig.update_xaxes(title="Gameweek", dtick=1)
        fig.update_yaxes(title="Points Lost")
        st.plotly_chart(fig, use_container_width=True)

    # --- Per-GW Table ---
    rows = []
    for g in per_gw:
        chip_label = ""
        if g.get("active_chip"):
            chip_names = {"bboost": "BB", "3xc": "3xC", "freehit": "FH", "wildcard": "WC"}
            chip_label = f" ({chip_names.get(g['active_chip'], g['active_chip'])})"

        rows.append({
            "GW": f"{g['gw']}{chip_label}",
            "Actual": g["actual"],
            "Bench Pts": g["bench_pts"],
            "Optimal": g["optimal"],
            "Pts Lost": g["points_lost"],
            "Top Bench Player": g["top_bench"],
        })

    # Sort most recent first
    rows.reverse()
    table_df = pd.DataFrame(rows)

    render_styled_table(
        table_df,
        positive_color_cols=["Pts Lost"],
    )
