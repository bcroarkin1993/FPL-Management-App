"""
Gameweek Review / Recap — cross-format post-GW summary page.

Shows top/bottom performers league-wide, then per-format sections:
- Classic: summary cards, squad table with captain, captain analysis, optimal lineup
- Draft: squad table, optimal lineup (no captain/rank)

Reuses existing API functions and the find_optimal_gw_lineup() algorithm
from bench_analysis.py.
"""

import logging

import requests
import streamlit as st

import config
from scripts.common.bench_analysis import find_optimal_gw_lineup
from scripts.common.fpl_classic_api import (
    get_classic_bootstrap_static,
    get_classic_team_history,
    get_classic_team_picks,
    _get_classic_gw_live_points,
    _get_classic_gw_live_minutes,
)
from scripts.common.fpl_draft_api import (
    get_fpl_player_mapping,
    _get_draft_gw_live_points,
)
from scripts.common.styled_tables import render_styled_table

_logger = logging.getLogger(__name__)

# Position display helpers
_POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
_POS_BADGE_CLS = {"GK": "pos-gk", "DEF": "pos-def", "MID": "pos-mid", "FWD": "pos-fwd"}
_POS_SORT = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
_DRAFT_POS_MAP = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
def _gw_review_css():
    return """
    <style>
    .gwr-stat-card {
        border: 1px solid #333; border-radius: 10px; padding: 16px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        text-align: center; color: #e0e0e0;
    }
    .gwr-stat-label {
        color: #9ca3af; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .gwr-stat-value {
        font-size: 22px; font-weight: 700;
    }
    .gwr-performer-row {
        display: flex; align-items: center; padding: 5px 10px;
        border-radius: 8px; margin-bottom: 4px;
        background: #16213e; border: 1px solid #333; color: #e0e0e0;
    }
    .gwr-rank { font-weight: 800; color: #00ff87; min-width: 24px; font-size: 0.9rem; }
    .gwr-name { flex: 1; font-weight: 600; color: #e0e0e0; font-size: 0.88rem; }
    .gwr-team-badge { color: #9ca3af; font-size: 0.78rem; margin-right: 8px; min-width: 32px; }
    .gwr-pts { font-weight: 800; padding: 2px 10px; border-radius: 10px; font-size: 0.82rem; }
    .gwr-pts-high { background: linear-gradient(135deg, #00ff87, #02efaa); color: #1a1a2e; }
    .gwr-pts-low { background: #5f2121; color: #fca5a5; }
    .pos-badge {
        display: inline-block; padding: 1px 6px; border-radius: 5px;
        font-size: 0.7rem; font-weight: 700; margin-right: 8px;
        min-width: 32px; text-align: center;
    }
    .pos-gk  { background: #f0c040; color: #333; }
    .pos-def { background: #4caf50; color: white; }
    .pos-mid { background: #2196f3; color: white; }
    .pos-fwd { background: #e91e63; color: white; }
    .gwr-captain-card {
        border: 1px solid #333; border-radius: 10px; padding: 16px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0; text-align: center;
    }
    .gwr-captain-name { font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; }
    .gwr-captain-pts { font-size: 1.4rem; font-weight: 800; }
    .gwr-swap-row {
        display: flex; align-items: center; padding: 6px 12px;
        border-radius: 8px; margin-bottom: 4px;
        background: #16213e; border: 1px solid #333; color: #e0e0e0;
        font-size: 0.88rem;
    }
    .gwr-swap-in { color: #00ff87; font-weight: 700; flex: 1; }
    .gwr-swap-out { color: #ff4757; font-weight: 700; flex: 1; }
    .gwr-swap-arrow { color: #9ca3af; margin: 0 12px; font-size: 1.1rem; }
    .gwr-swap-pts { color: #9ca3af; font-size: 0.82rem; min-width: 60px; text-align: right; }
    .gwr-section-header {
        display: flex; align-items: center; gap: 8px;
        margin: 0.3rem 0 0.6rem 0; font-size: 1.1rem; font-weight: 700;
        color: #00ff87; background: linear-gradient(135deg, #37003c, #5a0060);
        padding: 10px 16px; border-radius: 8px;
    }
    </style>
    """


def _stat_card(label, value, accent="#00ff87"):
    return (
        f'<div class="gwr-stat-card">'
        f'<div class="gwr-stat-label">{label}</div>'
        f'<div class="gwr-stat-value" style="color:{accent};">{value}</div>'
        f'</div>'
    )


def _section_header(icon, text):
    st.markdown(
        f'<div class="gwr-section-header"><span style="font-size:1.2rem;">{icon}</span> {text}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# GW selector helper
# ─────────────────────────────────────────────────────────────
def _get_default_review_gw(bootstrap):
    """Find the most recent GW with finished=True."""
    events = bootstrap.get("events", [])
    finished = [e for e in events if e.get("finished")]
    if finished:
        return max(finished, key=lambda e: e["id"])["id"]
    return None


# ─────────────────────────────────────────────────────────────
# Section 1: Top Performers (league-wide)
# ─────────────────────────────────────────────────────────────
def _render_gw_top_performers(gw, bootstrap):
    """Top 10 scorers and notable blankers for the GW."""
    _section_header("🔥", f"GW{gw} Top Performers")

    live_points = _get_classic_gw_live_points(gw)
    if not live_points:
        st.info("No live points data available for this gameweek.")
        return

    live_minutes = _get_classic_gw_live_minutes(gw)
    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    # Build player list with GW points
    players = []
    for el in elements:
        eid = el["id"]
        pts = live_points.get(eid, 0)
        pos = _POS_MAP.get(el.get("element_type"), "MID")
        players.append({
            "name": el.get("web_name", "?"),
            "team": teams.get(el.get("team"), "?"),
            "pos": pos,
            "pts": pts,
            "minutes": live_minutes.get(eid, 0),
            "selected_by": float(el.get("selected_by_percent", 0) or 0),
        })

    # Top 10 scorers
    top_scorers = sorted(players, key=lambda p: -p["pts"])[:10]

    # Notable blankers: >= 10% ownership, scored 0-1, but only players who actually played
    blankers = [p for p in players if p["selected_by"] >= 10.0 and p["pts"] <= 1 and p["minutes"] > 0]
    blankers.sort(key=lambda p: (-p["selected_by"], p["pts"]))
    blankers = blankers[:10]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Top Scorers**")
        html = ""
        for i, p in enumerate(top_scorers, 1):
            pos_cls = _POS_BADGE_CLS.get(p["pos"], "")
            html += (
                f'<div class="gwr-performer-row">'
                f'<span class="gwr-rank">{i}</span>'
                f'<span class="pos-badge {pos_cls}">{p["pos"]}</span>'
                f'<span class="gwr-name">{p["name"]}</span>'
                f'<span class="gwr-team-badge">{p["team"]}</span>'
                f'<span class="gwr-pts gwr-pts-high">{p["pts"]}</span>'
                f'</div>'
            )
        st.markdown(html, unsafe_allow_html=True)

    with col_right:
        st.markdown("**Notable Blankers**")
        if not blankers:
            st.caption("No widely-owned players blanked this GW.")
        else:
            html = ""
            for i, p in enumerate(blankers, 1):
                pos_cls = _POS_BADGE_CLS.get(p["pos"], "")
                html += (
                    f'<div class="gwr-performer-row">'
                    f'<span class="gwr-rank">{i}</span>'
                    f'<span class="pos-badge {pos_cls}">{p["pos"]}</span>'
                    f'<span class="gwr-name">{p["name"]}</span>'
                    f'<span class="gwr-team-badge">{p["team"]} ({p["selected_by"]:.0f}%)</span>'
                    f'<span class="gwr-pts gwr-pts-low">{p["pts"]}</span>'
                    f'</div>'
                )
            st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Section 2: Classic GW Review
# ─────────────────────────────────────────────────────────────
def _get_classic_gw_data(team_id, gw, bootstrap):
    """Fetch and enrich Classic team data for a single GW."""
    picks_data = get_classic_team_picks(team_id, gw)
    if not picks_data or not picks_data.get("picks"):
        return None

    live_points = _get_classic_gw_live_points(gw)
    history = get_classic_team_history(team_id)

    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    picks = picks_data["picks"]
    active_chip = picks_data.get("active_chip")
    entry_history = picks_data.get("entry_history", {})

    all_players = []
    for pick in picks:
        eid = pick["element"]
        elem = elements.get(eid, {})
        raw_pts = live_points.get(eid, 0)
        pos = _POS_MAP.get(elem.get("element_type"), "MID")
        all_players.append({
            "element_id": eid,
            "web_name": elem.get("web_name", "Unknown"),
            "team": teams.get(elem.get("team"), "?"),
            "position": pos,
            "points": raw_pts,
            "squad_position": pick["position"],
            "multiplier": pick.get("multiplier", 1),
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
        })

    starters = sorted(
        [p for p in all_players if p["squad_position"] <= 11],
        key=lambda p: (_POS_SORT.get(p["position"], 9), -p["points"]),
    )
    bench = sorted(
        [p for p in all_players if p["squad_position"] > 11],
        key=lambda p: p["squad_position"],
    )

    # Rank movement
    gw_rank = entry_history.get("rank", None)
    overall_rank = entry_history.get("overall_rank", None)
    gw_points = entry_history.get("points", 0)

    prev_overall_rank = None
    if history and history.get("current"):
        for h in history["current"]:
            if h.get("event") == gw - 1:
                prev_overall_rank = h.get("overall_rank")
                break

    rank_movement = None
    if prev_overall_rank is not None and overall_rank is not None:
        rank_movement = prev_overall_rank - overall_rank  # positive = improved

    return {
        "all_players": all_players,
        "starters": starters,
        "bench": bench,
        "active_chip": active_chip,
        "gw_points": gw_points,
        "gw_rank": gw_rank,
        "overall_rank": overall_rank,
        "rank_movement": rank_movement,
    }


def _render_classic_summary_cards(data):
    """Summary stat cards for Classic GW review."""
    cols = st.columns(4)

    with cols[0]:
        st.markdown(_stat_card("GW Points", str(data["gw_points"])), unsafe_allow_html=True)

    with cols[1]:
        rank_str = f"{data['gw_rank']:,}" if data["gw_rank"] else "—"
        st.markdown(_stat_card("GW Rank", rank_str), unsafe_allow_html=True)

    with cols[2]:
        or_str = f"{data['overall_rank']:,}" if data["overall_rank"] else "—"
        st.markdown(_stat_card("Overall Rank", or_str), unsafe_allow_html=True)

    with cols[3]:
        mv = data["rank_movement"]
        if mv is None:
            mv_str = "—"
            accent = "#9ca3af"
        elif mv > 0:
            mv_str = f"▲ {mv:,}"
            accent = "#00ff87"
        elif mv < 0:
            mv_str = f"▼ {abs(mv):,}"
            accent = "#ff4757"
        else:
            mv_str = "— 0"
            accent = "#9ca3af"
        st.markdown(_stat_card("Rank Movement", mv_str, accent=accent), unsafe_allow_html=True)


def _render_squad_table(starters, bench, active_chip, is_captain_format=True):
    """Render starting XI and bench tables."""
    import pandas as pd

    # Starting XI
    rows = []
    for p in starters:
        name = p["web_name"]
        if is_captain_format and p.get("is_captain"):
            name += " (C)"
        elif is_captain_format and p.get("is_vice_captain"):
            name += " (V)"

        effective_pts = p["points"] * p.get("multiplier", 1) if is_captain_format else p["points"]
        rows.append({
            "Player": name,
            "Team": p["team"],
            "Pos": p["position"],
            "Pts": effective_pts,
        })

    starter_df = pd.DataFrame(rows)

    chip_label = ""
    if active_chip:
        chip_names = {"bboost": "Bench Boost", "3xc": "Triple Captain", "freehit": "Free Hit", "wildcard": "Wildcard"}
        chip_label = f" — {chip_names.get(active_chip, active_chip)} active"

    render_styled_table(starter_df, title=f"Starting XI{chip_label}")

    # Bench
    if active_chip == "bboost":
        st.caption("Bench Boost active — all 15 players scored.")
    elif bench:
        bench_rows = []
        for p in bench:
            bench_rows.append({
                "Player": p["web_name"],
                "Team": p["team"],
                "Pos": p["position"],
                "Pts": p["points"],
            })
        bench_df = pd.DataFrame(bench_rows)
        render_styled_table(bench_df, title="Bench")


def _render_captain_analysis(starters, active_chip):
    """Captain vs best-captain comparison."""
    _section_header("👑", "Captain Analysis")

    captain = next((p for p in starters if p.get("is_captain")), None)
    if not captain:
        st.info("No captain data available.")
        return

    captain_mult = 3 if active_chip == "3xc" else 2
    captain_effective = captain["points"] * captain_mult

    # Best possible captain = highest scorer among starters (excluding GK typically,
    # but we include all non-GK outfield)
    non_gk_starters = [p for p in starters if p["position"] != "GK"]
    if not non_gk_starters:
        non_gk_starters = starters

    best_option = max(non_gk_starters, key=lambda p: p["points"])
    best_effective = best_option["points"] * captain_mult

    diff = captain_effective - best_effective

    col1, col2 = st.columns(2)

    with col1:
        chip_note = " (TC)" if active_chip == "3xc" else ""
        st.markdown(
            f'<div class="gwr-captain-card">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;margin-bottom:8px;">Your Captain{chip_note}</div>'
            f'<div class="gwr-captain-name">{captain["web_name"]}</div>'
            f'<div style="color:#9ca3af;font-size:0.85rem;">{captain["points"]} base pts x{captain_mult}</div>'
            f'<div class="gwr-captain-pts" style="color:#00ff87;">{captain_effective} pts</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        if best_option["element_id"] == captain["element_id"]:
            verdict_color = "#00ff87"
            verdict_text = "Optimal choice!"
        else:
            verdict_color = "#ff4757" if diff < 0 else "#00ff87"
            verdict_text = f"{diff:+d} pts from captain choice"

        st.markdown(
            f'<div class="gwr-captain-card">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;margin-bottom:8px;">Best Captain Option</div>'
            f'<div class="gwr-captain-name">{best_option["web_name"]}</div>'
            f'<div style="color:#9ca3af;font-size:0.85rem;">{best_option["points"]} base pts x{captain_mult}</div>'
            f'<div class="gwr-captain-pts" style="color:{verdict_color};">{best_effective} pts</div>'
            f'<div style="color:{verdict_color};font-size:0.85rem;margin-top:6px;">{verdict_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_optimal_lineup(all_players, starters, active_chip, is_classic=True):
    """Optimal lineup comparison with swap recommendations."""
    _section_header("🧠", "Optimal Lineup")

    # Skip for Bench Boost — all 15 play
    if active_chip == "bboost":
        st.info("Bench Boost was active — all 15 players scored. No lineup optimization needed.")
        return

    optimal_11 = find_optimal_gw_lineup(all_players)
    if not optimal_11:
        st.info("Could not compute optimal lineup.")
        return

    # Compute scores
    if is_classic:
        # Actual score with captain multiplier
        actual_score = sum(p["points"] * p.get("multiplier", 1) for p in starters)

        captain = next((p for p in all_players if p.get("is_captain")), None)
        captain_mult = 3 if active_chip == "3xc" else 2

        # Optimal: best 11 + best captain (highest non-GK scorer in optimal 11)
        optimal_captain = max(
            [p for p in optimal_11 if p["position"] != "GK"] or optimal_11,
            key=lambda p: p["points"],
        )
        optimal_score = (
            sum(p["points"] for p in optimal_11 if p["element_id"] != optimal_captain["element_id"])
            + optimal_captain["points"] * captain_mult
        )
    else:
        # Draft: no captain
        actual_score = sum(p["points"] for p in starters)
        optimal_score = sum(p["points"] for p in optimal_11)

    points_lost = max(0, optimal_score - actual_score)

    # Summary cards
    cols = st.columns(3)
    with cols[0]:
        st.markdown(_stat_card("Actual Points", str(actual_score)), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_stat_card("Optimal Points", str(optimal_score)), unsafe_allow_html=True)
    with cols[2]:
        accent = "#ff4757" if points_lost > 0 else "#00ff87"
        st.markdown(_stat_card("Points Lost", str(points_lost), accent=accent), unsafe_allow_html=True)

    # Swap recommendations
    if points_lost > 0:
        starter_ids = {p["element_id"] for p in starters}
        optimal_ids = {p["element_id"] for p in optimal_11}

        should_have_started = [p for p in optimal_11 if p["element_id"] not in starter_ids]
        should_have_benched = [p for p in starters if p["element_id"] not in optimal_ids]

        should_have_started.sort(key=lambda p: -p["points"])
        should_have_benched.sort(key=lambda p: p["points"])

        html = ""

        # Show optimal captain if different from actual
        if is_classic and captain and optimal_captain["element_id"] != captain["element_id"]:
            captain_diff = (optimal_captain["points"] - captain["points"]) * (captain_mult - 1)
            html += (
                f'<div class="gwr-swap-row" style="border-left:3px solid #00ff87;">'
                f'<span class="gwr-swap-in">👑 Captain {optimal_captain["web_name"]} '
                f'({optimal_captain["points"]} x{captain_mult} = {optimal_captain["points"] * captain_mult} pts)</span>'
                f'<span class="gwr-swap-arrow">instead of</span>'
                f'<span class="gwr-swap-out">👑 {captain["web_name"]} '
                f'({captain["points"]} x{captain_mult} = {captain["points"] * captain_mult} pts)</span>'
                f'</div>'
            )

        if should_have_started:
            for swap_in, swap_out in zip(should_have_started, should_have_benched):
                html += (
                    f'<div class="gwr-swap-row">'
                    f'<span class="gwr-swap-in">▲ {swap_in["web_name"]} ({swap_in["points"]} pts)</span>'
                    f'<span class="gwr-swap-arrow">⟷</span>'
                    f'<span class="gwr-swap-out">▼ {swap_out["web_name"]} ({swap_out["points"]} pts)</span>'
                    f'</div>'
                )

        if html:
            st.markdown("")
            st.markdown("**Hindsight Changes**")
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.success("Your lineup was optimal! No points were left on the bench.")


def _render_classic_review(gw, bootstrap):
    """Full Classic GW review section."""
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        return

    _section_header("🏆", f"Classic FPL — GW{gw} Review")

    data = _get_classic_gw_data(team_id, gw, bootstrap)
    if not data:
        st.info(f"No pick data available for GW{gw}. This gameweek may not have started yet.")
        return

    _render_classic_summary_cards(data)
    st.markdown("")

    _render_squad_table(data["starters"], data["bench"], data["active_chip"], is_captain_format=True)
    st.markdown("")

    _render_captain_analysis(data["starters"], data["active_chip"])
    st.markdown("")

    _render_optimal_lineup(data["all_players"], data["starters"], data["active_chip"], is_classic=True)


# ─────────────────────────────────────────────────────────────
# Section 3: Draft GW Review
# ─────────────────────────────────────────────────────────────
def _get_draft_gw_data(entry_id, gw, bootstrap):
    """Fetch and enrich Draft team data for a single GW."""
    try:
        url = f"https://draft.premierleague.com/api/entry/{entry_id}/event/{gw}"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        picks = data.get("picks", [])
    except Exception:
        _logger.warning("Failed to fetch draft picks for entry %s GW %s", entry_id, gw, exc_info=True)
        return None

    if not picks:
        return None

    live_points = _get_draft_gw_live_points(gw)
    player_map = get_fpl_player_mapping()

    all_players = []
    for pick in picks:
        eid = pick["element"]
        raw_pts = live_points.get(eid, 0)
        pinfo = player_map.get(eid, {})
        pos_short = pinfo.get("Position", "M")
        pos = _DRAFT_POS_MAP.get(pos_short, "MID")

        # Try bootstrap for web_name first (more reliable), fall back to player_map
        bootstrap_elem = None
        for el in bootstrap.get("elements", []):
            if el["id"] == eid:
                bootstrap_elem = el
                break

        if bootstrap_elem:
            web_name = bootstrap_elem.get("web_name", "Unknown")
            team_id = bootstrap_elem.get("team")
            teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
            team = teams.get(team_id, "?")
        else:
            web_name = pinfo.get("Web_Name") or pinfo.get("Player", "Unknown")
            team = pinfo.get("Team", "?")

        all_players.append({
            "element_id": eid,
            "web_name": web_name,
            "team": team,
            "position": pos,
            "points": raw_pts,
            "squad_position": pick["position"],
        })

    starters = sorted(
        [p for p in all_players if p["squad_position"] <= 11],
        key=lambda p: (_POS_SORT.get(p["position"], 9), -p["points"]),
    )
    bench = sorted(
        [p for p in all_players if p["squad_position"] > 11],
        key=lambda p: p["squad_position"],
    )

    return {
        "all_players": all_players,
        "starters": starters,
        "bench": bench,
    }


def _render_draft_review(gw, bootstrap):
    """Full Draft GW review section."""
    entry_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    if not entry_id:
        return

    _section_header("📋", f"Draft FPL — GW{gw} Review")

    data = _get_draft_gw_data(entry_id, gw, bootstrap)
    if not data:
        st.info(f"No pick data available for GW{gw}. This gameweek may not have started yet.")
        return

    # Simple GW total
    total_pts = sum(p["points"] for p in data["starters"])
    cols = st.columns(3)
    with cols[0]:
        st.markdown(_stat_card("GW Points", str(total_pts)), unsafe_allow_html=True)
    with cols[1]:
        bench_pts = sum(p["points"] for p in data["bench"])
        st.markdown(_stat_card("Bench Points", str(bench_pts)), unsafe_allow_html=True)
    with cols[2]:
        best_scorer = max(data["starters"], key=lambda p: p["points"]) if data["starters"] else None
        if best_scorer:
            st.markdown(
                _stat_card("Top Scorer", f"{best_scorer['web_name']} ({best_scorer['points']})"),
                unsafe_allow_html=True,
            )

    st.markdown("")

    _render_squad_table(data["starters"], data["bench"], active_chip=None, is_captain_format=False)
    st.markdown("")

    _render_optimal_lineup(data["all_players"], data["starters"], active_chip=None, is_classic=False)


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def show_gw_review_page():
    st.header("Gameweek Review")

    st.markdown(_gw_review_css(), unsafe_allow_html=True)

    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        st.error("Could not load FPL data. Please try again later.")
        return

    # GW selector
    default_gw = _get_default_review_gw(bootstrap)
    events = bootstrap.get("events", [])
    all_gws = sorted([e["id"] for e in events])

    if not all_gws:
        st.info("No gameweek data available.")
        return

    if default_gw is None:
        st.info("No completed gameweeks yet. Check back after the first gameweek finishes.")
        return

    default_idx = all_gws.index(default_gw) if default_gw in all_gws else len(all_gws) - 1
    selected_gw = st.selectbox(
        "Select Gameweek",
        all_gws,
        index=default_idx,
        format_func=lambda gw: f"Gameweek {gw}",
    )

    # Check if GW is finished
    gw_event = next((e for e in events if e["id"] == selected_gw), None)
    if gw_event and not gw_event.get("finished"):
        st.warning(f"Gameweek {selected_gw} has not finished yet. Results may be incomplete.")

    st.markdown("")

    # Section 1: Top performers (always shown)
    _render_gw_top_performers(selected_gw, bootstrap)

    st.divider()

    # Section 2 & 3: Format-specific reviews
    classic_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    draft_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)

    if classic_id:
        _render_classic_review(selected_gw, bootstrap)

    if classic_id and draft_id:
        st.divider()

    if draft_id:
        _render_draft_review(selected_gw, bootstrap)

    if not classic_id and not draft_id:
        st.info("Configure FPL_CLASSIC_TEAM_ID or FPL_DRAFT_TEAM_ID in your .env file to see your personal GW review.")
