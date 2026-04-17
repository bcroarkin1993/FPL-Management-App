"""
Classic FPL - Transfer Suggestions Page

Displays transfer targets ranked by projected points, form, FDR, and price.
Shows squad analysis with suggested transfers and upcoming fixtures.
"""

import config
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List
from fuzzywuzzy import fuzz

from scripts.common.error_helpers import show_api_error
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_classic_team_picks,
    get_classic_team_history,
    get_entry_details,
    get_current_gameweek,
    get_rotowire_player_projections,
    get_classic_transfers,
    position_converter,
)
from scripts.common.styled_tables import render_styled_table
from scripts.common.analytics import (
    compute_player_scores,
    compute_healthy_form,
    _fetch_element_history,
    compute_positional_depth,
    compute_transfer_urgency,
    blend_multi_gw_projections,
    positional_rank,
    merge_season_projections,
    merge_ffp_single_gw_data,
)
from scripts.common.scraping import get_ffp_projections_data, get_rotowire_season_rankings


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def _format_money(value: int) -> str:
    """Format FPL money value (stored as tenths) to display format."""
    if value is None:
        return "N/A"
    return f"£{value / 10:.1f}m"


def _format_price_change(change: int) -> str:
    """Format price change with indicator."""
    if change is None or change == 0:
        return ""
    if change > 0:
        return f"↑{change/10:.1f}"
    return f"↓{abs(change)/10:.1f}"


@st.cache_data(ttl=300)
def _load_future_fixtures() -> pd.DataFrame:
    """
    Returns future fixtures with difficulties.
    Columns: event, team_h, team_a, team_h_difficulty, team_a_difficulty
    """
    import requests
    url = "https://fantasy.premierleague.com/api/fixtures/?future=1"
    try:
        fx = requests.get(url, timeout=30).json()
        df = pd.DataFrame(fx)
        keep = ["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
        df = df[[c for c in keep if c in df.columns]].copy()
        return df
    except Exception:
        return pd.DataFrame()


def _get_team_fixtures(team_id: int, n_weeks: int, current_gw: int) -> List[Dict]:
    """Get next n fixtures for a team with FDR."""
    fixtures = _load_future_fixtures()
    if fixtures.empty:
        return []

    fixtures = fixtures.dropna(subset=["event"])
    fixtures["event"] = fixtures["event"].astype(int)

    upcoming = fixtures[
        (fixtures["event"] >= current_gw) &
        (fixtures["event"] < current_gw + n_weeks)
    ].copy()

    result = []
    for _, row in upcoming.iterrows():
        if row.get("team_h") == team_id:
            result.append({
                "gw": int(row["event"]),
                "opponent": int(row["team_a"]),
                "home": True,
                "fdr": row.get("team_h_difficulty", 3)
            })
        elif row.get("team_a") == team_id:
            result.append({
                "gw": int(row["event"]),
                "opponent": int(row["team_h"]),
                "home": False,
                "fdr": row.get("team_a_difficulty", 3)
            })

    return sorted(result, key=lambda x: x["gw"])


def _avg_fdr_for_team(team_id: int, current_gw: int, n_weeks: int) -> Optional[float]:
    """Average FDR over next n_weeks for a team."""
    fixtures = _get_team_fixtures(team_id, n_weeks, current_gw)
    if not fixtures:
        return None
    fdr_values = [f["fdr"] for f in fixtures if f.get("fdr")]
    return float(np.mean(fdr_values)) if fdr_values else None


def _get_fdr_color(fdr: float) -> str:
    """Get background color for FDR value."""
    if fdr is None:
        return "#808080"
    if fdr <= 2:
        return "#00c853"  # Green - easy
    elif fdr <= 2.5:
        return "#7cb342"  # Light green
    elif fdr <= 3:
        return "#ffc107"  # Yellow - medium
    elif fdr <= 3.5:
        return "#ff9800"  # Orange
    else:
        return "#dc3545"  # Red - hard


def _lookup_projection(player_name: str, team: str, position: str, projections_df: pd.DataFrame) -> dict:
    """Look up projection for a player using fuzzy matching."""
    if projections_df is None or projections_df.empty:
        return {"Points": None, "Pos Rank": None}

    best_match = None
    best_score = 0

    for _, row in projections_df.iterrows():
        proj_name = str(row.get("Player", ""))
        proj_team = str(row.get("Team", ""))
        proj_pos = str(row.get("Position", ""))

        # Calculate name similarity
        score = fuzz.ratio(player_name.lower(), proj_name.lower())

        # Boost score if team and position match
        if proj_team == team and proj_pos == position:
            score += 15

        if score > best_score and score >= 60:
            best_score = score
            best_match = row

    if best_match is not None:
        return {
            "Points": best_match.get("Points"),
            "Pos Rank": best_match.get("Pos Rank", "N/A"),
        }

    return {"Points": None, "Pos Rank": None}


def _build_all_players_df(bootstrap: dict, current_gw: int, n_weeks: int) -> pd.DataFrame:
    """Build a DataFrame of all players with relevant stats."""
    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}

    rows = []
    for p in elements:
        team_id = p.get("team")
        team_info = teams.get(team_id, {})

        rows.append({
            "Player_ID": p.get("id"),
            "Player": p.get("web_name"),
            "Full Name": f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
            "Team": team_info.get("short_name", "???"),
            "Team_ID": team_id,
            "Position": position_converter(p.get("element_type")),
            "element_type": p.get("element_type"),
            "now_cost": p.get("now_cost", 0),
            "selling_price": p.get("selling_price", p.get("now_cost", 0)),
            "form": float(p.get("form", 0) or 0),
            "points_per_game": float(p.get("points_per_game", 0) or 0),
            "total_points": p.get("total_points", 0),
            "selected_by_percent": float(p.get("selected_by_percent", 0) or 0),
            "transfers_in_event": p.get("transfers_in_event", 0),
            "transfers_out_event": p.get("transfers_out_event", 0),
            "cost_change_event": p.get("cost_change_event", 0),
            "ep_next": float(p.get("ep_next", 0) or 0),  # Expected points next GW
            "minutes": p.get("minutes", 0),
            "starts": p.get("starts", 0),
            "news": p.get("news", ""),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
        })

    df = pd.DataFrame(rows)

    # Add average FDR for next n weeks
    df["AvgFDR"] = df["Team_ID"].apply(lambda t: _avg_fdr_for_team(t, current_gw, n_weeks))

    return df


def _build_squad_df(picks: list, bootstrap: dict, entry_history: dict) -> pd.DataFrame:
    """Build squad DataFrame from picks."""
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    rows = []
    for pick in picks:
        element_id = pick["element"]
        player = elements.get(element_id, {})
        team_id = player.get("team")

        rows.append({
            "Player_ID": element_id,
            "Player": player.get("web_name", "Unknown"),
            "Full Name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "Team": teams.get(team_id, "???"),
            "Team_ID": team_id,
            "Position": position_converter(player.get("element_type")),
            "element_type": player.get("element_type"),
            "squad_position": pick["position"],
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "multiplier": pick.get("multiplier", 1),
            "now_cost": player.get("now_cost", 0),
            "selling_price": pick.get("selling_price", player.get("now_cost", 0)),
            "form": float(player.get("form", 0) or 0),
            "points_per_game": float(player.get("points_per_game", 0) or 0),
            "total_points": player.get("total_points", 0),
            "ep_next": float(player.get("ep_next", 0) or 0),
            "minutes": player.get("minutes", 0),
            "starts": player.get("starts", 0),
            "news": player.get("news", ""),
            "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
        })

    return pd.DataFrame(rows)


def _add_projections(df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """Add Rotowire projections to a DataFrame."""
    if projections_df is None or projections_df.empty:
        df["Projected_Points"] = None
        df["Pos_Rank"] = None
        return df

    proj_points = []
    proj_ranks = []

    for _, row in df.iterrows():
        proj = _lookup_projection(
            row["Player"],
            row["Team"],
            row["Position"],
            projections_df
        )
        proj_points.append(proj["Points"])
        proj_ranks.append(proj["Pos Rank"])

    df["Projected_Points"] = proj_points
    df["Pos_Rank"] = proj_ranks
    return df



def _compute_transfer_score(df: pd.DataFrame,
                            all_players_df: Optional[pd.DataFrame] = None,
                            current_gw: int = 19) -> pd.DataFrame:
    """Compute transfer target scores using positional percentile scoring.

    Delegates to shared compute_player_scores() with format_context="classic"
    and adds a small price-efficiency adjustment to Transfer Score only.
    """
    tmp = compute_player_scores(df, all_players_df, current_gw, format_context="classic")

    # Classic-specific: small price-efficiency adjustment for Transfer Score only
    # Cheaper players are more valuable as transfer targets (budget flexibility)
    if "now_cost" in tmp.columns and "Transfer Score" in tmp.columns:
        max_cost = pd.to_numeric(tmp["now_cost"], errors="coerce").max()
        if pd.notna(max_cost) and max_cost > 0:
            price_bonus = (max_cost - pd.to_numeric(tmp["now_cost"], errors="coerce")) / max_cost * 0.05
            tmp["Transfer Score"] = (tmp["Transfer Score"] + price_bonus).clip(upper=1.0)

    return tmp


def _compute_keep_score(df: pd.DataFrame,
                        all_players_df: Optional[pd.DataFrame] = None,
                        current_gw: int = 19,
                        depth_map: Optional[Dict] = None) -> pd.DataFrame:
    """Compute keep scores using positional percentile scoring.

    Delegates to shared compute_player_scores() with format_context="classic"
    and depth_map for squad depth awareness.
    """
    return compute_player_scores(df, all_players_df, current_gw,
                                 format_context="classic", depth_map=depth_map)


def _format_fixtures_html(fixtures: List[Dict], teams: Dict[int, str], n_show: int = 5) -> str:
    """Format fixtures as HTML with colored FDR badges."""
    if not fixtures:
        return "<span style='color: #888;'>No fixtures</span>"

    html_parts = []
    for f in fixtures[:n_show]:
        opp = teams.get(f["opponent"], "???")
        venue = "H" if f["home"] else "A"
        fdr = f.get("fdr", 3)
        color = _get_fdr_color(fdr)
        html_parts.append(
            f"<span style='background-color:{color}; color:white; padding:2px 6px; "
            f"border-radius:4px; margin-right:4px; font-size:0.85em;'>{opp}({venue})</span>"
        )

    return "".join(html_parts)


def _get_availability_indicator(chance: Optional[int], news: str) -> str:
    """Get availability indicator based on chance of playing."""
    if chance is None:
        if news:
            return f"⚠️ {news[:30]}..."
        return "✓"
    if chance == 0:
        return f"❌ {news[:25]}..." if news else "❌ Out"
    elif chance <= 25:
        return f"🔴 {chance}%"
    elif chance <= 50:
        return f"🟠 {chance}%"
    elif chance <= 75:
        return f"🟡 {chance}%"
    else:
        return "✓"


def _build_transfer_suggestions(squad_df: pd.DataFrame, available_df: pd.DataFrame,
                                 bank: int, top_n: int = 3, depth_map: Optional[Dict] = None) -> List[Dict]:
    """Build transfer suggestions pairing lowest-keep-score squad players with best replacements."""
    if squad_df.empty or available_df.empty:
        return []

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}
    suggestions = []

    lowest_keep = squad_df.nsmallest(top_n, "Keep Score")

    for _, drop_row in lowest_keep.iterrows():
        pos = drop_row["Position"]
        selling_price = drop_row.get("selling_price", drop_row.get("now_cost", 0))
        budget = bank + selling_price

        # Find best replacement at same position within budget
        candidates = available_df[
            (available_df["Position"] == pos) &
            (available_df["now_cost"] <= budget)
        ].copy()

        if candidates.empty:
            continue

        add_row = candidates.iloc[0]  # Already sorted by Transfer Score desc

        # Calculate score improvement using Transfer Score vs Keep Score
        score_diff = add_row.get("Transfer Score", 0) - drop_row.get("Keep Score", 0)

        # Protect elite players from marginal swaps — scale threshold by Keep Score
        keep_score = float(drop_row.get("Keep Score", 0.5) or 0.5)
        if keep_score > 0.7:
            min_threshold = 0.15
        elif keep_score > 0.5:
            min_threshold = 0.08
        else:
            min_threshold = 0.02

        if score_diff < min_threshold:
            continue

        # Availability info
        drop_chance = drop_row.get("chance_of_playing_next_round")
        drop_news = drop_row.get("news", "")
        drop_injury = _get_availability_indicator(drop_chance, drop_news)

        add_chance = add_row.get("chance_of_playing_next_round")
        add_news = add_row.get("news", "")
        add_injury = _get_availability_indicator(add_chance, add_news)

        # Build rationale
        reasons = []
        add_form_col = "HealthyForm" if "HealthyForm" in add_row.index else "form"
        drop_form_col = "HealthyForm" if "HealthyForm" in drop_row.index else "form"
        form_diff = float(add_row.get(add_form_col, 0) or 0) - float(drop_row.get(drop_form_col, 0) or 0)
        if form_diff > 0:
            reasons.append(f"+{form_diff:.1f} form improvement")
        proj_add = pd.to_numeric(add_row.get("Projected_Points"), errors="coerce")
        proj_drop = pd.to_numeric(drop_row.get("Projected_Points"), errors="coerce")
        if pd.notna(proj_add) and pd.notna(proj_drop) and proj_add > proj_drop:
            reasons.append(f"+{proj_add - proj_drop:.1f} projected points")
        add_multi = float(add_row.get("MultiGW_Proj", 0) or 0)
        drop_multi = float(drop_row.get("MultiGW_Proj", 0) or 0)
        if add_multi > drop_multi and add_multi > 0:
            reasons.append(f"3GW outlook: {add_multi:.1f} vs {drop_multi:.1f} pts")
        add_fdr = add_row.get("AvgFDR")
        drop_fdr = drop_row.get("AvgFDR")
        if pd.notna(add_fdr) and pd.notna(drop_fdr) and add_fdr < drop_fdr:
            reasons.append("easier upcoming fixtures")
        if drop_news:
            reasons.append(f"current player: {drop_news[:40]}")

        rationale = " • ".join(reasons) if reasons else "Better overall transfer score"
        urgency = compute_transfer_urgency(pos, depth_map) if depth_map else ""

        add_form_val = float(add_row.get(add_form_col, 0) or 0)
        drop_form_val = float(drop_row.get(drop_form_col, 0) or 0)

        suggestions.append({
            "position": pos_labels.get(pos, pos),
            "score_diff": score_diff,
            "drop_player": drop_row["Player"],
            "drop_team": drop_row["Team"],
            "drop_price": f"£{drop_row['now_cost']/10:.1f}m",
            "drop_form": f"{drop_form_val:.1f}",
            "drop_season_pts": drop_row.get("total_points", 0),
            "drop_injury": drop_injury,
            "add_player": add_row["Player"],
            "add_team": add_row["Team"],
            "add_price": f"£{add_row['now_cost']/10:.1f}m",
            "add_form": f"{add_form_val:.1f}",
            "add_proj_pts": f"{proj_add:.1f}" if pd.notna(proj_add) else "N/A",
            "add_injury": add_injury,
            "rationale": rationale,
            "urgency": urgency,
        })

    return suggestions


def _render_depth_card(depth_map: Dict):
    """Render a compact horizontal positional depth summary."""
    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}
    level_colors = {'Critical': '#dc3545', 'Low': '#ff9800', 'Adequate': '#4ecca3'}

    items_html = []
    for pos_code in ['G', 'D', 'M', 'F']:
        depth = depth_map.get(pos_code)
        if depth is None or depth.total == 0:
            continue
        label = pos_labels.get(pos_code, pos_code)
        color = level_colors.get(depth.depth_level, '#888')
        # Three-state dots, uniform color from depth level
        # Use inline CSS circles for consistent sizing
        dot_full = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                    f'background:{color};margin:0 1px;vertical-align:middle;"></span>')
        dot_half = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                    f'background:linear-gradient(90deg,{color} 50%,transparent 50%);'
                    f'border:1.5px solid {color};box-sizing:border-box;'
                    f'margin:0 1px;vertical-align:middle;"></span>')
        dot_empty = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                     f'border:1.5px solid {color};box-sizing:border-box;'
                     f'margin:0 1px;vertical-align:middle;"></span>')
        dots_html = (dot_full * depth.healthy) + (dot_half * depth.doubtful) + (dot_empty * depth.injured)
        # Count doubtful as 0.5
        effective = depth.healthy + depth.doubtful * 0.5
        count_str = f"{effective:g}/{depth.total}"
        level_text = depth.depth_level if depth.depth_level != "Adequate" else ""
        level_span = (
            f'<span style="color:{color};font-weight:bold;font-size:0.8em;margin-left:4px;">'
            f'{level_text}</span>' if level_text else ""
        )
        items_html.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="font-weight:bold;color:#e0e0e0;">{label}</span>'
            f'<span style="color:#aaa;font-size:0.85em;">{count_str}</span>'
            f'<span style="display:inline-flex;align-items:center;">{dots_html}</span>'
            f'{level_span}'
            f'</div>'
        )

    if items_html:
        card = (
            '<div style="border:1px solid #444;border-radius:8px;padding:10px 16px;margin-bottom:10px;'
            'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#e0e0e0;'
            'display:flex;align-items:center;gap:24px;flex-wrap:wrap;">'
            '<span style="font-weight:bold;font-size:0.9em;color:#aaa;">Squad Depth</span>'
            + ''.join(items_html)
            + '</div>'
        )
        st.markdown(card, unsafe_allow_html=True)


def _render_transfer_suggestions(suggestions: List[Dict]):
    """Render transfer suggestion cards using styled HTML."""
    if not suggestions:
        st.info("No beneficial transfers found. Your squad looks strong at all positions.")
        return

    st.subheader("Transfer Suggestions")

    for s in suggestions:
        # Urgency badge
        urgency = s.get('urgency', '')
        urgency_html = ""
        if urgency == "URGENT":
            urgency_html = ('<span style="background:#dc3545;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">URGENT</span>')
        elif urgency == "LOW DEPTH":
            urgency_html = ('<span style="background:#ff9800;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">LOW DEPTH</span>')

        card_html = f"""
        <div style="border: 1px solid #444; border-radius: 10px; padding: 16px; margin-bottom: 12px;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div>
                    <span style="background: #0f3460; color: #e0e0e0; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">{s['position']}</span>{urgency_html}
                </div>
                <span style="background: #1a472a; color: #4ecca3; padding: 3px 12px; border-radius: 12px;
                             font-size: 0.85em; font-weight: bold;">+{s['score_diff']:.3f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="flex: 1;">
                    <div style="color: #e74c3c; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">DROP</div>
                    <div style="color: #e0e0e0; font-weight: bold;">{s['drop_player']} ({s['drop_team']})</div>
                    <div style="color: #999; font-size: 0.85em;">
                        Price: {s['drop_price']} &bull; Form: {s['drop_form']} (healthy) &bull;
                        Season: {s['drop_season_pts']} &bull; {s['drop_injury']}
                    </div>
                </div>
                <div style="color: #888; font-size: 1.5em; padding: 0 16px;">&rarr;</div>
                <div style="flex: 1; text-align: right;">
                    <div style="color: #4ecca3; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">ADD</div>
                    <div style="color: #e0e0e0; font-weight: bold;">{s['add_player']} ({s['add_team']})</div>
                    <div style="color: #999; font-size: 0.85em;">
                        Price: {s['add_price']} &bull; Proj: {s['add_proj_pts']} &bull;
                        Form: {s['add_form']} (healthy) &bull; {s['add_injury']}
                    </div>
                </div>
            </div>
            <div style="color: #aaa; font-size: 0.82em; font-style: italic; border-top: 1px solid #333;
                        padding-top: 6px;">{s['rationale']}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_classic_transfers_page():
    """Display the Classic FPL Transfers page."""

    st.title("Transfer Suggestions")
    st.caption("Find the best transfer targets based on projections, form, fixtures, and price.")

    # Check configuration
    team_id = config.FPL_CLASSIC_TEAM_ID
    if not team_id:
        st.warning("No Classic FPL team configured.")
        st.info(
            "Add your team ID to your `.env` file:\n\n"
            "```\nFPL_CLASSIC_TEAM_ID=123456\n```\n\n"
            "You can find your team ID in the URL when viewing your team on the FPL website."
        )
        return

    # Load data
    with st.spinner("Loading data..."):
        bootstrap = get_classic_bootstrap_static()
        entry = get_entry_details(team_id)
        current_gw = get_current_gameweek() or 1
        history = get_classic_team_history(team_id)

    if not bootstrap:
        show_api_error("loading player data for transfer analysis")
        return

    if not entry:
        show_api_error(f"loading team details for team ID {team_id}", hint_key="team_id")
        return

    # Team name (used in squad header below)
    team_name = entry.get("name", "Unknown Team")

    # Get current squad
    picks_data = get_classic_team_picks(team_id, current_gw)
    if not picks_data:
        # Try previous gameweek
        picks_data = get_classic_team_picks(team_id, current_gw - 1)

    if not picks_data:
        show_api_error("loading your current squad")
        return

    picks = picks_data.get("picks", [])
    entry_history = picks_data.get("entry_history", {})

    bank = entry_history.get("bank", 0)
    squad_value = entry_history.get("value", 0)
    transfers_made = entry_history.get("event_transfers", 0)
    transfer_cost = entry_history.get("event_transfers_cost", 0)

    # Controls
    with st.expander("Filters", expanded=True):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            pos_filter = st.multiselect(
                "Position Filter",
                ["GK", "DEF", "MID", "FWD"],
                default=["GK", "DEF", "MID", "FWD"]
            )

        with col_b:
            fdr_weeks = st.slider("FDR Lookahead (weeks)", 1, 8, 5)

        with col_c:
            max_price = st.slider(
                "Max Price",
                5.0, 15.0, 15.0, 0.5,
                format="£%.1fm"
            )

        st.caption("Scores use positional percentiles (0-1) against the full FPL pool. "
                   "A score of 0.85 = top 15% at this position. Weights auto-adjust by gameweek.")

    # Build DataFrames
    with st.spinner("Analyzing players..."):
        # Build all players DataFrame
        all_players = _build_all_players_df(bootstrap, current_gw, fdr_weeks)

        # Build squad DataFrame
        squad_df = _build_squad_df(picks, bootstrap, entry_history)
        squad_df["AvgFDR"] = squad_df["Team_ID"].apply(
            lambda t: _avg_fdr_for_team(t, current_gw, fdr_weeks)
        )

        # Load projections
        projections_df = None
        try:
            rotowire_url = config.ROTOWIRE_URL
            if rotowire_url:
                projections_df = get_rotowire_player_projections(rotowire_url)
            else:
                st.warning(
                    "⚠️ Rotowire player projections are unavailable — the app could not discover the current "
                    "rankings article URL. Rotowire may have changed their URL format. "
                    "To fix immediately, add `ROTOWIRE_URL=<article URL>` to your `.env` file and restart the app."
                )
        except Exception as e:
            st.warning(f"Could not load projections: {e}")

        # Add projections
        all_players = _add_projections(all_players, projections_df)
        squad_df = _add_projections(squad_df, projections_df)

        # Compute healthy form for squad players (15 players — fast)
        for idx, row in squad_df.iterrows():
            pid = row.get("Player_ID")
            if pd.notna(pid):
                hf = compute_healthy_form(int(pid), element_history_fn=_fetch_element_history)
                squad_df.at[idx, "HealthyForm"] = hf if hf is not None else row.get("form", 0)
            else:
                squad_df.at[idx, "HealthyForm"] = row.get("form", 0)

        # Load FFP multi-GW projections
        try:
            ffp_df = get_ffp_projections_data()
        except Exception:
            ffp_df = None

        # Add multi-GW projections
        squad_df = blend_multi_gw_projections(
            squad_df, ffp_df, single_gw_col="Projected_Points"
        )
        all_players = blend_multi_gw_projections(
            all_players, ffp_df, single_gw_col="Projected_Points"
        )

        # Rotowire Season Projections
        try:
            season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
        except Exception:
            season_rankings_df = None
        squad_df = merge_season_projections(squad_df, season_rankings_df)
        all_players = merge_season_projections(all_players, season_rankings_df)

        # FFP Single-GW Data (Predicted, Start, LongStart)
        squad_df = merge_ffp_single_gw_data(squad_df, ffp_df)
        all_players = merge_ffp_single_gw_data(all_players, ffp_df)

    # Positional depth
    depth_map = {}
    if not squad_df.empty:
        # Map status from bootstrap for depth calculation
        elements_status = {p["id"]: p for p in bootstrap.get("elements", [])}
        for idx, row in squad_df.iterrows():
            pid = row.get("Player_ID")
            el = elements_status.get(pid, {})
            if "status" not in squad_df.columns:
                squad_df["status"] = None
            squad_df.at[idx, "status"] = el.get("status", "a")
        depth_map = compute_positional_depth(squad_df)

    # Compute keep scores (after depth is known)
    if not squad_df.empty:
        squad_df = _compute_keep_score(squad_df, all_players_df=all_players,
                                       current_gw=current_gw, depth_map=depth_map)

    # Get squad player IDs for filtering
    squad_ids = set(squad_df["Player_ID"].tolist())
    teams_map = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    # Position mapping for filter
    pos_map = {"GK": "G", "DEF": "D", "MID": "M", "FWD": "F"}
    filter_positions = [pos_map.get(p, p) for p in pos_filter]

    # Filter available players
    available = all_players[
        (~all_players["Player_ID"].isin(squad_ids)) &
        (all_players["Position"].isin(filter_positions)) &
        (all_players["now_cost"] <= max_price * 10) &
        (all_players["minutes"] > 0)  # Must have played this season
    ].copy()

    # Compute healthy form for top transfer candidates (selective — not all 600+ players)
    top_candidates = available.nlargest(50, "Projected_Points", keep="all") if "Projected_Points" in available.columns and available["Projected_Points"].notna().any() else available.head(50)
    for idx in top_candidates.index:
        pid = available.at[idx, "Player_ID"]
        if pd.notna(pid):
            hf = compute_healthy_form(int(pid), element_history_fn=_fetch_element_history)
            available.at[idx, "HealthyForm"] = hf if hf is not None else available.at[idx, "form"]
        else:
            available.at[idx, "HealthyForm"] = available.at[idx, "form"]
    # Fill remaining candidates with FPL form
    if "HealthyForm" not in available.columns:
        available["HealthyForm"] = available["form"]
    available["HealthyForm"] = available["HealthyForm"].fillna(available["form"])

    # Compute transfer score
    available = _compute_transfer_score(available, all_players_df=all_players, current_gw=current_gw)
    available = available.sort_values("Transfer Score", ascending=False)

    # ---------------------------
    # TRANSFER SUGGESTION CARDS
    # ---------------------------
    suggestions = _build_transfer_suggestions(squad_df, available, bank, top_n=3, depth_map=depth_map)
    _render_transfer_suggestions(suggestions)

    st.markdown("---")

    # ---------------------------
    # SQUAD ANALYSIS SECTION (with depth card)
    # ---------------------------
    st.header(f"Your Squad — {team_name}")

    # Stat cards for bank, value, transfers
    def _stat_card(label: str, value: str, accent: str = "#00ff87") -> str:
        return (
            f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
            f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
            f'</div>'
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(_stat_card("Bank", _format_money(bank)), unsafe_allow_html=True)
    with col2:
        st.markdown(_stat_card("Squad Value", _format_money(squad_value)), unsafe_allow_html=True)
    with col3:
        st.markdown(_stat_card("Transfers Made", str(transfers_made)), unsafe_allow_html=True)
    with col4:
        cost_text = f"-{transfer_cost} pts" if transfer_cost else "0 pts"
        cost_color = "#f87171" if transfer_cost else "#00ff87"
        st.markdown(_stat_card("Transfer Cost", cost_text, accent=cost_color), unsafe_allow_html=True)

    st.markdown("")  # spacing between stat cards and depth card

    if depth_map:
        _render_depth_card(depth_map)

    # Split into starting XI and bench
    starting_xi = squad_df[squad_df["squad_position"] <= 11].copy()
    bench = squad_df[squad_df["squad_position"] > 11].copy()

    # Show squad with keep scores (sort by Keep Score for holistic view)
    squad_display = squad_df.sort_values("Keep Score", ascending=True).copy()

    # Add fixtures for each player
    fixture_html_list = []
    for _, row in squad_display.iterrows():
        fixtures = _get_team_fixtures(row["Team_ID"], fdr_weeks, current_gw)
        fixture_html_list.append(_format_fixtures_html(fixtures, teams_map, 5))
    squad_display["Fixtures"] = fixture_html_list

    # Add availability
    squad_display["Status"] = squad_display.apply(
        lambda r: _get_availability_indicator(r["chance_of_playing_next_round"], r["news"]),
        axis=1
    )

    # Add positional rank
    squad_display["Pos_Rank"] = positional_rank(
        squad_display, all_players, "total_points", ref_value_col="total_points"
    )

    # Format display columns — use HealthyForm if available
    form_display_col = "HealthyForm" if "HealthyForm" in squad_display.columns else "form"
    display_cols = ["Player", "Team", "Position", "Pos_Rank", "now_cost", form_display_col, "total_points",
                    "Projected_Points", "AvgFDR", "1GW", "ROS", "Keep Score", "Status"]
    display_cols = [c for c in display_cols if c in squad_display.columns]
    squad_show = squad_display[display_cols].copy()
    squad_show["Price"] = squad_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
    squad_show["AvgFDR"] = squad_show["AvgFDR"].round(2)
    for sc in ["1GW", "ROS", "Keep Score"]:
        if sc in squad_show.columns:
            squad_show[sc] = squad_show[sc].round(3)
    squad_show["Projected_Points"] = squad_show["Projected_Points"].fillna("-")

    # Rename for display
    squad_show = squad_show.rename(columns={
        form_display_col: "Form",
        "total_points": "Season Pts",
        "Projected_Points": "Proj Pts",
        "Pos_Rank": "Pos Rank",
        "AvgFDR": "Avg FDR",
    })

    render_styled_table(
        squad_show[["Player", "Team", "Position", "Pos Rank", "Price", "Form", "Season Pts",
                    "Proj Pts", "Avg FDR", "1GW", "ROS", "Keep Score", "Status"]],
        col_formats={"Form": "{:.1f}", "Avg FDR": "{:.2f}", "1GW": "{:.3f}", "ROS": "{:.3f}", "Keep Score": "{:.3f}"},
        positive_color_cols=["1GW", "ROS", "Keep Score"],
    )

    # ---------------------------
    # TRANSFER TARGETS SECTION
    # ---------------------------
    st.header("Transfer Targets")

    # Show top targets
    st.subheader("Top Transfer Targets (All Positions)")

    # Prepare display DataFrame
    top_targets = available.head(20).copy()

    # Add fixtures
    target_fixture_list = []
    for _, row in top_targets.iterrows():
        fixtures = _get_team_fixtures(row["Team_ID"], fdr_weeks, current_gw)
        target_fixture_list.append(_format_fixtures_html(fixtures, teams_map, 5))
    top_targets["Fixtures"] = target_fixture_list

    # Add availability
    top_targets["Status"] = top_targets.apply(
        lambda r: _get_availability_indicator(r["chance_of_playing_next_round"], r["news"]),
        axis=1
    )

    # Add price change indicator
    top_targets["Price_Change"] = top_targets["cost_change_event"].apply(_format_price_change)

    # Format for display — use HealthyForm if available
    target_form_col = "HealthyForm" if "HealthyForm" in top_targets.columns else "form"
    target_display_cols = [
        "Player", "Team", "Position", "now_cost", "Price_Change", target_form_col,
        "total_points", "Projected_Points", "selected_by_percent",
        "AvgFDR", "1GW", "ROS", "Transfer Score", "Status"
    ]
    target_display_cols = [c for c in target_display_cols if c in top_targets.columns]
    targets_show = top_targets[target_display_cols].copy()

    targets_show["Price"] = targets_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
    targets_show["AvgFDR"] = targets_show["AvgFDR"].round(2)
    for sc in ["1GW", "ROS", "Transfer Score"]:
        if sc in targets_show.columns:
            targets_show[sc] = targets_show[sc].round(3)
    targets_show["Projected_Points"] = targets_show["Projected_Points"].fillna("-")
    targets_show["Ownership"] = targets_show["selected_by_percent"].apply(lambda x: f"{x:.1f}%")

    # Rename for display
    targets_show = targets_show.rename(columns={
        target_form_col: "Form",
        "total_points": "Season Pts",
        "Projected_Points": "Proj Pts",
        "AvgFDR": "Avg FDR",
        "Price_Change": "Δ",
    })

    render_styled_table(
        targets_show[["Player", "Team", "Position", "Price", "Δ", "Form",
                      "Season Pts", "Proj Pts", "Ownership", "Avg FDR", "1GW", "ROS", "Transfer Score", "Status"]],
        col_formats={"Form": "{:.1f}", "Avg FDR": "{:.2f}", "1GW": "{:.3f}", "ROS": "{:.3f}", "Transfer Score": "{:.3f}"},
        positive_color_cols=["1GW", "ROS", "Transfer Score"],
        max_height=500,
    )

    st.markdown("---")

    # ---------------------------
    # POSITION-SPECIFIC TARGETS
    # ---------------------------
    st.subheader("Position-Specific Targets")

    tabs = st.tabs(["Goalkeepers", "Defenders", "Midfielders", "Forwards"])

    position_codes = {"Goalkeepers": "G", "Defenders": "D", "Midfielders": "M", "Forwards": "F"}

    for tab, (pos_name, pos_code) in zip(tabs, position_codes.items()):
        with tab:
            pos_targets = available[available["Position"] == pos_code].head(10).copy()

            if pos_targets.empty:
                st.info(f"No {pos_name.lower()} match your filter criteria.")
                continue

            # Add fixtures
            pos_fixture_list = []
            for _, row in pos_targets.iterrows():
                fixtures = _get_team_fixtures(row["Team_ID"], fdr_weeks, current_gw)
                pos_fixture_list.append(_format_fixtures_html(fixtures, teams_map, 5))
            pos_targets["Fixtures"] = pos_fixture_list

            # Format for display
            pos_cols = ["Player", "Team", "now_cost", "form", "total_points",
                        "Projected_Points", "selected_by_percent", "AvgFDR",
                        "1GW", "ROS", "Transfer Score"]
            pos_cols = [c for c in pos_cols if c in pos_targets.columns]
            pos_show = pos_targets[pos_cols].copy()

            pos_show["Price"] = pos_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
            pos_show["AvgFDR"] = pos_show["AvgFDR"].round(2)
            for sc in ["1GW", "ROS", "Transfer Score"]:
                if sc in pos_show.columns:
                    pos_show[sc] = pos_show[sc].round(3)
            pos_show["Projected_Points"] = pos_show["Projected_Points"].fillna("-")
            pos_show["Own%"] = pos_show["selected_by_percent"].apply(lambda x: f"{x:.1f}%")

            pos_display_cols = ["Player", "Team", "Price", "form", "total_points",
                          "Projected_Points", "Own%", "AvgFDR", "1GW", "ROS", "Transfer Score"]
            pos_display_cols = [c for c in pos_display_cols if c in pos_show.columns]
            pos_display = pos_show[pos_display_cols].copy()
            pos_display = pos_display.rename(columns={
                "form": "Form", "total_points": "Season Pts",
                "Projected_Points": "Proj Pts", "AvgFDR": "Avg FDR",
            })
            render_styled_table(
                pos_display,
                col_formats={"Form": "{:.1f}", "Avg FDR": "{:.2f}", "1GW": "{:.3f}", "ROS": "{:.3f}", "Transfer Score": "{:.3f}"},
                positive_color_cols=["1GW", "ROS", "Transfer Score"],
            )

    st.markdown("---")

    # ---------------------------
    # TRANSFER COMPARISON TOOL
    # ---------------------------
    st.header("Transfer Comparison")
    st.caption("Compare a player from your squad with potential replacements.")

    col_out, col_in = st.columns(2)

    with col_out:
        st.subheader("Transfer Out")
        squad_options = squad_df["Player"].tolist()
        selected_out = st.selectbox("Select player to transfer out", squad_options)

        if selected_out:
            out_player = squad_df[squad_df["Player"] == selected_out].iloc[0]
            st.markdown(f"**{out_player['Player']}** ({out_player['Team']})")
            st.caption(f"Position: {out_player['Position']} | Price: £{out_player['now_cost']/10:.1f}m")
            st.caption(f"Form: {out_player['form']:.1f} | Season Pts: {out_player['total_points']}")

            fixtures = _get_team_fixtures(out_player["Team_ID"], fdr_weeks, current_gw)
            st.markdown("**Upcoming fixtures:**")
            st.markdown(_format_fixtures_html(fixtures, teams_map, 6), unsafe_allow_html=True)

    with col_in:
        st.subheader("Transfer In")

        if selected_out:
            out_player = squad_df[squad_df["Player"] == selected_out].iloc[0]
            out_pos = out_player["Position"]
            selling_price = out_player["selling_price"]

            # Calculate budget
            budget = bank + selling_price

            # Filter replacements
            replacements = available[
                (available["Position"] == out_pos) &
                (available["now_cost"] <= budget)
            ].head(20)

            if replacements.empty:
                st.warning("No affordable replacements found.")
            else:
                in_options = replacements["Player"].tolist()
                selected_in = st.selectbox("Select replacement", in_options)

                if selected_in:
                    in_player = replacements[replacements["Player"] == selected_in].iloc[0]
                    st.markdown(f"**{in_player['Player']}** ({in_player['Team']})")
                    st.caption(f"Position: {in_player['Position']} | Price: £{in_player['now_cost']/10:.1f}m")
                    st.caption(f"Form: {in_player['form']:.1f} | Season Pts: {in_player['total_points']}")
                    st.caption(f"Ownership: {in_player['selected_by_percent']:.1f}%")

                    fixtures = _get_team_fixtures(in_player["Team_ID"], fdr_weeks, current_gw)
                    st.markdown("**Upcoming fixtures:**")
                    st.markdown(_format_fixtures_html(fixtures, teams_map, 6), unsafe_allow_html=True)

                    # Show comparison summary
                    st.markdown("---")
                    st.markdown("**Transfer Summary:**")
                    cost_diff = in_player["now_cost"] - selling_price
                    if cost_diff > 0:
                        st.caption(f"Cost: +£{cost_diff/10:.1f}m (Budget: £{budget/10:.1f}m)")
                    else:
                        st.caption(f"Cost: -£{abs(cost_diff)/10:.1f}m (saves money)")

                    form_diff = in_player["form"] - out_player["form"]
                    st.caption(f"Form change: {'+' if form_diff >= 0 else ''}{form_diff:.1f}")

                    if pd.notna(in_player["Projected_Points"]) and pd.notna(out_player.get("Projected_Points")):
                        proj_diff = in_player["Projected_Points"] - out_player.get("Projected_Points", 0)
                        st.caption(f"Projected points change: {'+' if proj_diff >= 0 else ''}{proj_diff:.1f}")

    st.markdown("---")

    # ---------------------------
    # RECENT TRANSFERS SECTION
    # ---------------------------
    st.header("Your Recent Transfers")

    transfers = get_classic_transfers(team_id)
    if transfers:
        elements = {p["id"]: p for p in bootstrap.get("elements", [])}

        recent = transfers[:10]  # Last 10 transfers

        transfer_rows = []
        for t in recent:
            in_id = t.get("element_in")
            out_id = t.get("element_out")
            in_player = elements.get(in_id, {})
            out_player = elements.get(out_id, {})

            transfer_rows.append({
                "GW": t.get("event", "?"),
                "Out": out_player.get("web_name", "Unknown"),
                "In": in_player.get("web_name", "Unknown"),
                "In Cost": f"£{t.get('element_in_cost', 0)/10:.1f}m",
                "Out Cost": f"£{t.get('element_out_cost', 0)/10:.1f}m",
            })

        transfers_df = pd.DataFrame(transfer_rows)
        render_styled_table(transfers_df, text_align={"GW": "center"})
    else:
        st.info("No transfers found for this season.")
