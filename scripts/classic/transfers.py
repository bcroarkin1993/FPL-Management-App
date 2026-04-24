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


def _parse_chip_status(history: dict, current_gw: int) -> dict:
    """Parse chip usage and availability from team history.

    Both Wildcard and Bench Boost are now double-use chips:
      - Slot 1: used before GW20 (event < 20)
      - Slot 2: used in GW20+ (event >= 20)
    Free Hit and Triple Captain remain single-use.
    """
    chips_used = history.get("chips", []) if history else []

    # Wildcard and Bench Boost are tracked as lists (double-use)
    used: Dict[str, Any] = {"wildcard": [], "bboost": [], "freehit": None, "3xc": None}
    for chip in chips_used:
        name = chip.get("name", "")
        event = chip.get("event", 0)
        if name in ("wildcard", "bboost"):
            used[name].append(event)
        elif name in used:
            used[name] = event

    def _double_chip_available(events: list) -> bool:
        slot1_used = any(e < 20 for e in events)
        slot2_used = any(e >= 20 for e in events)
        return not slot1_used or not slot2_used

    wildcard_1_used = any(e < 20 for e in used["wildcard"])
    wildcard_2_used = any(e >= 20 for e in used["wildcard"])
    wildcard_available = _double_chip_available(used["wildcard"])

    bboost_1_used = any(e < 20 for e in used["bboost"])
    bboost_2_used = any(e >= 20 for e in used["bboost"])
    bboost_available = _double_chip_available(used["bboost"])

    available = []
    if wildcard_available:
        available.append("wildcard")
    if bboost_available:
        available.append("bboost")
    if used["freehit"] is None:
        available.append("freehit")
    if used["3xc"] is None:
        available.append("3xc")

    return {
        "used": used,
        "available": available,
        "wildcard_1_used": wildcard_1_used,
        "wildcard_2_used": wildcard_2_used,
        "wildcard_available": wildcard_available,
        "bboost_1_used": bboost_1_used,
        "bboost_2_used": bboost_2_used,
        "bboost_available": bboost_available,
    }


def _compute_free_transfers(history: dict, entry_history: dict, current_gw: int) -> int:
    """Compute free transfers available this gameweek.

    FPL rules: 1 FT per GW, bank 1 extra if 0 transfers last GW (max 2 banked).
    """
    if not history:
        return 1

    gw_history = history.get("current", [])

    # Check if a hit was taken this GW already
    transfer_cost = entry_history.get("event_transfers_cost", 0)
    if transfer_cost and transfer_cost < 0:
        return 0

    # history["current"] only contains completed GWs — [-1] is the last finished GW.
    # If 0 transfers were made last GW, 1 FT was banked → 2 FTs available now.
    if len(gw_history) >= 1:
        last_gw = gw_history[-1]
        if last_gw.get("event_transfers", 1) == 0:
            return 2  # Banked from last GW

    return 1


def _ownership_badge(pct: float) -> str:
    """Return HTML ownership badge for template (>20%) or differential (<5%) players."""
    pct = float(pct or 0)
    if pct >= 20:
        return ('<span style="background:#7c3aed;color:#fff;padding:2px 8px;border-radius:10px;'
                'font-size:0.75em;font-weight:bold;">Template</span>')
    elif pct <= 5:
        return ('<span style="background:#0891b2;color:#fff;padding:2px 8px;border-radius:10px;'
                'font-size:0.75em;font-weight:bold;">Differential</span>')
    return ""


def _format_price_trend(cost_change_event: int, transfers_in_event: int,
                         transfers_out_event: int) -> dict:
    """Format price trend badge from GW transfer activity."""
    cost_change = int(cost_change_event or 0)
    t_in = int(transfers_in_event or 0)
    t_out = int(transfers_out_event or 0)

    if cost_change > 0:
        indicator = "Rising"
        badge_html = (f'<span style="color:#4ecca3;font-size:0.78em;font-weight:bold;">'
                      f'&#8593; Rising +£{cost_change/10:.1f}m</span>')
    elif cost_change < 0:
        indicator = "Falling"
        badge_html = (f'<span style="color:#f87171;font-size:0.78em;font-weight:bold;">'
                      f'&#8595; Falling £{cost_change/10:.1f}m</span>')
    elif t_in > t_out * 1.5 and t_in > 50_000:
        indicator = "Likely Rising"
        badge_html = '<span style="color:#86efac;font-size:0.78em;">&#8593; Likely Rising</span>'
    else:
        indicator = "Stable"
        badge_html = ""

    rush_html = ""
    if t_in > 200_000:
        rush_html = ('<span style="background:#d97706;color:#fff;padding:1px 6px;border-radius:8px;'
                     'font-size:0.72em;font-weight:bold;margin-left:4px;">Transfer Rush</span>')

    return {"indicator": indicator, "badge_html": badge_html, "rush_html": rush_html}


def _compute_hit_verdict(ep_delta: float, is_hit: bool) -> dict:
    """Determine whether a transfer is worth a -4 hit based on FPL expected points delta."""
    net_gain = ep_delta - 4.0 if is_hit else ep_delta

    if not is_hit:
        verdict = "FREE"
        verdict_color = "#4ecca3"
    elif net_gain >= 2.0:
        verdict = "YES"
        verdict_color = "#4ecca3"
    elif net_gain >= 0:
        verdict = "MARGINAL"
        verdict_color = "#fbbf24"
    else:
        verdict = "NO"
        verdict_color = "#f87171"

    sign = "+" if net_gain >= 0 else ""
    hit_label = " (hit)" if is_hit else " (free)"
    display_str = f"{sign}{net_gain:.1f} pts net{hit_label}"

    return {
        "net_gain": net_gain,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "display_str": display_str,
    }


def _render_transfer_status_panel(bank: int, squad_value: int, free_transfers: int,
                                   chip_status: dict, active_chip: Optional[str]):
    """Render a top-of-page status panel: free transfers, bank, chips."""
    chip_names = {"wildcard": "Wildcard", "bboost": "Bench Boost",
                  "freehit": "Free Hit", "3xc": "Triple Captain"}
    chip_colors = {"wildcard": "#7c3aed", "bboost": "#166534",
                   "freehit": "#0891b2", "3xc": "#d97706"}

    def stat_card_html(label: str, value: str, accent: str = "#00ff87", subtitle: str = "") -> str:
        sub = (f'<div style="color:#aaa;font-size:11px;margin-top:4px;">{subtitle}</div>'
               if subtitle else "")
        return (
            f'<div style="border:1px solid #333;border-radius:10px;padding:14px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            f'text-align:center;color:#e0e0e0;height:100%;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
            f'<div style="color:{accent};font-size:20px;font-weight:700;">{value}</div>'
            f'{sub}</div>'
        )

    # Free transfer card
    if free_transfers == 2:
        ft_val, ft_color, ft_sub = "2 FTs Banked", "#4ecca3", "Both transfers are free"
    elif free_transfers == 1:
        ft_val, ft_color, ft_sub = "1 Free Transfer", "#00ff87", "Next costs &minus;4 pts"
    else:
        ft_val, ft_color, ft_sub = "0 FTs (Hit GW)", "#f87171", "&minus;4 pts per transfer"

    # Active chip card
    if active_chip:
        active_val = chip_names.get(active_chip, active_chip)
        active_color = chip_colors.get(active_chip, "#fbbf24")
        active_sub = "Currently active!"
    else:
        active_val, active_color, active_sub = "None Active", "#6b7280", ""

    # Available chips card (with pill badges)
    avail = chip_status.get("available", [])
    if avail:
        badges = "".join(
            f'<span style="background:{chip_colors.get(c, "#444")};color:#fff;'
            f'padding:2px 8px;border-radius:10px;font-size:0.72em;font-weight:bold;margin:2px;">'
            f'{chip_names.get(c, c)}</span>'
            for c in avail
        )
        chips_card = (
            f'<div style="border:1px solid #333;border-radius:10px;padding:14px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            f'text-align:center;color:#e0e0e0;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:8px;">Chips Available</div>'
            f'<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:4px;">{badges}</div>'
            f'</div>'
        )
    else:
        chips_card = stat_card_html("Chips Available", "All Used", "#6b7280")

    cols = st.columns(5)
    with cols[0]:
        st.markdown(stat_card_html("Free Transfers", ft_val, ft_color, ft_sub),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(stat_card_html("In the Bank", _format_money(bank)), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(stat_card_html("Squad Value", _format_money(squad_value)),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(stat_card_html("Active Chip", active_val, active_color, active_sub),
                    unsafe_allow_html=True)
    with cols[4]:
        st.markdown(chips_card, unsafe_allow_html=True)

    st.markdown("")  # spacing


def _render_chip_advisor(chip_status: dict, squad_df: pd.DataFrame, current_gw: int):
    """Render chip strategy advisor with rule-based advice."""
    chip_names = {"wildcard": "Wildcard", "bboost": "Bench Boost",
                  "freehit": "Free Hit", "3xc": "Triple Captain"}
    chip_colors = {"wildcard": "#7c3aed", "bboost": "#166534",
                   "freehit": "#0891b2", "3xc": "#d97706"}
    chip_used_events = chip_status.get("used", {})

    with st.expander("Chip Strategy", expanded=False):
        # Chip status display
        avail = chip_status.get("available", [])
        all_chips = ["wildcard", "bboost", "freehit", "3xc"]

        avail_html = ""
        for c in all_chips:
            if c in avail:
                avail_html += (
                    f'<span style="background:{chip_colors.get(c, "#444")};color:#fff;'
                    f'padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:bold;margin:3px;">'
                    f'{chip_names.get(c, c)}</span>'
                )
            else:
                events = chip_used_events.get(c)
                if isinstance(events, list):
                    used_label = f'GW{events[0]}' if events else '?'
                else:
                    used_label = f'GW{events}' if events else 'Used'
                avail_html += (
                    f'<span style="background:#2d2d2d;color:#666;'
                    f'padding:3px 10px;border-radius:12px;font-size:0.8em;margin:3px;">'
                    f'{chip_names.get(c, c)} ({used_label})</span>'
                )

        st.markdown(
            f'<div style="margin-bottom:10px;color:#e0e0e0;">{avail_html}</div>',
            unsafe_allow_html=True
        )

        # Rule-based advice
        if "Keep Score" in squad_df.columns and not squad_df.empty:
            avg_keep = float(squad_df["Keep Score"].mean())
            weak_count = int((squad_df["Keep Score"] < 0.50).sum())

            if chip_status["wildcard_available"] and current_gw >= 25 and avg_keep < 0.45:
                st.warning(
                    f"**Wildcard Alert:** Your squad's avg Keep Score is {avg_keep:.2f} (below 0.45). "
                    "Consider using your Wildcard to rebuild with a stronger set of players."
                )
            elif chip_status["wildcard_available"] and weak_count >= 6:
                st.warning(
                    f"**Wildcard Alert:** {weak_count} players have Keep Score below 0.50. "
                    "A Wildcard rebuild could significantly improve your squad quality."
                )

            if not chip_status["wildcard_2_used"] and current_gw >= 30:
                st.info(
                    f"**WC2 available** — your second Wildcard has not been used yet. "
                    "Best deployed in GW30–35 for the run-in."
                )

        if not avail:
            st.success("All chips used — focus on optimizing weekly transfers.")


def _build_multi_transfer_plan(squad_df: pd.DataFrame, available_df: pd.DataFrame,
                                bank: int, depth_map: Optional[Dict] = None) -> List[Dict]:
    """Find optimal 2-player swap when both transfers are free."""
    if squad_df.empty or available_df.empty or "Keep Score" not in squad_df.columns:
        return []

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}

    # Bottom-6 Keep Score as drop candidates
    drop_candidates = squad_df.nsmallest(6, "Keep Score")

    best_score = -999.0
    best_plan: List[Dict] = []

    drop_list = list(drop_candidates.iterrows())
    for i, (_, drop1) in enumerate(drop_list):
        for _, drop2 in drop_list[i + 1:]:
            if drop1["Player_ID"] == drop2["Player_ID"]:
                continue

            combined_budget = bank + drop1.get("selling_price", drop1.get("now_cost", 0)) + \
                              drop2.get("selling_price", drop2.get("now_cost", 0))

            # Find best add for each drop position independently
            pair_suggestions = []
            for drop_row in [drop1, drop2]:
                pos = drop_row["Position"]
                cands = available_df[
                    (available_df["Position"] == pos) &
                    (available_df["now_cost"] <= combined_budget)
                ].head(10)

                found = None
                for _, add_row in cands.iterrows():
                    # Club rule check
                    squad_without = squad_df[
                        ~squad_df["Player_ID"].isin([drop1["Player_ID"], drop2["Player_ID"]])
                    ]
                    add_team = add_row.get("Team")
                    if (squad_without["Team"] == add_team).sum() >= 3:
                        continue
                    found = add_row
                    break

                if found is None:
                    break

                add_form_col = "HealthyForm" if "HealthyForm" in found.index else "form"
                proj = pd.to_numeric(found.get("Projected_Points"), errors="coerce")
                pair_suggestions.append({
                    "position": pos_labels.get(pos, pos),
                    "score_diff": float(found.get("Transfer Score", 0)) - float(drop_row.get("Keep Score", 0)),
                    "drop_player": drop_row["Player"],
                    "drop_team": drop_row["Team"],
                    "drop_price": f"£{drop_row['now_cost']/10:.1f}m",
                    "drop_form": f"{float(drop_row.get('HealthyForm' if 'HealthyForm' in drop_row.index else 'form', 0) or 0):.1f}",
                    "drop_season_pts": drop_row.get("total_points", 0),
                    "drop_injury": _get_availability_indicator(
                        drop_row.get("chance_of_playing_next_round"), drop_row.get("news", "")),
                    "add_player": found["Player"],
                    "add_team": found["Team"],
                    "add_price": f"£{found['now_cost']/10:.1f}m",
                    "add_form": f"{float(found.get(add_form_col, 0) or 0):.1f}",
                    "add_proj_pts": f"{proj:.1f}" if pd.notna(proj) else "N/A",
                    "add_injury": _get_availability_indicator(
                        found.get("chance_of_playing_next_round"), found.get("news", "")),
                    "rationale": "Part of optimal 2-transfer plan",
                    "urgency": compute_transfer_urgency(pos, depth_map) if depth_map else "",
                    "ep_delta": None,
                    "price_trend": None,
                    "add_ownership_badge": _ownership_badge(found.get("selected_by_percent", 0)),
                    "hit_verdict": None,
                    "plan_label": "2-Transfer Plan (Both Free)",
                })

            if len(pair_suggestions) == 2:
                combined_score = sum(s["score_diff"] for s in pair_suggestions)
                if combined_score > best_score:
                    best_score = combined_score
                    best_plan = pair_suggestions

    return best_plan


def _render_multi_transfer_plan(plan: List[Dict]):
    """Render the optimal 2-transfer plan side-by-side."""
    if not plan:
        return

    st.subheader("2-Transfer Plan (Both Free)")
    st.caption("Optimal pair of transfers when you have 2 free transfers banked.")

    cols = st.columns(2)
    for col, s in zip(cols, plan):
        with col:
            card_html = f"""
            <div style="border: 1px solid #444; border-radius: 10px; padding: 16px; margin-bottom: 12px;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="background: #0f3460; color: #e0e0e0; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">{s['position']}</span>
                    <span style="background: #1a472a; color: #4ecca3; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">+{s['score_diff']:.3f}</span>
                </div>
                <div style="color: #e74c3c; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">DROP</div>
                <div style="color: #e0e0e0; font-weight: bold;">{s['drop_player']} ({s['drop_team']})</div>
                <div style="color: #999; font-size: 0.82em; margin-bottom: 8px;">
                    {s['drop_price']} &bull; Form: {s['drop_form']} &bull; {s['drop_injury']}
                </div>
                <div style="color: #4ecca3; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">ADD</div>
                <div style="color: #e0e0e0; font-weight: bold;">{s['add_player']} ({s['add_team']})</div>
                <div style="color: #999; font-size: 0.82em;">
                    {s['add_price']} &bull; Proj: {s['add_proj_pts']} &bull; {s['add_injury']}
                    {s.get('add_ownership_badge', '')}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    st.caption("Both transfers are free this gameweek — optimal pair based on Transfer/Keep Scores.")


def _build_transfer_suggestions(squad_df: pd.DataFrame, available_df: pd.DataFrame,
                                 bank: int, top_n: int = 3, depth_map: Optional[Dict] = None,
                                 free_transfers: int = 1) -> List[Dict]:
    """Build transfer suggestions pairing lowest-keep-score squad players with best replacements."""
    if squad_df.empty or available_df.empty:
        return []

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}
    suggestions = []
    remaining_ft = free_transfers

    # Pick at most one drop candidate per position — prevents e.g. two GK suggestions
    # both recommending the same replacement (like Areola twice).
    seen_positions: set = set()
    drop_candidates = []
    for _, row in squad_df.nsmallest(top_n * 4, "Keep Score").iterrows():
        pos = row["Position"]
        if pos not in seen_positions:
            seen_positions.add(pos)
            drop_candidates.append(row)
        if len(drop_candidates) == top_n:
            break

    for drop_row in drop_candidates:
        pos = drop_row["Position"]
        drop_id = drop_row["Player_ID"]
        selling_price = drop_row.get("selling_price", drop_row.get("now_cost", 0))
        budget = bank + selling_price

        # Find best replacement at same position within budget
        candidates = available_df[
            (available_df["Position"] == pos) &
            (available_df["now_cost"] <= budget)
        ].copy()

        if candidates.empty:
            continue

        # Club rule: max 3 players from same club — skip candidates that would breach it
        squad_without_drop = squad_df[squad_df["Player_ID"] != drop_id]
        add_row = None
        for _, candidate in candidates.head(10).iterrows():
            cand_team = candidate.get("Team")
            if (squad_without_drop["Team"] == cand_team).sum() >= 3:
                continue  # would violate 3-per-club rule
            add_row = candidate
            break

        if add_row is None:
            add_row = candidates.iloc[0]  # Fallback if all top-10 breach rule

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

        # ep_next delta (FPL's own expected points signal)
        ep_next_add = float(add_row.get("ep_next", 0) or 0)
        ep_next_drop = float(drop_row.get("ep_next", 0) or 0)
        ep_delta = ep_next_add - ep_next_drop

        # Price trend and ownership intelligence
        price_trend = _format_price_trend(
            add_row.get("cost_change_event", 0),
            add_row.get("transfers_in_event", 0),
            add_row.get("transfers_out_event", 0),
        )
        add_ownership_badge = _ownership_badge(add_row.get("selected_by_percent", 0))
        add_ownership_pct = float(add_row.get("selected_by_percent", 0) or 0)

        # Hit verdict
        is_hit = remaining_ft <= 0
        hit_verdict = _compute_hit_verdict(ep_delta, is_hit)
        remaining_ft = max(0, remaining_ft - 1)

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
            "ep_delta": ep_delta,
            "ep_next_add": ep_next_add,
            "ep_next_drop": ep_next_drop,
            "price_trend": price_trend,
            "add_ownership_badge": add_ownership_badge,
            "add_ownership_pct": add_ownership_pct,
            "hit_verdict": hit_verdict,
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


def _build_hit_verdict_row(s: dict) -> str:
    """Build the hit verdict HTML row for a suggestion card."""
    verdict_data = s.get("hit_verdict")
    if not verdict_data:
        return ""
    ep_add = s.get("ep_next_add", 0)
    ep_drop = s.get("ep_next_drop", 0)
    ep_delta = s.get("ep_delta", 0)
    sign = "+" if ep_delta >= 0 else ""
    color = verdict_data["verdict_color"]
    verdict = verdict_data["verdict"]
    display = verdict_data["display_str"]
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-top:6px;padding-top:6px;border-top:1px solid #2d2d2d;">'
        f'<span style="color:#9ca3af;font-size:0.78em;">'
        f'FPL xPts: {ep_add:.1f} &minus; {ep_drop:.1f} = <b style="color:#e0e0e0;">{sign}{ep_delta:.1f}</b>'
        f'</span>'
        f'<span style="background:{color};color:#0d1117;padding:2px 10px;border-radius:10px;'
        f'font-size:0.78em;font-weight:bold;">{verdict} &nbsp; {display}</span>'
        f'</div>'
    )


def _build_trend_ownership_row(s: dict) -> str:
    """Build the price trend + ownership HTML row for a suggestion card."""
    pt = s.get("price_trend")
    badge = s.get("add_ownership_badge", "")
    pct = s.get("add_ownership_pct", 0)

    badges = []
    if pt:
        if pt.get("badge_html"):
            badges.append(pt["badge_html"])
        if pt.get("rush_html"):
            badges.append(pt["rush_html"])
    if pct:
        badges.append(
            f'<span style="color:#9ca3af;font-size:0.78em;">Own: {pct:.1f}%</span>'
        )
    if badge:
        badges.append(badge)

    if not badges:
        return ""
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:4px;">'
        + "".join(badges)
        + "</div>"
    )


def _render_transfer_suggestions(suggestions: List[Dict], free_transfers: int = 1):
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
            {_build_hit_verdict_row(s)}
            {_build_trend_ownership_row(s)}
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

    # Get current squad — skipping any GW where Free Hit was active.
    # After a Free Hit GW the squad REVERTS to the pre-FH squad, so loading
    # picks from a Free Hit GW would show the wrong (temporary) 15 players.
    chips_list = history.get("chips", []) if history else []
    fh_gws = {c["event"] for c in chips_list if c.get("name") == "freehit"}

    picks_data = None
    picks_source_gw = None
    for try_gw in [current_gw, current_gw - 1, current_gw - 2, current_gw - 3]:
        if try_gw < 1:
            break
        if try_gw in fh_gws:
            continue  # Free Hit GW — squad is temporary, skip it
        candidate = get_classic_team_picks(team_id, try_gw)
        if candidate:
            picks_data = candidate
            picks_source_gw = try_gw
            break

    if not picks_data:
        show_api_error("loading your current squad")
        return

    if picks_source_gw and picks_source_gw < current_gw - 1:
        st.info(
            f"Squad loaded from GW{picks_source_gw} — GW{picks_source_gw + 1} used a Free Hit "
            f"(squad reverted after that GW)."
        )

    picks = picks_data.get("picks", [])
    active_chip = picks_data.get("active_chip")  # e.g., "wildcard", "freehit", "bboost", "3xc"
    entry_history = picks_data.get("entry_history", {})

    bank = entry_history.get("bank", 0)
    squad_value = entry_history.get("value", 0)
    transfers_made = entry_history.get("event_transfers", 0)
    transfer_cost = entry_history.get("event_transfers_cost", 0)

    # Compute chip status and free transfers
    chip_status = _parse_chip_status(history, current_gw)
    free_transfers = _compute_free_transfers(history, entry_history, current_gw)

    # Status panel — shown before filters so users see FT count immediately
    _render_transfer_status_panel(bank, squad_value, free_transfers, chip_status, active_chip)

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
            # Default lookahead shrinks near end of season (no point looking past GW38)
            default_fdr = min(5, max(2, 38 - current_gw))
            fdr_weeks = st.slider("FDR Lookahead (weeks)", 1, 8, default_fdr)

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

        # Fallback: when Rotowire hasn't published yet, use FPL's own ep_next so
        # players don't all land on the neutral 0.5 in the 1GW score.
        for _df in [all_players, squad_df]:
            if "ep_next" in _df.columns and "Projected_Points" in _df.columns:
                mask = _df["Projected_Points"].isna() & (_df["ep_next"] > 0)
                _df.loc[mask, "Projected_Points"] = _df.loc[mask, "ep_next"]

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

    # Chip strategy advisor (needs Keep Scores to be computed first)
    _render_chip_advisor(chip_status, squad_df, current_gw)

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
    suggestions = _build_transfer_suggestions(
        squad_df, available, bank, top_n=3, depth_map=depth_map, free_transfers=free_transfers
    )
    _render_transfer_suggestions(suggestions, free_transfers=free_transfers)

    # 2-Transfer Plan (only when both FTs banked)
    if free_transfers >= 2:
        multi_plan = _build_multi_transfer_plan(squad_df, available, bank, depth_map=depth_map)
        if multi_plan:
            st.markdown("")
            _render_multi_transfer_plan(multi_plan)

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

    # Add transfer rush indicator
    top_targets["Rush"] = top_targets["transfers_in_event"].apply(
        lambda x: "🔥" if int(x or 0) > 200_000 else ""
    )

    # Format for display — use HealthyForm if available
    target_form_col = "HealthyForm" if "HealthyForm" in top_targets.columns else "form"
    target_display_cols = [
        "Player", "Team", "Position", "now_cost", "Price_Change", "Rush", target_form_col,
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

    display_cols_final = ["Player", "Team", "Position", "Price", "Δ", "Rush", "Form",
                          "Season Pts", "Proj Pts", "Ownership", "Avg FDR",
                          "1GW", "ROS", "Transfer Score", "Status"]
    display_cols_final = [c for c in display_cols_final if c in targets_show.columns]
    render_styled_table(
        targets_show[display_cols_final],
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
