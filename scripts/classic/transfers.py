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


def _min_max_norm(series: pd.Series) -> pd.Series:
    """Min-max normalization to [0,1]."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def _compute_transfer_score(df: pd.DataFrame, w_proj: float, w_form: float,
                            w_fdr: float, w_price: float) -> pd.DataFrame:
    """Compute transfer target score."""
    tmp = df.copy()

    # Normalize components
    tmp["Proj_norm"] = _min_max_norm(tmp["Projected_Points"]).fillna(0.5)
    tmp["Form_norm"] = _min_max_norm(tmp["form"]).fillna(0.5)

    # Invert FDR (lower is better)
    tmp["FDREase"] = 6 - pd.to_numeric(tmp["AvgFDR"], errors="coerce")
    tmp["FDREase_norm"] = _min_max_norm(tmp["FDREase"]).fillna(0.5)

    # Invert price (lower price = more value)
    max_cost = tmp["now_cost"].max()
    tmp["PriceValue"] = max_cost - tmp["now_cost"]
    tmp["Price_norm"] = _min_max_norm(tmp["PriceValue"]).fillna(0.5)

    denom = max(w_proj + w_form + w_fdr + w_price, 1e-9)
    tmp["Transfer_Score"] = (
        w_proj * tmp["Proj_norm"] +
        w_form * tmp["Form_norm"] +
        w_fdr * tmp["FDREase_norm"] +
        w_price * tmp["Price_norm"]
    ) / denom

    # Cleanup
    drop_cols = ["Proj_norm", "Form_norm", "FDREase", "FDREase_norm", "PriceValue", "Price_norm"]
    return tmp.drop(columns=[c for c in drop_cols if c in tmp.columns])


def _compute_keep_score(df: pd.DataFrame, w_proj: float, w_form: float,
                        w_fdr: float, w_points: float) -> pd.DataFrame:
    """Compute keep score for current squad players."""
    tmp = df.copy()

    # Normalize components
    tmp["Proj_norm"] = _min_max_norm(tmp["Projected_Points"]).fillna(0.5)
    tmp["Form_norm"] = _min_max_norm(tmp["form"]).fillna(0.5)
    tmp["Points_norm"] = _min_max_norm(tmp["total_points"]).fillna(0.5)

    # Invert FDR
    tmp["FDREase"] = 6 - pd.to_numeric(tmp["AvgFDR"], errors="coerce")
    tmp["FDREase_norm"] = _min_max_norm(tmp["FDREase"]).fillna(0.5)

    denom = max(w_proj + w_form + w_fdr + w_points, 1e-9)
    tmp["Keep_Score"] = (
        w_proj * tmp["Proj_norm"] +
        w_form * tmp["Form_norm"] +
        w_fdr * tmp["FDREase_norm"] +
        w_points * tmp["Points_norm"]
    ) / denom

    # Cleanup
    drop_cols = ["Proj_norm", "Form_norm", "Points_norm", "FDREase", "FDREase_norm"]
    return tmp.drop(columns=[c for c in drop_cols if c in tmp.columns])


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
        st.error("Failed to load player data. Please try again later.")
        return

    if not entry:
        st.error(f"Failed to load team details for team ID {team_id}.")
        return

    # Team header info
    team_name = entry.get("name", "Unknown Team")
    st.markdown(f"### {team_name}")

    # Get current squad
    picks_data = get_classic_team_picks(team_id, current_gw)
    if not picks_data:
        # Try previous gameweek
        picks_data = get_classic_team_picks(team_id, current_gw - 1)

    if not picks_data:
        st.error("Failed to load current squad. Please try again later.")
        return

    picks = picks_data.get("picks", [])
    entry_history = picks_data.get("entry_history", {})

    # Display bank and value
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bank = entry_history.get("bank", 0)
        st.metric("Bank", _format_money(bank))
    with col2:
        squad_value = entry_history.get("value", 0)
        st.metric("Squad Value", _format_money(squad_value))
    with col3:
        transfers_made = entry_history.get("event_transfers", 0)
        st.metric("Transfers Made", transfers_made)
    with col4:
        transfer_cost = entry_history.get("event_transfers_cost", 0)
        st.metric("Transfer Cost", f"-{transfer_cost} pts" if transfer_cost else "0 pts")

    st.markdown("---")

    # Controls
    with st.expander("Filters & Weights", expanded=True):
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

        st.markdown("**Transfer Score Weights:**")
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        w_proj = float(wcol1.slider("Projections", 0.0, 1.0, 0.4, 0.05, key="w_proj"))
        w_form = float(wcol2.slider("Form", 0.0, 1.0, 0.3, 0.05, key="w_form"))
        w_fdr = float(wcol3.slider("Fixture Ease", 0.0, 1.0, 0.2, 0.05, key="w_fdr"))
        w_price = float(wcol4.slider("Value (Price)", 0.0, 1.0, 0.1, 0.05, key="w_price"))

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
        except Exception as e:
            st.warning(f"Could not load projections: {e}")

        # Add projections
        all_players = _add_projections(all_players, projections_df)
        squad_df = _add_projections(squad_df, projections_df)

        # Compute scores
        squad_df = _compute_keep_score(squad_df, w_proj, w_form, w_fdr, 0.2)

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

    # Compute transfer score
    available = _compute_transfer_score(available, w_proj, w_form, w_fdr, w_price)
    available = available.sort_values("Transfer_Score", ascending=False)

    # ---------------------------
    # SQUAD ANALYSIS SECTION
    # ---------------------------
    st.header("Your Squad Analysis")

    # Split into starting XI and bench
    starting_xi = squad_df[squad_df["squad_position"] <= 11].copy()
    bench = squad_df[squad_df["squad_position"] > 11].copy()

    # Show squad with keep scores
    squad_display = squad_df.sort_values("Keep_Score", ascending=True).copy()

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

    # Format display columns
    display_cols = ["Player", "Team", "Position", "now_cost", "form", "total_points",
                    "Projected_Points", "AvgFDR", "Keep_Score", "Status"]
    squad_show = squad_display[display_cols].copy()
    squad_show["Price"] = squad_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
    squad_show["AvgFDR"] = squad_show["AvgFDR"].round(2)
    squad_show["Keep_Score"] = squad_show["Keep_Score"].round(3)
    squad_show["Projected_Points"] = squad_show["Projected_Points"].fillna("-")

    # Rename for display
    squad_show = squad_show.rename(columns={
        "form": "Form",
        "total_points": "Season Pts",
        "Projected_Points": "Proj Pts",
        "AvgFDR": "Avg FDR",
        "Keep_Score": "Keep Score"
    })

    st.dataframe(
        squad_show[["Player", "Team", "Position", "Price", "Form", "Season Pts",
                    "Proj Pts", "Avg FDR", "Keep Score", "Status"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Form": st.column_config.NumberColumn("Form", format="%.1f", width="small"),
            "Season Pts": st.column_config.NumberColumn("Season", width="small"),
            "Proj Pts": st.column_config.TextColumn("Proj", width="small"),
            "Avg FDR": st.column_config.NumberColumn("FDR", format="%.2f", width="small"),
            "Keep Score": st.column_config.NumberColumn("Keep", format="%.3f", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        }
    )

    # Show suggested transfers out
    st.subheader("Suggested Transfers Out")
    st.caption("Players with lowest Keep Score - consider transferring these out.")

    lowest_keep = squad_display.nsmallest(3, "Keep_Score")
    for _, player in lowest_keep.iterrows():
        with st.container():
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**{player['Player']}** ({player['Team']}, {player['Position']})")
                st.caption(f"Price: £{player['now_cost']/10:.1f}m | Form: {player['form']:.1f} | Keep Score: {player['Keep_Score']:.3f}")
            with col2:
                fixtures = _get_team_fixtures(player["Team_ID"], fdr_weeks, current_gw)
                st.markdown(_format_fixtures_html(fixtures, teams_map, 6), unsafe_allow_html=True)
                if player["news"]:
                    st.warning(f"⚠️ {player['news']}")

    st.markdown("---")

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

    # Format for display
    targets_show = top_targets[[
        "Player", "Team", "Position", "now_cost", "Price_Change", "form",
        "total_points", "Projected_Points", "selected_by_percent",
        "AvgFDR", "Transfer_Score", "Status"
    ]].copy()

    targets_show["Price"] = targets_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
    targets_show["AvgFDR"] = targets_show["AvgFDR"].round(2)
    targets_show["Transfer_Score"] = targets_show["Transfer_Score"].round(3)
    targets_show["Projected_Points"] = targets_show["Projected_Points"].fillna("-")
    targets_show["Ownership"] = targets_show["selected_by_percent"].apply(lambda x: f"{x:.1f}%")

    # Rename for display
    targets_show = targets_show.rename(columns={
        "form": "Form",
        "total_points": "Season Pts",
        "Projected_Points": "Proj Pts",
        "AvgFDR": "Avg FDR",
        "Transfer_Score": "Score",
        "Price_Change": "Δ"
    })

    st.dataframe(
        targets_show[["Player", "Team", "Position", "Price", "Δ", "Form",
                      "Season Pts", "Proj Pts", "Ownership", "Avg FDR", "Score", "Status"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Δ": st.column_config.TextColumn("Δ", width="small"),
            "Form": st.column_config.NumberColumn("Form", format="%.1f", width="small"),
            "Season Pts": st.column_config.NumberColumn("Season", width="small"),
            "Proj Pts": st.column_config.TextColumn("Proj", width="small"),
            "Ownership": st.column_config.TextColumn("Own%", width="small"),
            "Avg FDR": st.column_config.NumberColumn("FDR", format="%.2f", width="small"),
            "Score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        }
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
            pos_show = pos_targets[[
                "Player", "Team", "now_cost", "form", "total_points",
                "Projected_Points", "selected_by_percent", "AvgFDR", "Transfer_Score"
            ]].copy()

            pos_show["Price"] = pos_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
            pos_show["AvgFDR"] = pos_show["AvgFDR"].round(2)
            pos_show["Transfer_Score"] = pos_show["Transfer_Score"].round(3)
            pos_show["Projected_Points"] = pos_show["Projected_Points"].fillna("-")
            pos_show["Own%"] = pos_show["selected_by_percent"].apply(lambda x: f"{x:.1f}%")

            st.dataframe(
                pos_show[["Player", "Team", "Price", "form", "total_points",
                          "Projected_Points", "Own%", "AvgFDR", "Transfer_Score"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player", width="medium"),
                    "Team": st.column_config.TextColumn("Team", width="small"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "form": st.column_config.NumberColumn("Form", format="%.1f"),
                    "total_points": st.column_config.NumberColumn("Season Pts"),
                    "Projected_Points": st.column_config.TextColumn("Proj Pts"),
                    "Own%": st.column_config.TextColumn("Own%", width="small"),
                    "AvgFDR": st.column_config.NumberColumn("Avg FDR", format="%.2f"),
                    "Transfer_Score": st.column_config.NumberColumn("Score", format="%.3f"),
                }
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
        st.dataframe(transfers_df, use_container_width=True, hide_index=True)
    else:
        st.info("No transfers found for this season.")
