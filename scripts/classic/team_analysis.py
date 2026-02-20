"""
Classic FPL - My Team Analysis Page

Displays the user's current squad with projected points, captain picks,
squad value, and gameweek history.
"""

import config
import pandas as pd
import plotly.express as px
import streamlit as st
from scripts.common.error_helpers import show_api_error
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_classic_team_picks,
    get_classic_team_history,
    get_classic_team_position_data,
    get_entry_details,
    get_current_gameweek,
    get_rotowire_player_projections,
    position_converter,
)
from scripts.common.team_analysis_helpers import render_season_highlights
from scripts.common.styled_tables import render_styled_table
from fuzzywuzzy import fuzz


def _format_money(value: int) -> str:
    """Format FPL money value (stored as tenths) to display format."""
    if value is None:
        return "N/A"
    return f"£{value / 10:.1f}m"


def _get_chip_display(active_chip: str) -> str:
    """Format chip name for display."""
    chip_names = {
        "wildcard": "Wildcard",
        "freehit": "Free Hit",
        "bboost": "Bench Boost",
        "3xc": "Triple Captain",
    }
    if not active_chip:
        return "None"
    return chip_names.get(active_chip, active_chip.title())


def _format_rank_change(current: int, previous: int) -> str:
    """Format rank change with arrow indicator."""
    if current is None or previous is None:
        return ""
    diff = previous - current  # Positive = improved (lower rank is better)
    if diff > 0:
        return f"↑ {diff:,}"
    elif diff < 0:
        return f"↓ {abs(diff):,}"
    return "→ 0"


def _build_squad_dataframe(picks: list, bootstrap: dict) -> pd.DataFrame:
    """Map element IDs from picks to player info from bootstrap data."""
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    rows = []
    for pick in picks:
        element_id = pick["element"]
        player = elements.get(element_id, {})

        rows.append({
            "element_id": element_id,
            "Player": player.get("web_name", "Unknown"),
            "Full Name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "Team": teams.get(player.get("team"), "???"),
            "Position": position_converter(player.get("element_type")),
            "squad_position": pick["position"],
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "multiplier": pick.get("multiplier", 1),
        })

    return pd.DataFrame(rows)


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


def _add_projections_to_squad(squad_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """Add Rotowire projections to squad dataframe."""
    points_list = []
    rank_list = []

    for _, row in squad_df.iterrows():
        proj = _lookup_projection(
            row["Player"],
            row["Team"],
            row["Position"],
            projections_df
        )
        points_list.append(proj["Points"])
        rank_list.append(proj["Pos Rank"])

    squad_df["Points"] = points_list
    squad_df["Pos Rank"] = rank_list
    return squad_df


def _get_sub_priority_label(position: int) -> str:
    """Get display label for bench position."""
    labels = {12: "GK Sub", 13: "1st Sub", 14: "2nd Sub", 15: "3rd Sub"}
    return labels.get(position, "")


def _style_rank_change(val):
    """Style rank change values with colors."""
    if isinstance(val, str):
        if val.startswith("↑"):
            return "color: #00c853; font-weight: bold;"
        elif val.startswith("↓"):
            return "color: #ff5252; font-weight: bold;"
    return ""


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


def _stat_card(label: str, value: str, accent: str = "#00ff87") -> str:
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'</div>'
    )


def show_classic_team_analysis_page():
    """Display Classic FPL My Team page with squad, projections, and metrics."""

    st.title("My Classic FPL Team")

    # Check if team ID is configured
    team_id = config.FPL_CLASSIC_TEAM_ID
    if not team_id:
        st.warning("No Classic FPL team configured.")
        st.info(
            "Add your team ID to your `.env` file:\n\n"
            "```\nFPL_CLASSIC_TEAM_ID=123456\n```\n\n"
            "You can find your team ID in the URL when viewing your team on the FPL website."
        )
        return

    # Fetch entry details for header
    with st.spinner("Loading team data..."):
        entry = get_entry_details(team_id)

    if not entry:
        show_api_error(f"loading details for team ID {team_id}", hint_key="team_id")
        return

    # Team Header Section
    st.markdown("---")
    team_name = entry.get("name", "Unknown Team")
    manager_first = entry.get("player_first_name", "")
    manager_last = entry.get("player_last_name", "")
    manager_name = f"{manager_first} {manager_last}".strip() or "Unknown Manager"
    overall_rank = entry.get("summary_overall_rank")
    total_points = entry.get("summary_overall_points")

    def _header_card(team: str, manager: str) -> str:
        return (
            f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
            f'background:linear-gradient(135deg,#37003c 0%,#5a0060 100%);">'
            f'<div style="color:#00ff87;font-size:22px;font-weight:800;margin-bottom:4px;">{team}</div>'
            f'<div style="color:rgba(255,255,255,0.8);font-size:14px;">Manager: {manager}</div>'
            f'</div>'
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_header_card(team_name, manager_name), unsafe_allow_html=True)
    with col2:
        rank_val = f"{overall_rank:,}" if overall_rank else "N/A"
        st.markdown(_stat_card("Overall Rank", rank_val, accent="#e0e0e0"), unsafe_allow_html=True)
    with col3:
        pts_val = f"{total_points:,}" if total_points else "N/A"
        st.markdown(_stat_card("Total Points", pts_val), unsafe_allow_html=True)

    st.markdown("---")

    # Get current gameweek and history
    current_gw = get_current_gameweek()
    history = get_classic_team_history(team_id)

    if not history or not history.get("current"):
        st.warning("No gameweek history available yet.")
        return

    gw_history = history["current"]
    available_gws = [gw["event"] for gw in gw_history]

    if not available_gws:
        st.warning("No gameweek data available yet.")
        return

    # Gameweek Selector and Chip Display
    col_gw, col_chip, col_spacer = st.columns([1, 1, 2])
    with col_gw:
        selected_gw = st.selectbox(
            "Select Gameweek",
            options=sorted(available_gws, reverse=True),
            index=0,
            format_func=lambda x: f"GW {x}",
        )

    # Fetch picks for selected gameweek
    picks_data = get_classic_team_picks(team_id, selected_gw)
    if not picks_data:
        show_api_error(f"loading squad for Gameweek {selected_gw}")
        return

    picks = picks_data.get("picks", [])
    active_chip = picks_data.get("active_chip")
    entry_history = picks_data.get("entry_history", {})

    with col_chip:
        chip_display = _get_chip_display(active_chip)
        if active_chip:
            st.success(f"**Chip:** {chip_display}")
        else:
            st.info("**Chip:** None")

    # GW Metrics in styled cards
    st.markdown("#### Gameweek Performance")
    gw_points = entry_history.get("points", "N/A")
    gw_rank = entry_history.get("rank")
    squad_value = entry_history.get("value")
    bank = entry_history.get("bank")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(_stat_card("GW Points", str(gw_points)), unsafe_allow_html=True)
    with col2:
        st.markdown(_stat_card("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A", accent="#e0e0e0"), unsafe_allow_html=True)
    with col3:
        st.markdown(_stat_card("Squad Value", _format_money(squad_value), accent="#e0e0e0"), unsafe_allow_html=True)
    with col4:
        st.markdown(_stat_card("Bank", _format_money(bank)), unsafe_allow_html=True)

    st.markdown("---")

    # Historical Trends Section
    if len(gw_history) > 1:
        with st.expander("Season Trends", expanded=True):
            # Get gameweek averages from bootstrap data
            bootstrap = get_classic_bootstrap_static()
            gw_averages = {}
            if bootstrap and "events" in bootstrap:
                for event in bootstrap["events"]:
                    gw_num = event.get("id")
                    avg_score = event.get("average_entry_score")
                    if gw_num and avg_score:
                        gw_averages[gw_num] = avg_score

            # Create trend dataframe
            trend_data = []
            cumulative_avg = 0
            for i, gw in enumerate(gw_history):
                gw_num = gw["event"]
                gw_avg = gw_averages.get(gw_num, 0)
                cumulative_avg += gw_avg
                trend_data.append({
                    "GW": gw_num,
                    "Your Points": gw["points"],
                    "Average Points": gw_avg,
                    "Your Total": gw["total_points"],
                    "Average Total": cumulative_avg,
                    "GW Rank": gw.get("rank"),
                    "Overall Rank": gw.get("overall_rank"),
                    "Value": gw.get("value", 0) / 10,
                    "Bank": gw.get("bank", 0) / 10,
                    "Transfers": gw.get("event_transfers", 0),
                    "Hits": gw.get("event_transfers_cost", 0),
                })

            trend_df = pd.DataFrame(trend_data)

            # Display charts
            tab1, tab2, tab3, tab4 = st.tabs(["GW Points", "Total Points", "Rank", "Value"])

            with tab1:
                fig = px.line(trend_df, x="GW", y=["Your Points", "Average Points"],
                              markers=True, title="Gameweek Points vs Average")
                fig.update_layout(**_DARK_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                trend_df["vs Avg"] = trend_df["Your Points"] - trend_df["Average Points"]
                total_vs_avg = trend_df["vs Avg"].sum()
                if total_vs_avg > 0:
                    st.success(f"**+{total_vs_avg:.0f}** points above average this season")
                elif total_vs_avg < 0:
                    st.error(f"**{total_vs_avg:.0f}** points below average this season")
                else:
                    st.info("Exactly average this season")

            with tab2:
                fig = px.line(trend_df, x="GW", y=["Your Total", "Average Total"],
                              markers=True, title="Cumulative Total Points vs Average")
                fig.update_layout(**_DARK_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                latest = trend_df.iloc[-1]
                diff = latest["Your Total"] - latest["Average Total"]
                if diff > 0:
                    st.success(f"**+{diff:.0f}** points ahead of average manager")
                elif diff < 0:
                    st.error(f"**{abs(diff):.0f}** points behind average manager")

            with tab3:
                fig = px.line(trend_df, x="GW", y="Overall Rank",
                              markers=True, title="Overall Rank Progression")
                fig.update_layout(**_DARK_CHART_LAYOUT)
                fig.update_yaxes(autorange="reversed", title="Overall Rank (lower = better)")
                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                fig = px.line(trend_df, x="GW", y=["Value", "Bank"],
                              markers=True, title="Squad Value & Bank")
                fig.update_layout(**_DARK_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            st.markdown("#### Season Summary")
            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

            with sum_col1:
                try:
                    points_col = pd.to_numeric(trend_df["Your Points"], errors="coerce")
                    if points_col.notna().any():
                        best_idx = points_col.idxmax()
                        best_gw = trend_df.loc[best_idx]
                        st.markdown(_stat_card("Best GW", f"GW{int(best_gw['GW'])} ({int(best_gw['Your Points'])} pts)"), unsafe_allow_html=True)
                    else:
                        st.markdown(_stat_card("Best GW", "N/A", accent="#9ca3af"), unsafe_allow_html=True)
                except Exception:
                    st.markdown(_stat_card("Best GW", "N/A", accent="#9ca3af"), unsafe_allow_html=True)

            with sum_col2:
                total_transfers = trend_df["Transfers"].sum()
                st.markdown(_stat_card("Total Transfers", str(int(total_transfers)), accent="#e0e0e0"), unsafe_allow_html=True)

            with sum_col3:
                total_hits = trend_df["Hits"].sum()
                hits_color = "#f87171" if total_hits > 0 else "#00ff87"
                st.markdown(_stat_card("Total Hits", f"-{int(total_hits)} pts", accent=hits_color), unsafe_allow_html=True)

            with sum_col4:
                if len(trend_df) >= 2:
                    start_rank = trend_df.iloc[0]["Overall Rank"]
                    end_rank = trend_df.iloc[-1]["Overall Rank"]
                    if start_rank and end_rank:
                        rank_change = int(start_rank - end_rank)
                        if rank_change > 0:
                            st.markdown(_stat_card("Rank Change", f"↑ {rank_change:,}"), unsafe_allow_html=True)
                        elif rank_change < 0:
                            st.markdown(_stat_card("Rank Change", f"↓ {abs(rank_change):,}", accent="#f87171"), unsafe_allow_html=True)
                        else:
                            st.markdown(_stat_card("Rank Change", "→ 0", accent="#9ca3af"), unsafe_allow_html=True)

    st.markdown("---")

    # Fetch bootstrap data for player info
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        show_api_error("loading player data")
        return

    # ---------------------------
    # SEASON HIGHLIGHTS
    # ---------------------------
    st.markdown("### Season Highlights")

    # Get position data for highlights (needed for MVP, Best 11, Best Clubs, and Points by Position)
    latest_played_gw = max((gw["event"] for gw in gw_history), default=None)
    pos_result = None
    player_list = []

    if latest_played_gw:
        with st.spinner("Loading season data..."):
            pos_result = get_classic_team_position_data(team_id, latest_played_gw)
            player_list = pos_result.get("players", [])

    render_season_highlights(player_list, bootstrap_data=bootstrap, team_id=team_id, is_classic=True)

    st.markdown("---")

    # ---------------------------
    # SEASON HISTORY
    # ---------------------------
    past_seasons = history.get("past", [])
    if past_seasons:
        st.markdown("### Season History")

        past_df = pd.DataFrame(past_seasons)
        past_df = past_df.rename(columns={
            "season_name": "Season",
            "total_points": "Points",
            "rank": "Rank",
        })

        tab_rank, tab_points = st.tabs(["Overall Rank", "Total Points"])

        with tab_rank:
            fig_rank = px.line(
                past_df, x="Season", y="Rank",
                markers=True, title="Overall Rank by Season",
            )
            fig_rank.update_layout(**_DARK_CHART_LAYOUT, height=400)
            fig_rank.update_yaxes(autorange="reversed", title="Overall Rank")
            fig_rank.update_xaxes(title="Season")
            st.plotly_chart(fig_rank, use_container_width=True)

        with tab_points:
            fig_pts = px.line(
                past_df, x="Season", y="Points",
                markers=True, title="Total Points by Season",
            )
            fig_pts.update_layout(**_DARK_CHART_LAYOUT, height=400)
            fig_pts.update_xaxes(title="Season")
            fig_pts.update_yaxes(title="Total Points")
            st.plotly_chart(fig_pts, use_container_width=True)

        # Data table
        display_df = past_df[["Season", "Points", "Rank"]].copy()
        display_df["Points"] = display_df["Points"].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
        display_df["Rank"] = display_df["Rank"].apply(lambda x: f"{x:,}" if pd.notna(x) and x else "N/A")
        render_styled_table(display_df)

        st.markdown("---")

    # Build squad dataframe
    squad_df = _build_squad_dataframe(picks, bootstrap)

    # Try to fetch and merge projections
    projections_available = False
    projections_df = None
    try:
        rotowire_url = config.ROTOWIRE_URL
        if rotowire_url:
            with st.spinner("Loading projections..."):
                projections_df = get_rotowire_player_projections(rotowire_url)
            if projections_df is not None and not projections_df.empty:
                squad_df = _add_projections_to_squad(squad_df, projections_df)
                projections_available = True
    except Exception as e:
        st.warning(f"Could not load projections: {str(e)}")

    if not projections_available:
        st.info("Rotowire projections unavailable. Displaying squad without projected points.")
        squad_df["Points"] = None
        squad_df["Pos Rank"] = None

    # Split into starting XI and bench
    starting_xi = squad_df[squad_df["squad_position"] <= 11].copy()
    bench = squad_df[squad_df["squad_position"] > 11].copy()

    # Format player names with captain indicators
    def format_player_name(row):
        name = row["Player"]
        if row["is_captain"]:
            return f"{name} (C)"
        elif row["is_vice_captain"]:
            return f"{name} (V)"
        return name

    starting_xi["Display Name"] = starting_xi.apply(format_player_name, axis=1)
    bench["Display Name"] = bench.apply(format_player_name, axis=1)
    bench["Priority"] = bench["squad_position"].apply(_get_sub_priority_label)

    # Calculate totals
    if projections_available:
        starting_proj_total = pd.to_numeric(starting_xi["Points"], errors="coerce").sum()
        bench_proj_total = pd.to_numeric(bench["Points"], errors="coerce").sum()
    else:
        starting_proj_total = None
        bench_proj_total = None

    # Starting XI Section
    st.markdown("### Starting XI")
    if projections_available and starting_proj_total and pd.notna(starting_proj_total):
        st.markdown(f"**Total Projected Points: {starting_proj_total:.1f}**")

    # Prepare display columns
    starting_display = starting_xi[["Display Name", "Team", "Position"]].copy()
    if projections_available:
        starting_display["Proj Pts"] = starting_xi["Points"]
        starting_display["Pos Rank"] = starting_xi["Pos Rank"]

    starting_display = starting_display.rename(columns={"Display Name": "Player"})

    render_styled_table(
        starting_display,
        col_formats={"Proj Pts": "{:.1f}"},
    )

    st.markdown("---")

    # Bench Section
    st.markdown("### Bench")
    if projections_available and bench_proj_total and pd.notna(bench_proj_total):
        st.markdown(f"**Total Projected Points: {bench_proj_total:.1f}**")

    bench_display = bench[["Display Name", "Team", "Position", "Priority"]].copy()
    if projections_available:
        bench_display["Proj Pts"] = bench["Points"]

    bench_display = bench_display.rename(columns={"Display Name": "Player"})

    render_styled_table(
        bench_display,
        col_formats={"Proj Pts": "{:.1f}"},
    )

    st.markdown("---")

    # ---------------------------
    # POINTS BY POSITION
    # ---------------------------
    st.markdown("### Points by Position")

    POSITION_COLORS = {
        "GK": "#f39c12",
        "DEF": "#3498db",
        "MID": "#2ecc71",
        "FWD": "#e74c3c",
    }

    # Use pos_result from Season Highlights section (already fetched above)
    if pos_result:
        team_pos = pos_result["positions"]
        total = sum(team_pos.values())

        if total > 0:
            pos_cols = ["GK", "DEF", "MID", "FWD"]

            col_pie, col_metrics = st.columns([1, 1])

            with col_pie:
                pie_data = pd.DataFrame({
                    "Position": pos_cols,
                    "Points": [team_pos[c] for c in pos_cols]
                })
                fig = px.pie(
                    pie_data,
                    values="Points",
                    names="Position",
                    color="Position",
                    color_discrete_map=POSITION_COLORS,
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300,
                    paper_bgcolor="#1a1a2e",
                    font=dict(color="#ffffff"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_metrics:
                for pos in pos_cols:
                    pts = team_pos[pos]
                    pct = f"{pts / total * 100:.1f}%" if total > 0 else "0%"
                    st.markdown(_stat_card(pos, f"{pts} pts ({pct})", accent=POSITION_COLORS.get(pos, "#00ff87")), unsafe_allow_html=True)

            # Player detail table
            if player_list:
                st.markdown("**Player Breakdown**")
                players_df = pd.DataFrame(player_list)
                players_df.columns = ["Player", "Position", "Total Points", "Team"]

                pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
                players_df["_pos_order"] = players_df["Position"].map(pos_order)
                players_df = players_df.sort_values(
                    ["_pos_order", "Total Points"], ascending=[True, False]
                ).drop(columns=["_pos_order"])

                render_styled_table(
                    players_df,
                    positive_color_cols=["Total Points"],
                )
        else:
            st.info("No position data available yet (season may not have started).")
    else:
        st.info("No position data available yet (season may not have started).")

    st.markdown("---")

    # Chips Used Section
    with st.expander("Chips Used This Season"):
        chips = history.get("chips", [])
        if chips:
            chip_cols = st.columns(len(chips)) if len(chips) <= 4 else st.columns(4)
            for i, chip in enumerate(chips):
                with chip_cols[i % len(chip_cols)]:
                    chip_name = _get_chip_display(chip.get("name", ""))
                    chip_gw = chip.get("event", "?")
                    st.success(f"**{chip_name}**\nGW {chip_gw}")
        else:
            st.info("No chips used yet this season.")

