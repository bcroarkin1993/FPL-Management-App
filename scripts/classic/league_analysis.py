"""
Classic FPL - League Analysis Page

Provides advanced league-wide analytics including:
- Chip usage analysis across the league
- Rank movement and points behind leader
- Team value comparison
- Transfer activity analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Dict, List

import config
from scripts.common.error_helpers import show_api_error
from scripts.common.utils import (
    get_league_standings,
    get_classic_team_history,
    get_current_gameweek,
    get_classic_bootstrap_static,
    get_classic_team_position_data,
)
from scripts.common.styled_tables import render_styled_table

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title=dict(font=dict(size=20, color="#ffffff"), x=0.5, xanchor="center"),
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
)


def _stat_card(label: str, value: str, detail: str = "", accent: str = "#00ff87") -> str:
    detail_html = f'<div style="color:#9ca3af;font-size:12px;margin-top:2px;">{detail}</div>' if detail else ""
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'{detail_html}'
        f'</div>'
    )


# ---------------------------
# DATA FETCHING
# ---------------------------

def get_league_display_options() -> list:
    """Build list of league options for the dropdown."""
    leagues = config.FPL_CLASSIC_LEAGUE_IDS or []
    options = []

    for league in leagues:
        league_id = league["id"]
        if league.get("name"):
            name = league["name"]
        else:
            data = get_league_standings(league_id)
            if data and "league" in data:
                name = data["league"].get("name", f"League {league_id}")
            else:
                name = f"League {league_id}"

        options.append({
            "id": league_id,
            "name": name,
            "display": f"{name} ({league_id})"
        })

    return options


@st.cache_data(ttl=600)
def fetch_team_histories_for_league(team_ids: List[int]) -> Dict[int, dict]:
    """Fetch history data for multiple teams."""
    histories = {}
    for team_id in team_ids:
        history = get_classic_team_history(team_id)
        if history:
            histories[team_id] = history
    return histories


# ---------------------------
# CHIP ANALYSIS
# ---------------------------

def analyze_chip_usage(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Analyze chip usage across all teams in the league."""
    chip_data = []

    chip_display_names = {
        "wildcard": "Wildcard",
        "freehit": "Free Hit",
        "bboost": "Bench Boost",
        "3xc": "Triple Captain"
    }

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        chips_used = history.get("chips", [])

        # Track which chips this team has used
        team_chips = {
            "Team": team_name,
            "Wildcard": "-",
            "Free Hit": "-",
            "Bench Boost": "-",
            "Triple Captain": "-",
            "Total Used": 0
        }

        for chip in chips_used:
            chip_name = chip.get("name", "")
            chip_gw = chip.get("event", "?")
            display_name = chip_display_names.get(chip_name, chip_name)

            if display_name in team_chips:
                team_chips[display_name] = f"GW{chip_gw}"
                team_chips["Total Used"] += 1

        chip_data.append(team_chips)

    df = pd.DataFrame(chip_data)
    return df.sort_values("Total Used", ascending=False)


def plot_chip_timing(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> Optional[go.Figure]:
    """Create a timeline of when chips were used."""
    chip_events = []

    chip_display_names = {
        "wildcard": "Wildcard",
        "freehit": "Free Hit",
        "bboost": "Bench Boost",
        "3xc": "Triple Captain"
    }

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        chips_used = history.get("chips", [])

        for chip in chips_used:
            chip_name = chip.get("name", "")
            chip_gw = chip.get("event")
            display_name = chip_display_names.get(chip_name, chip_name)

            if chip_gw:
                chip_events.append({
                    "Team": team_name,
                    "Chip": display_name,
                    "Gameweek": chip_gw
                })

    if not chip_events:
        return None

    df = pd.DataFrame(chip_events)

    # Color mapping for chips
    color_map = {
        "Wildcard": "#e74c3c",
        "Free Hit": "#3498db",
        "Bench Boost": "#2ecc71",
        "Triple Captain": "#9b59b6"
    }

    fig = px.scatter(
        df,
        x="Gameweek",
        y="Team",
        color="Chip",
        title="Chip Usage Timeline",
        color_discrete_map=color_map,
        symbol="Chip"
    )

    fig.update_traces(marker=dict(size=15))
    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        xaxis_title="Gameweek",
        legend_title="Chip",
        height=max(400, len(df["Team"].unique()) * 30),
    )
    fig.update_yaxes(title="")

    return fig


# ---------------------------
# RANK ANALYSIS
# ---------------------------

def build_rank_progression(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Build rank progression data over gameweeks."""
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        for gw_data in current:
            gw = gw_data.get("event")
            overall_rank = gw_data.get("overall_rank")
            total_pts = gw_data.get("total_points", 0)

            if gw and overall_rank:
                all_data.append({
                    "Team": team_name,
                    "Gameweek": gw,
                    "Overall_Rank": overall_rank,
                    "Total_Points": total_pts
                })

    return pd.DataFrame(all_data)


def calculate_rank_changes(rank_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rank changes from start to current."""
    if rank_df.empty:
        return pd.DataFrame()

    # Get first and last gameweek for each team
    first_gw = rank_df.groupby("Team").first().reset_index()
    last_gw = rank_df.groupby("Team").last().reset_index()

    changes = first_gw[["Team"]].copy()
    changes["Start_Rank"] = first_gw["Overall_Rank"]
    changes["Current_Rank"] = last_gw["Overall_Rank"]
    changes["Rank_Change"] = changes["Start_Rank"] - changes["Current_Rank"]
    changes["Start_GW"] = first_gw["Gameweek"]
    changes["Current_GW"] = last_gw["Gameweek"]

    return changes.sort_values("Rank_Change", ascending=False)


def plot_rank_progression(rank_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot overall rank progression over time."""
    if rank_df.empty:
        return None

    fig = px.line(
        rank_df,
        x="Gameweek",
        y="Overall_Rank",
        color="Team",
        title="Overall Rank Progression",
        labels={"Overall_Rank": "Overall Rank", "Gameweek": "Gameweek"}
    )

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        hovermode="x unified",
    )
    fig.update_yaxes(autorange="reversed", title="Overall Rank (lower = better)")

    return fig


# ---------------------------
# POINTS ANALYSIS
# ---------------------------

def calculate_points_behind_leader(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Calculate points behind the leader over time."""
    # Build points progression
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        for gw_data in current:
            gw = gw_data.get("event")
            total_pts = gw_data.get("total_points", 0)

            if gw:
                all_data.append({
                    "Team": team_name,
                    "Gameweek": gw,
                    "Total_Points": total_pts
                })

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Calculate leader's points each gameweek
    leader_pts = df.groupby("Gameweek")["Total_Points"].max().to_dict()

    # Calculate points behind
    df["Leader_Points"] = df["Gameweek"].map(leader_pts)
    df["Points_Behind"] = df["Leader_Points"] - df["Total_Points"]

    return df


def plot_points_behind_leader(points_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot points behind leader over time."""
    if points_df.empty:
        return None

    fig = px.line(
        points_df,
        x="Gameweek",
        y="Points_Behind",
        color="Team",
        title="Points Behind Leader Over Time",
        labels={"Points_Behind": "Points Behind", "Gameweek": "Gameweek"}
    )

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        hovermode="x unified",
    )
    fig.update_yaxes(title="Points Behind Leader")

    return fig


# ---------------------------
# TEAM VALUE ANALYSIS
# ---------------------------

def analyze_team_values(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Analyze team values across the league."""
    value_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        if current:
            # Get latest gameweek data
            latest = current[-1]
            value = latest.get("value", 0) / 10  # Convert to millions
            bank = latest.get("bank", 0) / 10

            # Get starting value (GW1)
            start_value = current[0].get("value", 1000) / 10 if current else 100.0

            value_data.append({
                "Team": team_name,
                "Squad Value": value,
                "Bank": bank,
                "Total Value": value + bank,
                "Value Gain": (value + bank) - start_value
            })

    df = pd.DataFrame(value_data)
    return df.sort_values("Total Value", ascending=False)


def plot_value_comparison(value_df: pd.DataFrame) -> Optional[go.Figure]:
    """Create bar chart comparing team values."""
    if value_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Squad Value",
        x=value_df["Team"],
        y=value_df["Squad Value"],
        marker_color="#3498db"
    ))

    fig.add_trace(go.Bar(
        name="Bank",
        x=value_df["Team"],
        y=value_df["Bank"],
        marker_color="#2ecc71"
    ))

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        title="Team Value Breakdown",
        barmode="stack",
        xaxis_tickangle=-45,
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Value (millions)")

    return fig


# ---------------------------
# GAMEWEEK PERFORMANCE
# ---------------------------

def analyze_gameweek_performance(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Analyze gameweek-by-gameweek performance."""
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        gw_points = [gw.get("points", 0) for gw in current if gw.get("event")]

        if gw_points:
            all_data.append({
                "Team": team_name,
                "Games": len(gw_points),
                "Total_Points": sum(gw_points),
                "Avg_Points": round(sum(gw_points) / len(gw_points), 1),
                "Best_GW": max(gw_points),
                "Worst_GW": min(gw_points),
                "Std_Dev": round(pd.Series(gw_points).std(), 1) if len(gw_points) > 1 else 0
            })

    df = pd.DataFrame(all_data)
    return df.sort_values("Total_Points", ascending=False)


def get_weekly_scores(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Get weekly scores for all teams."""
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        for gw_data in current:
            gw = gw_data.get("event")
            pts = gw_data.get("points", 0)

            if gw:
                all_data.append({
                    "Team": team_name,
                    "Gameweek": gw,
                    "Points": pts
                })

    return pd.DataFrame(all_data)


def plot_scoring_distribution(weekly_scores: pd.DataFrame) -> Optional[go.Figure]:
    """Create box plot of scoring distribution."""
    if weekly_scores.empty:
        return None

    # Sort by median
    team_order = weekly_scores.groupby("Team")["Points"].median().sort_values(ascending=False).index.tolist()

    fig = px.box(
        weekly_scores,
        x="Team",
        y="Points",
        color="Team",
        title="Gameweek Scoring Distribution",
        category_orders={"Team": team_order}
    )

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        showlegend=False,
        xaxis_tickangle=-45,
    )
    fig.update_yaxes(title="Gameweek Points")

    return fig


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_classic_league_analysis_page():
    """Display the Classic League Analysis page."""

    st.title("Classic League Analysis")

    # Get configured leagues
    league_options = get_league_display_options()

    if not league_options:
        st.warning("No Classic leagues configured.")
        st.info(
            "Add leagues to `FPL_CLASSIC_LEAGUE_IDS` in your `.env` file:\n\n"
            "```\nFPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League\n```"
        )
        return

    # League selector
    if "classic_analysis_league_index" not in st.session_state:
        st.session_state.classic_analysis_league_index = 0

    display_options = [opt["display"] for opt in league_options]
    selected_display = st.selectbox(
        "Select League",
        options=display_options,
        index=st.session_state.classic_analysis_league_index,
        key="classic_analysis_league_selector"
    )

    new_index = display_options.index(selected_display)
    if new_index != st.session_state.classic_analysis_league_index:
        st.session_state.classic_analysis_league_index = new_index

    selected_league = league_options[st.session_state.classic_analysis_league_index]
    league_id = selected_league["id"]

    # Fetch standings to get team list
    with st.spinner("Loading league data..."):
        standings_data = get_league_standings(league_id)

    if not standings_data:
        show_api_error(f"loading data for league {league_id}", hint_key="league_id")
        return

    league_info = standings_data.get("league", {})
    standings = standings_data.get("standings", {})
    results = standings.get("results", [])

    if not results:
        st.info("No standings data available yet.")
        return

    # Build team info
    team_ids = [r.get("entry") for r in results if r.get("entry")]
    team_names = {r.get("entry"): r.get("entry_name", f"Team {r.get('entry')}") for r in results}

    # Slider to limit teams for performance
    num_teams = len(team_ids)
    if num_teams > 10:
        teams_to_analyze = st.slider(
            "Number of teams to analyze",
            min_value=5,
            max_value=min(num_teams, 30),
            value=min(num_teams, 15),
            help="Analyzing more teams takes longer but provides broader insights"
        )
        team_ids = team_ids[:teams_to_analyze]
        team_names = {k: v for k, v in team_names.items() if k in team_ids}

    # Fetch team histories
    with st.spinner(f"Loading data for {len(team_ids)} teams..."):
        team_histories = fetch_team_histories_for_league(team_ids)

    if not team_histories:
        st.info("Could not load team history data.")
        return

    # League header
    league_name = league_info.get("name", "Classic League")
    st.markdown(f"### {league_name}")

    current_gw = get_current_gameweek()
    if current_gw:
        st.caption(f"Analysis through Gameweek {current_gw}")

    st.divider()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Chip Usage",
        "Rank Movement",
        "Points Analysis",
        "Team Values",
        "Performance Stats",
        "Points by Position"
    ])

    # ---------------------------
    # TAB 1: CHIP USAGE
    # ---------------------------
    with tab1:
        st.subheader("Chip Usage Analysis")

        chip_df = analyze_chip_usage(team_histories, team_names)

        if not chip_df.empty:
            render_styled_table(
                chip_df,
                text_align={"Wildcard": "center", "Free Hit": "center",
                             "Bench Boost": "center", "Triple Captain": "center",
                             "Total Used": "center"},
            )

            st.divider()

            fig = plot_chip_timing(team_histories, team_names)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No chips have been played yet.")
        else:
            st.info("No chip data available.")

    # ---------------------------
    # TAB 2: RANK MOVEMENT
    # ---------------------------
    with tab2:
        st.subheader("Rank Movement Analysis")

        rank_df = build_rank_progression(team_histories, team_names)

        if not rank_df.empty:
            # Rank changes summary
            changes_df = calculate_rank_changes(rank_df)

            if not changes_df.empty:
                st.markdown("**Biggest Movers (Overall Rank)**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("*Most Improved*")
                    top_movers = changes_df.head(3)
                    for _, row in top_movers.iterrows():
                        change = row["Rank_Change"]
                        arrow = "+" if change > 0 else ""
                        st.markdown(
                            _stat_card(row["Team"], f"#{row['Current_Rank']:,}",
                                       detail=f"{arrow}{change:,} places"),
                            unsafe_allow_html=True,
                        )

                with col2:
                    st.markdown("*Biggest Drops*")
                    bottom_movers = changes_df.tail(3).iloc[::-1]
                    for _, row in bottom_movers.iterrows():
                        change = row["Rank_Change"]
                        arrow = "+" if change > 0 else ""
                        st.markdown(
                            _stat_card(row["Team"], f"#{row['Current_Rank']:,}",
                                       detail=f"{arrow}{change:,} places", accent="#f87171"),
                            unsafe_allow_html=True,
                        )

            st.divider()

            fig = plot_rank_progression(rank_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rank data available.")

    # ---------------------------
    # TAB 3: POINTS ANALYSIS
    # ---------------------------
    with tab3:
        st.subheader("Points Behind Leader")

        points_df = calculate_points_behind_leader(team_histories, team_names)

        if not points_df.empty:
            # Current standings summary
            latest_gw = points_df["Gameweek"].max()
            latest_df = points_df[points_df["Gameweek"] == latest_gw].sort_values("Points_Behind")

            st.markdown(f"**Current Standings (GW{latest_gw})**")

            display_latest = latest_df[["Team", "Total_Points", "Points_Behind"]].copy()
            display_latest.columns = ["Team", "Total Points", "Behind Leader"]

            render_styled_table(
                display_latest,
                col_formats={"Total Points": "{:,.0f}", "Behind Leader": "{:,.0f}"},
                negative_color_cols=["Behind Leader"],
            )

            st.divider()

            fig = plot_points_behind_leader(points_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No points data available.")

    # ---------------------------
    # TAB 4: TEAM VALUES
    # ---------------------------
    with tab4:
        st.subheader("Team Value Comparison")

        value_df = analyze_team_values(team_histories, team_names)

        if not value_df.empty:
            render_styled_table(
                value_df,
                col_formats={"Squad Value": "£{:.1f}m", "Bank": "£{:.1f}m",
                              "Total Value": "£{:.1f}m", "Value Gain": "{:+.1f}"},
                positive_color_cols=["Value Gain"],
            )

            st.divider()

            fig = plot_value_comparison(value_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No team value data available.")

    # ---------------------------
    # TAB 5: PERFORMANCE STATS
    # ---------------------------
    with tab5:
        st.subheader("Gameweek Performance Statistics")

        perf_df = analyze_gameweek_performance(team_histories, team_names)

        if not perf_df.empty:
            perf_display = perf_df.rename(columns={
                "Games": "GP", "Total_Points": "Total", "Avg_Points": "Avg/GW",
                "Best_GW": "Best", "Worst_GW": "Worst", "Std_Dev": "Std Dev",
            })
            render_styled_table(
                perf_display,
                col_formats={"Avg/GW": "{:.1f}", "Std Dev": "{:.1f}"},
                positive_color_cols=["Total", "Avg/GW"],
            )

            st.divider()

            weekly_scores = get_weekly_scores(team_histories, team_names)
            fig = plot_scoring_distribution(weekly_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available.")

    # ---------------------------
    # TAB 6: POINTS BY POSITION
    # ---------------------------
    with tab6:
        st.subheader("Points by Position")

        POSITION_COLORS = {
            "GK": "#f39c12",
            "DEF": "#3498db",
            "MID": "#2ecc71",
            "FWD": "#e74c3c",
        }

        # Determine latest played gameweek from team histories
        latest_gw = None
        for _tid, _hist in team_histories.items():
            gw_list = _hist.get("current", [])
            if gw_list:
                last_event = gw_list[-1].get("event")
                if last_event and (latest_gw is None or last_event > latest_gw):
                    latest_gw = last_event

        if latest_gw is None:
            st.info("Could not determine latest played gameweek.")
        else:
            with st.spinner(f"Loading position data for {len(team_ids)} teams (GW1-{latest_gw})..."):
                pos_rows = []
                for tid in team_ids:
                    tname = team_names.get(tid, f"Team {tid}")
                    result = get_classic_team_position_data(tid, latest_gw)
                    positions = result["positions"]
                    total = sum(positions.values())
                    pos_rows.append({
                        "Team": tname,
                        "GK": positions["GK"],
                        "DEF": positions["DEF"],
                        "MID": positions["MID"],
                        "FWD": positions["FWD"],
                        "Total": total,
                    })

            pos_df = pd.DataFrame(pos_rows)

            if pos_df.empty:
                st.info("No position data available yet.")
            else:
                pos_df = pos_df.sort_values("Total", ascending=False).reset_index(drop=True)

                # Toggle between raw points and percentage
                view_mode = st.radio(
                    "View",
                    ["Raw Points", "Percentage"],
                    horizontal=True,
                    key="classic_league_pos_view"
                )

                pos_cols = ["GK", "DEF", "MID", "FWD"]

                # Calculate league averages
                avg_row = {"Team": "League Average"}
                for col in pos_cols:
                    avg_row[col] = round(pos_df[col].mean(), 1)
                avg_row["Total"] = round(pos_df["Total"].mean(), 1)

                display_df = pos_df.copy()

                if view_mode == "Percentage":
                    for _, row in display_df.iterrows():
                        total = row["Total"]
                        if total > 0:
                            for col in pos_cols:
                                display_df.at[_, col] = round(row[col] / total * 100, 1)
                        else:
                            for col in pos_cols:
                                display_df.at[_, col] = 0.0
                    display_df["Total"] = 100.0

                    # Recalculate avg for percentage view
                    avg_total = avg_row["Total"]
                    if avg_total > 0:
                        for col in pos_cols:
                            avg_row[col] = round(avg_row[col] / avg_total * 100, 1)
                    avg_row["Total"] = 100.0

                # Append league average row
                avg_df_row = pd.DataFrame([avg_row])
                table_df = pd.concat([display_df, avg_df_row], ignore_index=True)

                # Style: gradient red-white-green based on distance from average
                def gradient_vs_avg(row):
                    styles = [""] * len(row)
                    if row["Team"] == "League Average":
                        styles = ["font-weight: bold; background-color: #f0f0f0"] * len(row)
                    else:
                        for i, col in enumerate(row.index):
                            if col in pos_cols:
                                avg_val = avg_row[col]
                                col_vals = table_df[table_df["Team"] != "League Average"][col]
                                col_min = col_vals.min()
                                col_max = col_vals.max()
                                val = row[col]

                                if val > avg_val and col_max > avg_val:
                                    t = min((val - avg_val) / (col_max - avg_val), 1.0)
                                    r = int(255 - t * (255 - 72))
                                    g = int(255 - t * (255 - 199))
                                    b = int(255 - t * (255 - 142))
                                    styles[i] = f"background-color: rgb({r},{g},{b})"
                                elif val < avg_val and col_min < avg_val:
                                    t = min((avg_val - val) / (avg_val - col_min), 1.0)
                                    r = int(255 - t * (255 - 248))
                                    g = int(255 - t * (255 - 105))
                                    b = int(255 - t * (255 - 107))
                                    styles[i] = f"background-color: rgb({r},{g},{b})"
                    return styles

                styled = table_df.style.apply(gradient_vs_avg, axis=1)

                suffix = "%" if view_mode == "Percentage" else " pts"
                styled = styled.format(
                    {col: f"{{:.1f}}{suffix}" for col in pos_cols + ["Total"]},
                    subset=pos_cols + ["Total"]
                )

                st.dataframe(styled, use_container_width=True, hide_index=True)

                st.divider()

                # Stacked bar chart
                bar_df = pos_df.melt(
                    id_vars=["Team"],
                    value_vars=pos_cols,
                    var_name="Position",
                    value_name="Points"
                )

                fig_bar = px.bar(
                    bar_df,
                    x="Team",
                    y="Points",
                    color="Position",
                    title="Points by Position (Team Breakdown)",
                    barmode="stack",
                    color_discrete_map=POSITION_COLORS,
                    category_orders={"Position": pos_cols}
                )
                fig_bar.update_layout(
                    **_DARK_CHART_LAYOUT,
                    xaxis_tickangle=-45,
                )
                fig_bar.update_xaxes(title="")
                fig_bar.update_yaxes(title="Total Points")
                st.plotly_chart(fig_bar, use_container_width=True)

                # League-wide pie chart
                league_totals = {col: pos_df[col].sum() for col in pos_cols}
                pie_df = pd.DataFrame({
                    "Position": list(league_totals.keys()),
                    "Points": list(league_totals.values())
                })

                fig_pie = px.pie(
                    pie_df,
                    values="Points",
                    names="Position",
                    title="League-Wide Points Distribution",
                    color="Position",
                    color_discrete_map=POSITION_COLORS,
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(
                    paper_bgcolor="#1a1a2e",
                    font=dict(color="#ffffff", size=14),
                    title=dict(font=dict(size=20, color="#ffffff"), x=0.5, xanchor="center"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
