"""
Classic FPL - Home Page

Displays league standings, points progression charts, and rank trends
for Classic FPL leagues. Supports multiple leagues via dropdown.
"""

import config
import pandas as pd
import plotly.express as px
import random
import streamlit as st
from typing import Optional, Dict, List

from scripts.common.luck_analysis import extract_classic_h2h_gw_scores, calculate_all_play_standings, render_luck_adjusted_table, render_standings_table
from scripts.common.utils import (
    get_league_standings,
    get_classic_team_history,
    get_entry_details,
    get_current_gameweek,
    get_all_h2h_league_matches,
)


# ---------------------------
# LEAGUE DATA FUNCTIONS
# ---------------------------

def get_league_display_options() -> list:
    """
    Build list of league options for the dropdown.
    Returns list of dicts with 'id', 'name', 'display' keys.
    """
    leagues = config.FPL_CLASSIC_LEAGUE_IDS or []
    options = []

    for league in leagues:
        league_id = league["id"]
        # Use configured name or fetch from API
        if league.get("name"):
            name = league["name"]
        else:
            # Try to fetch name from API
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


def fetch_standings_data(league_id: int) -> Optional[dict]:
    """
    Fetch league standings and metadata.
    Returns dict with 'league_info', 'standings', 'scoring_type'.
    """
    data = get_league_standings(league_id)

    if not data:
        return None

    league_info = data.get("league", {})
    standings_data = data.get("standings", {})

    # Determine scoring type
    league_type = league_info.get("league_type", "x")
    scoring = league_info.get("scoring", "c")

    if league_type == "s" or scoring == "h":
        scoring_type = "H2H"
    else:
        scoring_type = "Classic"

    return {
        "league_info": league_info,
        "standings": standings_data,
        "scoring_type": scoring_type
    }


def format_classic_standings(standings_data: dict) -> pd.DataFrame:
    """Format classic (total points) league standings."""
    results = standings_data.get("results", [])

    if not results:
        return pd.DataFrame()

    rows = []
    for entry in results:
        rows.append({
            "Rank": entry.get("rank", "-"),
            "Team": entry.get("entry_name", "Unknown"),
            "Manager": entry.get("player_name", "Unknown"),
            "GW Pts": entry.get("event_total", 0),
            "Total Pts": entry.get("total", 0),
            "entry_id": entry.get("entry", 0),
        })

    return pd.DataFrame(rows)


def format_h2h_standings(standings_data: dict) -> pd.DataFrame:
    """Format H2H league standings."""
    results = standings_data.get("results", [])

    if not results:
        return pd.DataFrame()

    rows = []
    for entry in results:
        rows.append({
            "Rank": entry.get("rank", "-"),
            "Team": entry.get("entry_name", "Unknown"),
            "Manager": entry.get("player_name", "Unknown"),
            "W": entry.get("matches_won", 0),
            "D": entry.get("matches_drawn", 0),
            "L": entry.get("matches_lost", 0),
            "PF": entry.get("points_for", 0),
            "PA": entry.get("points_against", 0),
            "Pts": entry.get("total", 0),
            "entry_id": entry.get("entry", 0),
        })

    return pd.DataFrame(rows)


def get_classic_h2h_luck_adjusted(league_id: int, standings_data: dict) -> pd.DataFrame:
    """
    Calculate luck-adjusted standings for a Classic H2H league using All-Play Record.

    Parameters:
    - league_id: The H2H FPL league ID.
    - standings_data: The standings dict from the league API (passed to avoid re-fetching).

    Returns:
    - DataFrame with All-Play standings.
    """
    matches = get_all_h2h_league_matches(league_id)
    if not matches:
        return pd.DataFrame()

    gw_scores = extract_classic_h2h_gw_scores(matches)
    if gw_scores.empty:
        return pd.DataFrame()

    # Build actual standings from formatted H2H standings
    h2h_df = format_h2h_standings(standings_data)
    if h2h_df.empty:
        return calculate_all_play_standings(gw_scores)

    actual_standings = h2h_df[['entry_id', 'Team', 'Rank', 'Pts']].rename(columns={
        'entry_id': 'team_id',
        'Team': 'team',
        'Rank': 'actual_rank',
        'Pts': 'actual_pts',
    })

    return calculate_all_play_standings(gw_scores, actual_standings)


# ---------------------------
# CHART FUNCTIONS
# ---------------------------

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title=dict(font=dict(size=22, color="#ffffff"), x=0.5, xanchor="center"),
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
)

@st.cache_data(ttl=600)
def fetch_team_histories(team_ids: List[int], team_names: Dict[int, str]) -> pd.DataFrame:
    """
    Fetch gameweek history for multiple teams.
    Returns DataFrame with columns: Team, Gameweek, GW_Points, Total_Points, Overall_Rank
    """
    all_data = []

    for team_id in team_ids:
        history = get_classic_team_history(team_id)
        if not history or not history.get("current"):
            continue

        team_name = team_names.get(team_id, f"Team {team_id}")

        for gw_data in history["current"]:
            all_data.append({
                "Team": team_name,
                "Gameweek": gw_data.get("event"),
                "GW_Points": gw_data.get("points", 0),
                "Total_Points": gw_data.get("total_points", 0),
                "Overall_Rank": gw_data.get("overall_rank"),
                "GW_Rank": gw_data.get("rank"),
            })

    return pd.DataFrame(all_data)


def plot_total_points_over_time(history_df: pd.DataFrame) -> Optional[px.line]:
    """Create line chart of total points over time."""
    if history_df.empty:
        return None

    fig = px.line(
        history_df,
        x="Gameweek",
        y="Total_Points",
        color="Team",
        title="Total Points Over Time",
        labels={
            "Total_Points": "Total Points",
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="Total Points",
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )

    return fig


def plot_gw_points_over_time(history_df: pd.DataFrame) -> Optional[px.line]:
    """Create line chart of gameweek points over time."""
    if history_df.empty:
        return None

    fig = px.line(
        history_df,
        x="Gameweek",
        y="GW_Points",
        color="Team",
        title="Gameweek Points Over Time",
        labels={
            "GW_Points": "GW Points",
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="GW Points",
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )

    return fig


def plot_rank_progression(history_df: pd.DataFrame, use_overall_rank: bool = False) -> Optional[px.line]:
    """Create line chart of rank over time (inverted - lower is better)."""
    if history_df.empty:
        return None

    rank_col = "Overall_Rank" if use_overall_rank else "GW_Rank"
    title = "Overall Rank Progression" if use_overall_rank else "Gameweek Rank Progression"
    y_label = "Overall Rank" if use_overall_rank else "GW Rank"

    # Filter out rows without rank data
    plot_df = history_df.dropna(subset=[rank_col])
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="Gameweek",
        y=rank_col,
        color="Team",
        title=title,
        labels={
            rank_col: y_label,
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    # Invert y-axis (rank 1 at top)
    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title=y_label,
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")

    return fig


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_classic_home_page():
    """Display the Classic FPL Home page with standings and charts."""

    st.title("Classic FPL League")

    # Get configured leagues
    league_options = get_league_display_options()

    if not league_options:
        st.warning("No Classic leagues configured.")
        st.info(
            "Add leagues to `FPL_CLASSIC_LEAGUE_IDS` in your `.env` file:\n\n"
            "```\nFPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League\n```\n\n"
            "The format is `league_id:League Name` with multiple leagues separated by commas."
        )
        return

    # Initialize random league selection on first load
    if "classic_home_league_index" not in st.session_state:
        st.session_state.classic_home_league_index = 0

    # League selector
    display_options = [opt["display"] for opt in league_options]
    selected_display = st.selectbox(
        "Select League",
        options=display_options,
        index=st.session_state.classic_home_league_index,
        key="classic_home_league_selector"
    )

    # Update session state when user changes selection
    new_index = display_options.index(selected_display)
    if new_index != st.session_state.classic_home_league_index:
        st.session_state.classic_home_league_index = new_index

    # Get selected league ID
    selected_league = league_options[st.session_state.classic_home_league_index]
    league_id = selected_league["id"]

    # Fetch standings
    with st.spinner("Loading league data..."):
        data = fetch_standings_data(league_id)

    if not data:
        st.error(f"Failed to load standings for league {league_id}. Please check the league ID.")
        return

    # Display league info header
    league_info = data["league_info"]
    scoring_type = data["scoring_type"]

    col1, col2 = st.columns([3, 1])
    with col1:
        league_name = league_info.get("name", "Unknown")
        st.markdown(f"### {league_name}")
    with col2:
        st.metric("Format", scoring_type)

    st.divider()

    # ---------------------------
    # STANDINGS TABLE
    # ---------------------------
    standings = data["standings"]

    if scoring_type == "H2H":
        st.subheader("League Standings")
        df = format_h2h_standings(standings)

        if df.empty:
            st.info("No standings data available yet.")
        else:
            show_luck = st.checkbox(
                "Show Luck Adjusted Standings",
                key=f"luck_{league_id}",
            )

            if show_luck:
                st.caption("**All-Play Record**: simulates every team playing every other team each gameweek. "
                           "Luck +/- shows how much the actual schedule helped or hurt (positive = lucky).")
                with st.spinner("Calculating all-play standings..."):
                    luck_df = get_classic_h2h_luck_adjusted(league_id, standings)

                render_luck_adjusted_table(luck_df)
            else:
                # Don't show entry_id in display
                display_df = df.drop(columns=["entry_id"], errors="ignore")
                render_standings_table(display_df, is_h2h=True)
    else:
        st.subheader("League Standings")
        df = format_classic_standings(standings)

        if df.empty:
            st.info("No standings data available yet.")
        else:
            # Don't show entry_id in display
            display_df = df.drop(columns=["entry_id"], errors="ignore")
            render_standings_table(display_df, is_h2h=False)

    # Show pagination info
    has_next = standings.get("has_next", False)
    if has_next:
        st.caption("Showing first page of standings. Large leagues have additional pages.")

    st.divider()

    # ---------------------------
    # POINTS PROGRESSION CHARTS
    # ---------------------------
    if df.empty:
        st.info("Charts will be available once the season starts.")
        return

    st.subheader("Season Progression")

    # Slider to select how many teams to show in charts
    num_teams = len(df)
    if num_teams > 5:
        teams_to_show = st.slider(
            "Number of teams to show in charts",
            min_value=3,
            max_value=min(num_teams, 20),
            value=min(num_teams, 10),
            help="Limit teams shown for cleaner charts. Teams are selected by current rank."
        )
    else:
        teams_to_show = num_teams

    # Get top N teams for charts
    top_teams_df = df.head(teams_to_show)
    team_ids = top_teams_df["entry_id"].tolist()
    team_names = dict(zip(top_teams_df["entry_id"], top_teams_df["Team"]))

    # Fetch history for these teams
    with st.spinner("Loading team histories..."):
        history_df = fetch_team_histories(team_ids, team_names)

    if history_df.empty:
        st.info("No gameweek data available yet. Charts will appear once games are played.")
        return

    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Total Points", "Gameweek Points", "Rank Progression"])

    with tab1:
        fig = plot_total_points_over_time(history_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display chart.")

    with tab2:
        fig = plot_gw_points_over_time(history_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display chart.")

    with tab3:
        # Option to show overall rank or GW rank
        rank_type = st.radio(
            "Rank Type",
            ["Gameweek Rank", "Overall Rank"],
            horizontal=True
        )
        use_overall = rank_type == "Overall Rank"

        fig = plot_rank_progression(history_df, use_overall_rank=use_overall)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lower rank = better position (rank 1 is best)")
        else:
            st.info("Not enough data to display chart.")

    st.divider()

    # ---------------------------
    # MY TEAM INFO
    # ---------------------------
    if config.FPL_CLASSIC_TEAM_ID:
        with st.expander("My Team Summary"):
            history = get_classic_team_history(config.FPL_CLASSIC_TEAM_ID)
            entry = get_entry_details(config.FPL_CLASSIC_TEAM_ID)

            if entry:
                st.markdown(f"**{entry.get('name', 'Unknown Team')}**")
                st.caption(f"Manager: {entry.get('player_first_name', '')} {entry.get('player_last_name', '')}")

            if history and history.get("current"):
                latest_gw = history["current"][-1] if history["current"] else {}

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    overall_rank = latest_gw.get('overall_rank')
                    st.metric("Overall Rank", f"{overall_rank:,}" if overall_rank else "N/A")
                with col2:
                    gw_rank = latest_gw.get('rank')
                    st.metric("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A")
                with col3:
                    st.metric("GW Points", latest_gw.get("points", "N/A"))
                with col4:
                    st.metric("Total Points", latest_gw.get("total_points", "N/A"))

                # Show chips used
                chips = history.get("chips", [])
                if chips:
                    st.markdown("**Chips Used:**")
                    chip_names = {
                        "wildcard": "Wildcard",
                        "freehit": "Free Hit",
                        "bboost": "Bench Boost",
                        "3xc": "Triple Captain"
                    }
                    chip_text = ", ".join([
                        f"{chip_names.get(c.get('name', ''), c.get('name', ''))} (GW{c.get('event', '?')})"
                        for c in chips
                    ])
                    st.caption(chip_text)
            else:
                st.info("Could not load team history.")
