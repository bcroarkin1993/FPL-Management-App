import config
import pandas as pd
import random
import streamlit as st
from scripts.common.utils import (
    get_league_standings,
    get_classic_team_history,
)


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
        if league["name"]:
            name = league["name"]
        else:
            # Try to fetch name from API (works for both Classic and H2H)
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


def fetch_standings_data(league_id: int) -> dict:
    """
    Fetch league standings and metadata.
    Returns dict with 'league_info', 'standings', 'scoring_type'.
    """
    data = get_league_standings(league_id)

    if not data:
        return None

    league_info = data.get("league", {})
    standings_data = data.get("standings", {})

    # Determine scoring type from league info
    # Classic FPL uses 'league_type': 'x' for classic, 's' for H2H
    # Also check 'scoring' field if available
    league_type = league_info.get("league_type", "x")
    scoring = league_info.get("scoring", "c")  # 'c' = classic, 'h' = h2h

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
    """
    Format classic (total points) league standings.
    """
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
        })

    df = pd.DataFrame(rows)
    return df


def format_h2h_standings(standings_data: dict) -> pd.DataFrame:
    """
    Format H2H league standings.
    """
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
        })

    df = pd.DataFrame(rows)
    return df


def highlight_my_team(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    Return a styled DataFrame with the user's team highlighted.
    """
    # Get team history to find the team name
    history = get_classic_team_history(team_id)
    my_team_name = None

    if history:
        # The entry name might be in the team history somewhere
        # We need to fetch entry details for the team name
        from scripts.common.utils import get_entry_details
        entry = get_entry_details(team_id)
        if entry:
            my_team_name = entry.get("name")

    def highlight_row(row):
        if my_team_name and row.get("Team") == my_team_name:
            return ["background-color: #e6f3ff"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight_row, axis=1)


def show_classic_league_standings_page():
    """
    Display Classic FPL league standings with league selector.
    """
    st.title("Classic League Standings")

    # Get configured leagues
    league_options = get_league_display_options()

    if not league_options:
        st.warning("No Classic leagues configured. Add leagues to FPL_CLASSIC_LEAGUE_IDS in your .env file.")
        st.code("FPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League")
        return

    # Initialize random league selection on first load
    if "classic_league_index" not in st.session_state:
        st.session_state.classic_league_index = random.randint(0, len(league_options) - 1)

    # League selector
    display_options = [opt["display"] for opt in league_options]
    selected_display = st.selectbox(
        "Select League",
        options=display_options,
        index=st.session_state.classic_league_index,
        key="classic_league_selector"
    )

    # Update session state when user changes selection
    new_index = display_options.index(selected_display)
    if new_index != st.session_state.classic_league_index:
        st.session_state.classic_league_index = new_index

    # Get selected league ID
    selected_league = league_options[st.session_state.classic_league_index]
    league_id = selected_league["id"]

    # Fetch standings
    with st.spinner("Loading standings..."):
        data = fetch_standings_data(league_id)

    if not data:
        st.error(f"Failed to load standings for league {league_id}. Please check the league ID.")
        return

    # Display league info
    league_info = data["league_info"]
    scoring_type = data["scoring_type"]

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        league_name = league_info.get("name", "Unknown")
        st.markdown(f"### {league_name}")
    with col2:
        st.metric("Format", scoring_type)
    with col3:
        created = league_info.get("created", "")[:10] if league_info.get("created") else "Unknown"
        st.metric("Created", created)

    st.divider()

    # Format and display standings based on type
    standings = data["standings"]

    if scoring_type == "H2H":
        st.subheader("Head-to-Head Standings")
        df = format_h2h_standings(standings)

        if df.empty:
            st.info("No standings data available yet.")
        else:
            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Team": st.column_config.TextColumn("Team", width="medium"),
                    "Manager": st.column_config.TextColumn("Manager", width="medium"),
                    "W": st.column_config.NumberColumn("W", width="small"),
                    "D": st.column_config.NumberColumn("D", width="small"),
                    "L": st.column_config.NumberColumn("L", width="small"),
                    "PF": st.column_config.NumberColumn("Points For", width="small"),
                    "PA": st.column_config.NumberColumn("Points Against", width="small"),
                    "Pts": st.column_config.NumberColumn("League Pts", width="small"),
                }
            )
    else:
        st.subheader("Overall Standings")
        df = format_classic_standings(standings)

        if df.empty:
            st.info("No standings data available yet.")
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Team": st.column_config.TextColumn("Team", width="medium"),
                    "Manager": st.column_config.TextColumn("Manager", width="medium"),
                    "GW Pts": st.column_config.NumberColumn("GW Points", width="small"),
                    "Total Pts": st.column_config.NumberColumn("Total Points", width="small"),
                }
            )

    # Show pagination info if available
    has_next = standings.get("has_next", False)
    if has_next:
        st.caption("Showing first page of standings. Large leagues have additional pages.")

    # Show my team's position if configured
    if config.FPL_CLASSIC_TEAM_ID:
        st.divider()
        with st.expander("My Team Info"):
            history = get_classic_team_history(config.FPL_CLASSIC_TEAM_ID)
            if history and history.get("current"):
                latest_gw = history["current"][-1] if history["current"] else {}

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Rank", f"{latest_gw.get('overall_rank', 'N/A'):,}" if latest_gw.get('overall_rank') else "N/A")
                with col2:
                    st.metric("GW Rank", f"{latest_gw.get('rank', 'N/A'):,}" if latest_gw.get('rank') else "N/A")
                with col3:
                    st.metric("GW Points", latest_gw.get("points", "N/A"))
                with col4:
                    st.metric("Total Points", latest_gw.get("total_points", "N/A"))
