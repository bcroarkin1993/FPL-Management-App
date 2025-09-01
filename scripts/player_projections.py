import config
import streamlit as st
from scripts.utils import get_rotowire_player_projections, get_rotowire_rankings_url

def rotowire_url_selector():
    st.subheader("Rotowire Rankings URL")
    auto_url = get_rotowire_rankings_url()

    if auto_url:
        st.success(f"Auto-detected article URL:\n{auto_url}")
    else:
        st.warning("Could not auto-detect this week’s rankings article from Rotowire.")

    manual_url = st.text_input(
        "Override URL (optional)",
        value=auto_url or "",
        placeholder="https://www.rotowire.com/soccer/article/...",
        help="Paste the Rotowire rankings article URL here if auto-detect fails."
    )

    # Always return whichever is non-empty; manual takes precedence if edited
    final_url = manual_url.strip() or auto_url
    if not final_url:
        st.info("No URL selected yet.")
    return final_url

def show_player_projections_page():
    st.title("FPL Player Projections")
    st.write("Displaying GW player projections from Rotowire.")

    # Slider to limit the number of players shown in the rankings
    num_players1 = st.slider("Select the number of players to display:", min_value=5, max_value=250, value=100, step=5)

    # Pull FPL player projections from Rotowire
    if config.ROTOWIRE_URL:
        player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL, num_players1)
    # Allow user to add FPL player projection URL if the auto-retrieval fails
    else:
        url = rotowire_url_selector()
        player_projections = get_rotowire_player_projections(url, num_players1)

    # Limit columns to show in player_projections
    player_projections = player_projections[['Player', 'Position', 'Team', 'Matchup', 'Points', 'Pos Rank']]

    # Text input to filter by player name
    player_filter = st.text_input("Filter by Player Name", value="")

    # Multiselect to filter by position (allows multiple positions to be selected)
    all_positions = player_projections['Position'].unique().tolist()
    position_filter = st.multiselect("Filter by Position", options=all_positions, default=all_positions)

    # Apply filtering based on player name
    if player_filter:
        player_projections = player_projections[player_projections['Player'].str.contains(player_filter, case=False, na=False)]

    # Apply filtering based on selected positions
    if position_filter:
        player_projections = player_projections[player_projections['Position'].isin(position_filter)]

    # Display FPL player rankings
    st.subheader(f"GW {config.CURRENT_GAMEWEEK} Player Rankings")
    st.dataframe(player_projections, use_container_width=True)