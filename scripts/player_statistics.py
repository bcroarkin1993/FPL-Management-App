from st_aggrid import AgGrid
import streamlit as st
from scripts.utils import pull_fpl_player_stats, get_rotowire_player_projections

def show_league_stats_page():
    st.title("Detailed Player Statistics")
    st.write("Displaying detailed player and team statistics.")
    player_stats_df = pull_fpl_player_stats()
    if not player_stats_df.empty:
        st.subheader("FPL Player Statistics")
        AgGrid(player_stats_df)
    else:
        st.error("No data available at the URL provided.")

    player_projections = get_rotowire_player_projections()
