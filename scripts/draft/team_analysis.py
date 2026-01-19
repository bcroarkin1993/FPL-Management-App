import config
import streamlit as st
from scripts.common.utils import get_league_player_ownership, get_league_teams, get_rotowire_player_projections, \
    get_team_composition_for_gameweek, get_team_id_by_name, merge_fpl_players_and_projections

def show_team_projections(team_id, fpl_player_projections, gameweek):
    # Get the team composition for the team in the current gameweek
    team_composition_df = get_team_composition_for_gameweek(config.FPL_DRAFT_LEAGUE_ID, team_id, gameweek)

    # Merge the FPL team df with the fpl_player_projections
    team_player_projections = merge_fpl_players_and_projections(team_composition_df, fpl_player_projections)

    # Format columns
    team_player_projections = team_player_projections[['Player', 'Team', 'Matchup', 'Position', 'Points', 'Pos Rank']]

    # Return the df
    return(team_player_projections)

def show_team_stats_page():
    st.title("Team Analysis")
    st.write("Displaying detailed statistics and projections for selected team.")

    # Pull FPL player projections from Rotowire
    player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    # Get FPL team names
    team_dict = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    team_list = list(team_dict.values())

    # Create a dropdown to select the team
    selected_team = st.selectbox("Select a Team", team_list)

    # Get the team ids based on the team names
    team_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, selected_team)

    # Display the team's player projected stats
    st.subheader(f"{selected_team} Projected Player Stats for Gameweek {config.CURRENT_GAMEWEEK}")
    st.dataframe(show_team_projections(team_id, player_projections, config.CURRENT_GAMEWEEK),
                 use_container_width=True, height=560)

    # Display the team's players by gameweek
    st.subheader("Display Team Composition by Gameweek")

    # Create dropdown to select Gameweek
    current_gameweek = config.CURRENT_GAMEWEEK
    gameweek = st.selectbox("Select Gameweek", list(range(1, current_gameweek + 1)))

    # Get and display the team composition for the selected gameweek
    if st.button("Show Team Composition"):
        # Get the team composition
        team_composition = get_team_composition_for_gameweek(config.FPL_DRAFT_LEAGUE_ID, team_id, gameweek)

        # Display the team's players for selected gameweek
        st.subheader(f"Team Composition for {selected_team} in Gameweek {gameweek}")
        st.write(team_composition)

