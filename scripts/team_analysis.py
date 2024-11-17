import config
import streamlit as st
from utils import get_league_player_dict_for_gameweek, get_league_teams, get_rotowire_player_projections, \
    get_team_composition_for_gameweek, merge_fpl_players_and_projections

def show_team_projections(selected_team, fpl_player_projections, gameweek):
    # Get the team composition for the team in the current gameweek
    team_composition_df = get_team_composition_for_gameweek(config.BRANDON_DRAFT_LEAGUE_ID, selected_team, gameweek)

    # Merge the FPL team df with the fpl_player_projections
    team_player_projections = merge_fpl_players_and_projections(team_composition_df, fpl_player_projections)

    # Format columns
    team_player_projections = team_player_projections[['Player', 'Team', 'Matchup', 'Position', 'Points', 'Pos Rank']]

    # Return the df
    return(team_player_projections)

def show_team_stats_page():
    st.title("Detailed Team Statistics")
    st.write("Displaying detailed statistics and projections for all players on a given team.")

    # Pull FPL player projections from Rotowire
    player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    # Pull the FPL team dict
    team_dict = get_league_player_dict_for_gameweek(config.BRANDON_DRAFT_LEAGUE_ID, config.CURRENT_GAMEWEEK)

    # Format it as a list
    team_list = list(team_dict.keys())

    # Dropdown to select the team
    selected_team = st.selectbox("Select a Team", team_list, index=team_list.index("CHANGE NAME"))

    # Display the team's player projected stats
    st.subheader(f"{selected_team} Projected Player Stats for Gameweek {config.CURRENT_GAMEWEEK}")
    st.dataframe(show_team_projections(selected_team, player_projections, config.CURRENT_GAMEWEEK),
                 use_container_width=True, height=560)

    # Display the team's players by gameweek
    st.subheader("Display Team Composition by Gameweek")

    # Step 1: Team names
    team_dict = get_league_teams(config.BRANDON_DRAFT_LEAGUE_ID)
    teams = list(team_dict.values())

    # Step 2: Layout for dropdowns in two columns
    col1, col2 = st.columns(2)

    with col1:
        team_name = st.selectbox("Select Team", teams)

    with col2:
        current_gameweek = config.CURRENT_GAMEWEEK
        gameweek = st.selectbox("Select Gameweek", list(range(1, current_gameweek + 1)))

    # Step 3: Get and display the team composition for the selected gameweek
    if st.button("Show Team Composition"):
        # Get the team composition
        team_composition = get_team_composition_for_gameweek(config.BRANDON_DRAFT_LEAGUE_ID, team_name, gameweek)

        # Display the team's players for selected gameweek
        st.subheader(f"Team Composition for {team_name} in Gameweek {gameweek}")
        st.write(team_composition)

