import config
import pandas as pd
import streamlit as st
from scripts.utils import find_optimal_lineup, get_current_gameweek, get_gameweek_fixtures, get_team_id_by_name, \
    get_rotowire_player_projections, get_team_composition_for_gameweek, merge_fpl_players_and_projections, \
    normalize_apostrophes

def analyze_fixture_projections(fixture, league_id, projections_df):
    """
    Returns two DataFrames representing optimal projected lineups and points for each team in a fixture,
    sorted by position (GK, DEF, MID, FWD) and then by descending projected points within each position.

    Parameters:
    - fixture (str): The selected fixture, formatted as "Team1 (Player1) vs Team2 (Player2)".
    - league_id (int): The ID of the FPL Draft league.
    - projections_df (DataFrame): DataFrame containing player projections from Rotowire.

    Returns:
    - Tuple of two DataFrames: (team1_df, team2_df, team1_name, team2_name)
    """
    # Normalize the apostrophes in the fixture string
    fixture = normalize_apostrophes(fixture)

    # Extract the team names only (ignore player names inside parentheses)
    team1_name = fixture.split(' vs ')[0].split(' (')[0].strip()
    team2_name = fixture.split(' vs ')[1].split(' (')[0].strip()

    # Get the team ids based on the team names
    team1_id = get_team_id_by_name(league_id, team1_name)
    team2_id = get_team_id_by_name(league_id, team2_name)

    # Get the current gameweek
    gameweek = get_current_gameweek()

    # Retrieve team compositions for the current gameweek and convert to dataframes
    team1_composition = get_team_composition_for_gameweek(league_id, team1_id, gameweek)
    team2_composition = get_team_composition_for_gameweek(league_id, team2_id, gameweek)

    # Merge FPL players with projections for both teams
    team1_df = merge_fpl_players_and_projections(
        team1_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    )
    team2_df = merge_fpl_players_and_projections(
        team2_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    )

    # Debugging: Check if 'Points' column exists
    if 'Points' not in team1_df or 'Points' not in team2_df:
        print("Error: 'Points' column not found in one or both dataframes.")
        print("Team 1 DataFrame:\n", team1_df.head())
        print("Team 2 DataFrame:\n", team2_df.head())
        return None  # Exit the function early if the column is missing

    # Fill NaN values in 'Points' column with 0.0
    team1_df['Points'] = pd.to_numeric(team1_df['Points'], errors='coerce').fillna(0.0)
    team2_df['Points'] = pd.to_numeric(team2_df['Points'], errors='coerce').fillna(0.0)

    # Find the optimal lineup (top 11 players) for each team
    team1_df = find_optimal_lineup(team1_df)
    team2_df = find_optimal_lineup(team2_df)

    # Define the position order for sorting
    position_order = ['G', 'D', 'M', 'F']
    for df in [team1_df, team2_df]:
        df['Position'] = pd.Categorical(df['Position'], categories=position_order, ordered=True)
        df.sort_values(by=['Position', 'Points'], ascending=[True, False], inplace=True)

    # Select the final columns to use
    team1_df = team1_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    team2_df = team2_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]

    # Format team DataFrames to use player names as the index
    team1_df.set_index('Player', inplace=True)
    team2_df.set_index('Player', inplace=True)

    # Return the final DataFrames and team names
    return team1_df, team2_df, team1_name, team2_name

def show_fixtures_page():
    st.title("Upcoming Fixtures & Projections")

    # Find the fixtures for the current gameweek
    gameweek_fixtures = get_gameweek_fixtures(config.FPL_DRAFT_LEAGUE_ID, config.CURRENT_GAMEWEEK)

    # Display each of the current gameweek fixtures
    if gameweek_fixtures:
        st.subheader(f"Gameweek {config.CURRENT_GAMEWEEK} Fixtures")
        for fixture in gameweek_fixtures:
            st.text(fixture)

    # Subheader for match projections
    st.subheader("Match Projections")

    # Pull FPL player projections from Rotowire
    fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    # Create a dropdown to choose a fixture
    fixture_selection = st.selectbox("Select a fixture to analyze deeper:", gameweek_fixtures)

    # Create the Streamlit visuals
    if fixture_selection:
        team1_df, team2_df, team1_name, team2_name = analyze_fixture_projections(fixture_selection,
                                                                                 config.FPL_DRAFT_LEAGUE_ID,
                                                                                 fpl_player_projections)

        # Create columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"{team1_name} Projections")
            st.dataframe(team1_df,
                         use_container_width=True,
                         height=422  # Adjust the height to ensure the entire table shows
                         )
            team1_score = team1_df['Points'].sum()
            st.markdown(f"**Projected Score: {team1_score:.2f}**")

        with col2:
            st.write(f"{team2_name}")
            st.dataframe(team2_df,
                         use_container_width=True,
                         height=422  # Adjust the height to ensure the entire table shows
                         )
            team2_score = team2_df['Points'].sum()
            st.markdown(f"**Projected Score: {team2_score:.2f}**")