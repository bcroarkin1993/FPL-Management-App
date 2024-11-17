import config
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import requests
import streamlit as st
from utils import get_current_gameweek

# Set page config
st.set_page_config(layout="wide")

def checkValidLineup(df):
    """
    Given a dataframe with a lineup, check to see if it is a valid lineup.
    Requirements:
    - 11 total players
    - 1 GK
    - Min of 3 DEF
    - Max of 5 DEF
    - Min of 3 MID
    - Max of 5 MID
    - Min of 1 FWD
    - Max of 3 MID
    """
    # Check the total players count
    players = len(df)

    # Count occurrences of each value in the 'Position' column
    position_counts = df['Position'].value_counts()

    # Perform the checks
    player_check = players == 11
    gk_check = position_counts['GK'] == 1
    def_check = position_counts['DEF'] >= 3 and position_counts['DEF'] <= 5
    mid_check = position_counts['MID'] >= 3 and position_counts['MID'] <= 5
    fwd_check = position_counts['FWD'] >= 1 and position_counts['FWD'] <= 3

    print("Player Check :", player_check)
    print("GK Check :", gk_check)
    print("DEF Check :", def_check)
    print("MID Check :", mid_check)
    print("FWD Check :", fwd_check)

    # Lineup is valid is all checks are true
    return (player_check & gk_check & def_check & mid_check & fwd_check)

def findOptimalLineup(df):
    # 1. Find the top scoring GK
    top_gk = df[df['Position'] == 'GK'].sort_values(by='Projected_Points', ascending=False).head(1)

    # 2. Find the top 3 scoring DEF
    top_def = df[df['Position'] == 'DEF'].sort_values(by='Projected_Points', ascending=False).head(3)

    # 3. Find the top 3 scoring MID
    top_mid = df[df['Position'] == 'MID'].sort_values(by='Projected_Points', ascending=False).head(3)

    # 4. Find the top scoring FWD
    top_fwd = df[df['Position'] == 'FWD'].sort_values(by='Projected_Points', ascending=False).head(1)

    # 5. Combine the selected players
    selected_players = pd.concat([top_gk, top_def, top_mid, top_fwd])

    # 6. Find the remaining top 3 scoring players not in the selected players
    remaining_players = df[~df['Player'].isin(selected_players['Player'])]
    top_remaining = remaining_players.sort_values(by='Projected_Points', ascending=False).head(3)

    # 7. Combine all selected players
    final_selection = pd.concat([selected_players, top_remaining])

    # 8. Organize the final selection by Position and descending Projected_Points
    final_selection = final_selection.sort_values(
        by=['Position', 'Projected_Points'],
        key=lambda x: x.map({'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}),
        ascending=[True, False]
    ).reset_index(drop=True)

    # Check if valid lineup
    if checkValidLineup(final_selection):
        # Display the final selection
        return(final_selection)
    else:
        print("Invalid Lineup")
        return(final_selection)

def get_fpl_draft_league_standings(draft_league_id, show_advanced_stats=False):
    """
    Fetches and displays the current league standings for an FPL Draft league, with an option to show advanced statistics.

    Parameters:
    - draft_league_id (int or str): The ID of the FPL Draft league for which to fetch standings data.
    - show_advanced_stats (bool): If True, the function will include advanced statistics such as luck-adjusted standings.

    Workflow:
    1. The function retrieves league details (standings and league entries) from the FPL Draft API.
    2. It extracts the standings and team/player information into separate DataFrames.
    3. The two DataFrames are merged on the team’s league entry ID to combine the standings with player information.
    4. The function renames and reformats the DataFrame, calculating the points based on wins (3 points per win) and draws (1 point per draw).
    5. It ranks teams based on their points, with higher points resulting in a better rank.
    6. Optionally, if `show_advanced_stats` is set to True, it computes advanced statistics such as the luck-adjusted standings
       and merges them with the basic standings.

    Returns:
    - DataFrame: A pandas DataFrame containing the league standings, including:
        - Rank: The team's rank based on points.
        - Team: The name of the team.
        - Player: The full name of the player managing the team.
        - W: The number of matches won.
        - D: The number of matches drawn.
        - L: The number of matches lost.
        - PF: The total points scored by the team (points for).
        - PA: The total points scored against the team (points against).
        - Pts: The calculated points from wins and draws.
        - Optionally, if `show_advanced_stats` is True:
            - Avg_Week_Rank: The average rank based on weekly performance.
            - Avg_Score: The average score of the team across all gameweeks.
            - Fair_Rank: The team's fair rank based on weekly performance (luck-adjusted standings).
            - Luck_Index: The difference between the actual rank and the fair rank.

    Example Usage:
    standings_df = get_draft_league_standings(49249, show_advanced_stats=True)  # Fetch league standings for league 49249 with advanced stats
    print(standings_df)
    """
    # Draft League API URLs
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"

    # Get league details, fixtures, and team details
    league_response = requests.get(league_url).json()

    # Extract the standings and league_entries sections of the JSON
    standings = league_response['standings']
    league_entries = league_response['league_entries']

    # Create DataFrame for standings
    standings_df = pd.DataFrame(standings)

    # Create DataFrame for league entries
    entries_df = pd.DataFrame(league_entries)

    # Merge standings with league entries on 'league_entry' and 'entry_id'
    league_standings_df = standings_df.merge(entries_df, left_on='league_entry', right_on='id', how='left')

    # Select and rename the required columns
    league_standings_df = league_standings_df[['entry_name', 'player_first_name', 'player_last_name', 'matches_won',
                                               'matches_drawn', 'matches_lost', 'points_for', 'points_against']]
    league_standings_df.columns = ['Team', 'Player Name', 'Player Last Name', 'W', 'D', 'L', 'PF', 'PA']

    # Concatenating 'Player Name' and 'Player Last Name' using .loc[] to avoid the warning
    league_standings_df.loc[:, 'Player'] = league_standings_df['Player Name'] + ' ' + league_standings_df[
        'Player Last Name']

    # Drop the 'Player Last Name' column if no longer needed
    league_standings_df = league_standings_df.drop(columns=['Player Last Name'])

    # Calculate the 'Pts' based on wins and draws
    league_standings_df['Pts'] = league_standings_df['W'] * 3 + league_standings_df['D'] * 1

    # Convert 'Total Points' to numeric for ranking
    league_standings_df['PF'] = pd.to_numeric(league_standings_df['PF'], errors='coerce')

    # Add a Rank column based on 'Total Points'
    league_standings_df['Rank'] = league_standings_df['Pts'].rank(ascending=False, method='min').astype(int)

    # Sort by Rank
    league_standings_df = league_standings_df.sort_values('Rank')

    # Reorder the columns
    league_standings_df = league_standings_df[['Rank', 'Team', 'Player', 'W', 'D', 'L', 'PF', 'PA', 'Pts']]

    # If show_advanced_stats is true, add the avg_gw_rank and avg_gw_score to the dataframe
    if show_advanced_stats == True:
        # Compute the luck adjusted standings
        luck_adjusted_df = get_luck_adjusted_league_standings(draft_league_id)
        # Merge with the league_standings
        advanced_league_standings_df = pd.merge(league_standings_df, luck_adjusted_df, on='Team')
        # Sort by Fair Rank
        advanced_league_standings_df = advanced_league_standings_df.sort_values('Fair_Rank')
        # Format the merged results
        advanced_league_standings_df = advanced_league_standings_df[['Fair_Rank', 'Team', 'Player', 'W', 'D', 'L',
                                                                     'PF', 'PA', 'Avg_GW_Rank', 'Avg_Score', 'Pts']]
        # Set the index
        advanced_league_standings_df.set_index('Fair_Rank', inplace=True)
        # Return the results
        return (advanced_league_standings_df)
    else:
        # Set the index
        league_standings_df.set_index('Rank', inplace=True)
        # Return the results
        return (league_standings_df)

def show_fixture_results(draft_league_id, gameweek):
    # Draft League API URLs
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"

    # Get league details, fixtures, and team details
    league_response = requests.get(league_url).json()

    # Extract the standings and league_entries sections of the JSON
    fixtures = league_response['matches']
    league_entries = league_response['league_entries']
    # Create a dictionary mapping entry_ids to team names
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Create a list to hold fixture details
    fixtures_list = []

    for fixture in fixtures:
        # Filter to just the gameweek of interest
        if fixture['event'] == gameweek:
            team1_id = fixture['league_entry_1']
            team2_id = fixture['league_entry_2']
            team1_name = team_names.get(team1_id)
            team2_name = team_names.get(team2_id)
            team1_score = fixture['league_entry_1_points']
            team2_score = fixture['league_entry_2_points']

            if (team1_score != 0) & (team2_score != 0):
                # Determine match results based on scores
                if team1_score > team2_score:
                    team1_result = 'W'
                    team2_result = 'L'
                elif team1_score < team2_score:
                    team1_result = 'L'
                    team2_result = 'W'
                else:
                    team1_result = team2_result = 'D'

                fixtures_list.append({
                    "Team 1": team1_name,
                    "Team 1 Score": team1_score,
                    "Team 1 Result": team1_result,
                    "Team 2": team2_name,
                    "Team 2 Score": team2_score,
                    "Team 2 Result": team2_result,
                })

    # Convert the list into a DataFrame
    if fixtures_list:
        fixtures_df = pd.DataFrame(fixtures_list)
        st.write(f"Results for Gameweek {gameweek}:")
        st.dataframe(fixtures_df, use_container_width=True)
    else:
        st.write(f"No results available for Gameweek {gameweek}.")

def get_luck_adjusted_league_standings(draft_league_id):
    """
    Calculates the luck-adjusted league standings based on gameweek performance and fixture results.

    Parameters:
    - fixtures_df: A DataFrame with the following columns:
        - 'Team 1', 'Team 1 Score', 'Team 1 Result', 'Team 2', 'Team 2 Score', 'Team 2 Result'

    Returns:
    - standings_df: A DataFrame showing the adjusted standings based on luck and average weekly rank.
    """
    # Draft League API URLs
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"

    # Get league details, fixtures, and team details
    league_response = requests.get(league_url).json()

    # Extract the standings and league_entries sections of the JSON
    fixtures = league_response['matches']
    league_entries = league_response['league_entries']
    # Create a dictionary mapping entry_ids to team names
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Create a list to hold fixture details
    fixtures_list = []

    for fixture in fixtures:
        team1_id = fixture['league_entry_1']
        team2_id = fixture['league_entry_2']
        team1_name = team_names.get(team1_id)
        team2_name = team_names.get(team2_id)
        team1_score = fixture['league_entry_1_points']
        team2_score = fixture['league_entry_2_points']

        if (team1_score != 0) & (team2_score != 0):
            # Determine match results based on scores
            if team1_score > team2_score:
                team1_result = 'W'
                team2_result = 'L'
            elif team1_score < team2_score:
                team1_result = 'L'
                team2_result = 'W'
            else:
                team1_result = team2_result = 'D'

            fixtures_list.append({
                "Team 1": team1_name,
                "Team 1 Score": team1_score,
                "Team 1 Result": team1_result,
                "Team 2": team2_name,
                "Team 2 Score": team2_score,
                "Team 2 Result": team2_result,
            })

    # Convert fixtures_list to dataframe
    fixtures_df = pd.DataFrame(fixtures_list)

    # Step 1: Calculate the Week_Rank for each team based on scores for each gameweek
    fixtures_df['Week'] = fixtures_df.index // (len(fixtures_df) // len(fixtures_df['Team 1'].unique())) + 1

    # Create a long-format dataframe with each team's score and result for the week
    long_format_df = pd.concat([
        fixtures_df[['Team 1', 'Team 1 Score', 'Team 1 Result', 'Week']].rename(
            columns={'Team 1': 'Team', 'Team 1 Score': 'Score', 'Team 1 Result': 'Result'}),
        fixtures_df[['Team 2', 'Team 2 Score', 'Team 2 Result', 'Week']].rename(
            columns={'Team 2': 'Team', 'Team 2 Score': 'Score', 'Team 2 Result': 'Result'})
    ])

    # Step 2: Calculate the rank for each team in each week based on their score
    long_format_df['Week_Rank'] = long_format_df.groupby('Week')['Score'].rank(ascending=False, method='min')

    # Step 3: Calculate the actual results for each team based on their results (W, L, D)
    long_format_df['Points'] = long_format_df['Result'].apply(lambda x: 3 if x == 'W' else (1 if x == 'D' else 0))

    # Step 4: Calculate the average weekly rank (Fair_Rank) and total points (actual standings) for each team
    team_stats_df = long_format_df.groupby('Team').agg(
        Avg_GW_Rank=('Week_Rank', 'mean'),  # The "fair" rank based on average weekly score rank
        Total_Points=('Points', 'sum'),  # The actual points based on match results
        Avg_Score=('Score', 'mean')  # The average score across all weeks
    ).reset_index()

    # Step 5: Create the adjusted standings by ranking teams based on Avg_Week_Rank
    team_stats_df['Fair_Rank'] = team_stats_df['Avg_GW_Rank'].rank(ascending=True, method='min').astype(int)

    # Rank the teams based on their actual points (Total_Points)
    team_stats_df['Actual_Rank'] = team_stats_df['Total_Points'].rank(ascending=False, method='min')

    # Step 6: Add luck index: Difference between actual rank and fair rank
    team_stats_df['Luck_Index'] = team_stats_df['Fair_Rank'] - team_stats_df['Actual_Rank']

    # Step 7: Sort by Fair_Rank to get the adjusted standings
    team_stats_df = team_stats_df.sort_values(by='Fair_Rank')

    # Step 8: Format number values
    team_stats_df['Avg_GW_Rank'] = round(team_stats_df['Avg_GW_Rank'], 2)
    team_stats_df['Avg_Score'] = round(team_stats_df['Avg_Score'], 2)

    return team_stats_df

def plot_league_points_over_time(draft_league_id):
    """
    Fetches the gameweek results for all teams in an FPL Draft league and plots the total league points
    over time based on wins, ties, and losses. Points are awarded: 3 for a win, 1 for a tie, and 0 for a loss.

    Parameters:
    - draft_league_id (int or str): The ID of the FPL Draft league to retrieve data from.

    Returns:
    - Plotly figure displaying the total league points by team over time (gameweek).
    """
    # Step 1: Get the current gameweek
    current_gameweek = get_current_gameweek()

    # Step 2: Fetch league details
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"
    league_response = requests.get(league_url).json()

    # Step 3: Create a dictionary mapping entry_ids to team names
    league_entries = league_response['league_entries']
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Step 4: Initialize league points for each team and data for plotting
    league_points = {team_id: 0 for team_id in team_names.keys()}  # Holds cumulative league points
    data = []  # To store data for plotting

    # Step 5: Process match results and accumulate points by gameweek
    matches = league_response['matches']
    for match in matches:
        gameweek = match['event']
        if gameweek > current_gameweek:  # Ignore future matches
            continue

        team1_id = match['league_entry_1']
        team2_id = match['league_entry_2']
        team1_points = match['league_entry_1_points']
        team2_points = match['league_entry_2_points']

        # Determine match result and update league points
        if team1_points > team2_points:
            league_points[team1_id] += 3  # 3 points for a win
        elif team1_points < team2_points:
            league_points[team2_id] += 3  # 3 points for a win
        else:
            team1_result = team2_result = 'T'
            league_points[team1_id] += 1  # 1 point for a tie
            league_points[team2_id] += 1

        # Step 6: Append data for both teams to track their points by gameweek
        data.append(
            {'Team': team_names[team1_id], 'Gameweek': gameweek, 'Total League Points': league_points[team1_id]})
        data.append(
            {'Team': team_names[team2_id], 'Gameweek': gameweek, 'Total League Points': league_points[team2_id]})

    # Step 7: Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Step 8: Plot the data using Plotly
    fig = px.line(df, x="Gameweek", y="Total League Points", color="Team",
                  labels={"Total League Points": "Total League Points", "Gameweek": "Gameweek", "Team": "Team"},
                  title="Team Total League Points Over Time (Gameweek)")

    # Customize the layout
    fig.update_layout(xaxis_title="Gameweek", yaxis_title="Total League Points", hovermode="x unified")

    # Return the figure
    return(fig)

def plot_team_points_over_time(draft_league_id):
    """
    Fetches the gameweek results for all teams in an FPL Draft league and plots the total points over time for each team.
    Only includes matches that have been played (up to the current gameweek).

    Parameters:
    - draft_league_id (int or str): The ID of the FPL Draft league to retrieve data from.

    Returns:
    - Plotly figure displaying the total points by team over time (gameweek).
    """
    # Step 1: Get the current gameweek
    current_gameweek = get_current_gameweek()

    # Step 2: Fetch league details
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"
    league_response = requests.get(league_url).json()

    # Step 3: Create a dictionary mapping entry_ids to team names
    league_entries = league_response['league_entries']
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Step 4: Extract matches data from the response
    matches = league_response['matches']

    # Step 5: Create a DataFrame to track points for each team over time
    total_points = {team_id: 0 for team_id in team_names.keys()}  # Dictionary to hold cumulative points
    data = []

    # Step 6: Process each match and accumulate points by gameweek, ignoring future matches
    for match in matches:
        gameweek = match['event']
        if gameweek >= current_gameweek:  # Ignore future matches
            continue

        team1_id = match['league_entry_1']
        team2_id = match['league_entry_2']
        team1_points = match['league_entry_1_points']
        team2_points = match['league_entry_2_points']

        # Update cumulative points for both teams
        total_points[team1_id] += team1_points
        total_points[team2_id] += team2_points

        # Append points data to DataFrame for plotting
        data.append({'Team': team_names[team1_id], 'Gameweek': gameweek, 'Total Points': total_points[team1_id]})
        data.append({'Team': team_names[team2_id], 'Gameweek': gameweek, 'Total Points': total_points[team2_id]})

    # Step 7: Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Step 8: Plot the data using Plotly
    fig = px.line(df, x="Gameweek", y="Total Points", color="Team",
                  labels={"Total Points": "Total Points", "Gameweek": "Gameweek", "Team": "Team"},
                  title="Team Total Points Over Time (Gameweek)")

    # Customize the layout
    fig.update_layout(xaxis_title="Gameweek", yaxis_title="Total Points", hovermode="x unified")

    # Return the figure
    return(fig)

def show_home_page():
    # st.image('images/fpl_logo_2.png', width=100)  # Adjust path and size as needed
    st.title("My Fantasy Draft Team & League Standings")

    st.subheader("FPL Draft League Table")
    # Toggle to show advanced stats
    show_advanced_stats = st.checkbox("Show Advanced Stats")
    # Fetch the league standings based on the toggle
    standings_df = get_fpl_draft_league_standings(config.BRANDON_DRAFT_LEAGUE_ID,
                                                  show_advanced_stats=show_advanced_stats)
    # Display the standings dataframe
    st.dataframe(standings_df, use_container_width=True)

    # Dropdown to select gameweek
    gameweek = st.selectbox("Select Gameweek to view results:", range(1, 39))  # Gameweeks 1 to 38
    show_fixture_results(config.BRANDON_DRAFT_LEAGUE_ID, gameweek)

    # Create the table standings plot
    fig = plot_league_points_over_time(config.BRANDON_DRAFT_LEAGUE_ID)
    # Display the chart
    st.plotly_chart(fig)

    # Create the total points plot
    fig = plot_team_points_over_time(config.BRANDON_DRAFT_LEAGUE_ID)
    # Display the chart
    st.plotly_chart(fig)