import config
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from scripts.utils import get_current_gameweek

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

def get_fpl_draft_league_standings(draft_league_id, show_luck_adjusted_stats=False):
    """
    Fetches and displays the current league standings for an FPL Draft league,
    with an option to show advanced statistics like luck-adjusted standings.

    Parameters:
    - draft_league_id (int or str): The ID of the FPL Draft league.
    - show_luck_adjusted_stats (bool): If True, returns advanced statistics such as:
        - Avg_Week_Rank, Avg_Score, Fair_Rank, Luck_Index

    Returns:
    - pandas.DataFrame: League standings with columns:
        Rank, Team, Player, W, D, L, PF, PA, Pts
      If the season hasn’t started (or data is incomplete), a placeholder table is returned with zero stats.
    """
    expected_columns = ['Rank', 'Team', 'Player', 'W', 'D', 'L', 'PF', 'PA', 'Pts']

    try:
        # API call
        league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"
        league_response = requests.get(league_url).json()

        standings = league_response.get('standings', [])
        league_entries = league_response.get('league_entries', [])
        entries_df = pd.DataFrame(league_entries)

        # --- Normalize entry ID column: support both 'id' and 'entry_id'
        id_col = None
        if 'id' in entries_df.columns:
            id_col = 'id'
        elif 'entry_id' in entries_df.columns:
            id_col = 'entry_id'

        # Build Team/Player columns (best-effort even if some fields are missing)
        if not entries_df.empty:
            first = entries_df['player_first_name'] if 'player_first_name' in entries_df else ''
            last  = entries_df['player_last_name']  if 'player_last_name'  in entries_df else ''
            entries_df['Player'] = (first.astype(str) + ' ' + last.astype(str)).str.strip()
            if 'entry_name' in entries_df:
                entries_df['Team'] = entries_df['entry_name']
            else:
                entries_df['Team'] = ''
            # Keep only what's needed for placeholder/merge
            base_entries_df = entries_df[['Team', 'Player']].copy()
        else:
            base_entries_df = pd.DataFrame(columns=['Team', 'Player'])

        # If standings are missing or incomplete, return placeholder table
        standings_df = pd.DataFrame(standings)
        required_stand_cols = {'league_entry', 'matches_won', 'matches_drawn', 'matches_lost', 'points_for', 'points_against'}
        if standings_df.empty or not required_stand_cols.issubset(standings_df.columns):
            placeholder_df = base_entries_df.copy()
            if placeholder_df.empty:
                # If even entries are missing, return empty shape with correct columns
                placeholder_df = pd.DataFrame(columns=['Team', 'Player'])
            placeholder_df['W'] = 0
            placeholder_df['D'] = 0
            placeholder_df['L'] = 0
            placeholder_df['PF'] = 0
            placeholder_df['PA'] = 0
            placeholder_df['Pts'] = 0
            placeholder_df['Rank'] = range(1, len(placeholder_df) + 1)
            placeholder_df = placeholder_df[expected_columns].set_index('Rank')
            st.info("⚠️ Season has not started yet (or standings incomplete). Displaying placeholder league table.")
            return placeholder_df

        # We have standings; ensure we have an ID column to merge on. If not, fallback to placeholder rather than error.
        if id_col is None:
            st.info("⚠️ League entries missing an ID field; showing placeholder table.")
            placeholder_df = base_entries_df.copy()
            placeholder_df['W'] = 0
            placeholder_df['D'] = 0
            placeholder_df['L'] = 0
            placeholder_df['PF'] = 0
            placeholder_df['PA'] = 0
            placeholder_df['Pts'] = 0
            placeholder_df['Rank'] = range(1, len(placeholder_df) + 1)
            placeholder_df = placeholder_df[expected_columns].set_index('Rank')
            return placeholder_df

        # Merge real standings with entries
        merged = standings_df.merge(entries_df, left_on='league_entry', right_on=id_col, how='left')

        # Validate merged fields exist; otherwise fallback
        req_merge_cols = {'entry_name', 'player_first_name', 'player_last_name',
                          'matches_won', 'matches_drawn', 'matches_lost', 'points_for', 'points_against'}
        if not req_merge_cols.issubset(merged.columns):
            st.info("⚠️ Merged standings missing fields; showing placeholder table.")
            placeholder_df = base_entries_df.copy()
            placeholder_df['W'] = 0
            placeholder_df['D'] = 0
            placeholder_df['L'] = 0
            placeholder_df['PF'] = 0
            placeholder_df['PA'] = 0
            placeholder_df['Pts'] = 0
            placeholder_df['Rank'] = range(1, len(placeholder_df) + 1)
            placeholder_df = placeholder_df[expected_columns].set_index('Rank')
            return placeholder_df

        # Build final table
        out = merged[['entry_name', 'player_first_name', 'player_last_name',
                      'matches_won', 'matches_drawn', 'matches_lost', 'points_for', 'points_against']].copy()
        out.columns = ['Team', 'First', 'Last', 'W', 'D', 'L', 'PF', 'PA']
        out['Player'] = (out['First'].astype(str) + ' ' + out['Last'].astype(str)).str.strip()
        out.drop(columns=['First', 'Last'], inplace=True)
        out['Pts'] = out['W'] * 3 + out['D']
        out['Rank'] = out['Pts'].rank(ascending=False, method='min').astype(int)
        out = out.sort_values('Rank')
        out = out[['Rank', 'Team', 'Player', 'W', 'D', 'L', 'PF', 'PA', 'Pts']].set_index('Rank')

        # Optional: luck-adjusted view
        if show_luck_adjusted_stats:
            return get_luck_adjusted_league_standings(draft_league_id)

        return out

    except Exception as e:
        st.error(f"Error fetching league standings: {e}")
        return pd.DataFrame(columns=expected_columns).set_index('Rank')

def show_fixture_results(draft_league_id, gameweek):
    """
    Display fixture results for a given gameweek in an FPL Draft league.
    Gracefully handles the case when the season has not started yet
    (i.e., no 'matches' key or empty fixtures list from the API).

    Parameters:
    - draft_league_id (int or str): The ID of the FPL Draft league.
    - gameweek (int): The gameweek number for which to display results.

    Behavior:
    - If fixture data is available, displays a DataFrame showing:
        - Team names
        - Scores
        - Win/Loss/Draw results
    - If no fixture data is available yet (season not started), displays an info message in Streamlit.
    """
    # Draft League API URL
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"

    # Fetch the league details
    league_response = requests.get(league_url).json()

    # Check if the season has started (i.e., 'matches' key exists)
    if 'matches' not in league_response or not league_response['matches']:
        st.info("⚠️ No fixture data available yet. The season might not have started.")
        return

    fixtures = league_response['matches']
    league_entries = league_response['league_entries']

    # Map entry IDs to team names
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Collect fixtures for the specified gameweek
    fixtures_list = []

    for fixture in fixtures:
        if fixture['event'] == gameweek:
            team1_id = fixture['league_entry_1']
            team2_id = fixture['league_entry_2']
            team1_name = team_names.get(team1_id)
            team2_name = team_names.get(team2_id)
            team1_score = fixture['league_entry_1_points']
            team2_score = fixture['league_entry_2_points']

            if (team1_score != 0) and (team2_score != 0):
                # Determine match result
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

    # Show the results or fallback message
    if fixtures_list:
        fixtures_df = pd.DataFrame(fixtures_list)
        st.write(f"Results for Gameweek {gameweek}:")
        st.dataframe(fixtures_df, use_container_width=True)
    else:
        st.warning(f"No results available for Gameweek {gameweek}.")

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
        gameweek = fixture['event']
        team1_id = fixture['league_entry_1']
        team2_id = fixture['league_entry_2']
        team1_name = team_names.get(team1_id)
        team2_name = team_names.get(team2_id)
        team1_score = fixture['league_entry_1_points']
        team2_score = fixture['league_entry_2_points']

        # Filter out games that have not been played yet
        if (team1_score != 0) & (team2_score != 0):
            fixtures_list.append({
                "Gameweek": gameweek,
                "Team 1": team1_name,
                "Team 1 Score": team1_score,
                "Team 2": team2_name,
                "Team 2 Score": team2_score,
            })

    # Convert fixtures_list to dataframe
    fixtures_df = pd.DataFrame(fixtures_list)

    # Create a long-format dataframe with each team's score and result for the week
    long_format_df = pd.concat([
        fixtures_df[['Team 1', 'Team 1 Score', 'Gameweek']].rename(columns={'Team 1': 'Team', 'Team 1 Score': 'Score'}),
        fixtures_df[['Team 2', 'Team 2 Score', 'Gameweek']].rename(columns={'Team 2': 'Team', 'Team 2 Score': 'Score'})
    ])

    # Sort by Gameweek for clarity
    long_format_df = long_format_df.sort_values(by=["Gameweek"]).reset_index(drop=True)

    # Calculate the average score per gameweek
    avg_scores = long_format_df.groupby("Gameweek")["Score"].transform("mean")

    # Assign result based on comparison to gameweek average
    long_format_df["Result"] = long_format_df["Score"].apply(
        lambda x: 'W' if x > avg_scores[long_format_df["Score"] == x].iloc[0]
        else 'D' if x == avg_scores[long_format_df["Score"] == x].iloc[0]
        else 'L')

    # Calculate W/L/T counts per team
    result_counts = pd.crosstab(long_format_df['Team'], long_format_df['Result']).reset_index()

    # Ensure all result columns are present
    for col in ['W', 'D', 'L']:
        if col not in result_counts.columns:
            result_counts[col] = 0

    # Calculate the rank for each team in each week based on their score
    long_format_df['GW_Rank'] = long_format_df.groupby("Gameweek")["Score"].rank(ascending=False, method='min').astype(
        int)

    # Calculate the actual results for each team based on their results (W, L, D)
    long_format_df['Points'] = long_format_df['Result'].apply(lambda x: 3 if x == 'W' else (1 if x == 'D' else 0))

    # Calculate the average weekly rank (Fair_Rank) and total points (actual standings) for each team
    luck_adjusted_df = long_format_df.groupby('Team').agg(
        Avg_GW_Rank=('GW_Rank', 'mean'),  # The "fair" rank based on average weekly score rank
        Total_Points=('Points', 'sum'),  # The actual points based on match results
        Avg_Score=('Score', 'mean')  # The average score across all weeks
    ).reset_index()

    # Merge luck_adjusted_df with result_counts
    luck_adjusted_df = luck_adjusted_df.merge(result_counts[['Team', 'W', 'D', 'L']], on='Team', how='left')

    # Create the adjusted standings by ranking teams based on Avg_Week_Rank
    luck_adjusted_df['Fair_Rank'] = luck_adjusted_df['Avg_GW_Rank'].rank(ascending=True, method='min').astype(int)

    # Sort by Fair_Rank to get the adjusted standings
    luck_adjusted_df = luck_adjusted_df.sort_values(by='Fair_Rank')

    # Format number values
    luck_adjusted_df['Avg_GW_Rank'] = round(luck_adjusted_df['Avg_GW_Rank'], 2)
    luck_adjusted_df['Avg_GW_Score'] = round(luck_adjusted_df['Avg_Score'], 2)
    # Format final results
    luck_adjusted_df = luck_adjusted_df[
        ['Fair_Rank', 'Team', 'W', 'D', 'L', 'Avg_GW_Score', 'Avg_GW_Rank', 'Total_Points']]
    # Set the index
    luck_adjusted_df.set_index('Fair_Rank', inplace=True)

    return luck_adjusted_df

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

    # Prevent failure if the season hasn't started
    if current_gameweek is None:
        st.info("⚠️ The season hasn't started yet. Points over time will be shown once games begin.")
        return None  # Or return an empty figure

    # Step 2: Fetch league details
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"
    league_response = requests.get(league_url).json()

    # Step 3: Create a dictionary mapping entry_ids to team names
    league_entries = league_response['league_entries']
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Step 4: Initialize league points for each team and data for plotting
    league_points = {team_id: 0 for team_id in team_names.keys()}  # Holds cumulative league points
    data = []  # To store data for plotting

    # Check if the season has started (i.e., 'matches' key exists)
    if 'matches' not in league_response or not league_response['matches']:
        st.info("⚠️ No fixture data available yet. The season might not have started.")
        return

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

    # Prevent failure if the season hasn't started
    if current_gameweek is None:
        return None  # Or return an empty figure

    # Step 2: Fetch league details
    league_url = f"https://draft.premierleague.com/api/league/{draft_league_id}/details"
    league_response = requests.get(league_url).json()

    # Step 3: Create a dictionary mapping entry_ids to team names
    league_entries = league_response['league_entries']
    team_names = {entry['id']: entry['entry_name'] for entry in league_entries}

    # Check if the season has started (i.e., 'matches' key exists)
    if 'matches' not in league_response or not league_response['matches']:
        st.info("⚠️ No fixture data available yet. The season might not have started.")
        return

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
    # Toggle to show luck adjusted standing
    show_luck_adjusted_stats = st.checkbox("Show Luck Adjusted Standings")
    # Fetch the league standings based on the toggle
    standings_df = get_fpl_draft_league_standings(config.FPL_DRAFT_LEAGUE_ID,
                                                  show_luck_adjusted_stats=show_luck_adjusted_stats)
    # Display the standings dataframe
    st.dataframe(standings_df, use_container_width=True)

    # Dropdown to select gameweek
    gameweek = st.selectbox("Select Gameweek to view results:", range(1, 39))  # Gameweeks 1 to 38
    show_fixture_results(config.FPL_DRAFT_LEAGUE_ID, gameweek)

    # Create the table standings plot
    fig = plot_league_points_over_time(config.FPL_DRAFT_LEAGUE_ID)
    # Display the chart (if season has started)
    if fig:
        st.plotly_chart(fig)

    # Create the total points plot
    fig = plot_team_points_over_time(config.FPL_DRAFT_LEAGUE_ID)
    # Display the chart (if season has started)
    if fig:
        st.plotly_chart(fig)