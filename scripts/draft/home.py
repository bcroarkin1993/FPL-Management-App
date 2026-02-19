import config
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from scripts.common.error_helpers import get_logger, show_api_error
from scripts.common.luck_analysis import extract_draft_gw_scores, calculate_all_play_standings, render_luck_adjusted_table, render_standings_table
from scripts.common.utils import get_current_gameweek, get_draft_league_details
from scripts.common.styled_tables import render_styled_table

_logger = get_logger("fpl_app.draft.home")

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

@st.cache_data(ttl=300)
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
      If the season hasn't started (or data is incomplete), a placeholder table is returned with zero stats.
    """
    expected_columns = ['Rank', 'Team', 'Player', 'W', 'D', 'L', 'PF', 'PA', 'Pts']

    try:
        # Use cached league details
        league_response = get_draft_league_details(draft_league_id)
        if not league_response:
            st.error("Failed to fetch league data.")
            return pd.DataFrame(columns=expected_columns).set_index('Rank')

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
        show_api_error("fetching league standings", exception=e)
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
    # Use cached league details
    league_response = get_draft_league_details(draft_league_id)
    if not league_response:
        show_api_error("fetching fixture results", exception=Exception("No league data available"))
        return

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

            # Skip only if BOTH teams scored 0 (unplayed match)
            if team1_score == 0 and team2_score == 0:
                continue

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
        render_styled_table(
            fixtures_df,
            title=f"Results for Gameweek {gameweek}",
            text_align={"Team 1 Score": "center", "Team 2 Score": "center",
                         "Team 1 Result": "center", "Team 2 Result": "center"},
        )
    else:
        st.warning(f"No results available for Gameweek {gameweek}.")

def get_luck_adjusted_league_standings(draft_league_id):
    """
    Calculate luck-adjusted standings using the All-Play Record method.

    For each gameweek, every team is compared against every other team.
    This reveals true schedule luck by showing how teams would rank
    if they played everyone each week instead of a single opponent.

    Parameters:
    - draft_league_id: The FPL Draft league ID.

    Returns:
    - DataFrame with All-Play standings including Luck +/- column.
    """
    league_response = get_draft_league_details(draft_league_id)
    if not league_response:
        _logger.warning("Failed to fetch luck-adjusted standings")
        return pd.DataFrame()

    # Extract GW scores using shared module
    gw_scores = extract_draft_gw_scores(league_response)
    if gw_scores.empty:
        return pd.DataFrame()

    # Build actual standings for Luck Delta calculation
    actual_df = get_fpl_draft_league_standings(draft_league_id, show_luck_adjusted_stats=False)
    if actual_df.empty:
        return calculate_all_play_standings(gw_scores)

    # Convert actual standings to the format expected by calculate_all_play_standings
    actual_standings = actual_df.reset_index()[['Rank', 'Team', 'Pts']].rename(columns={
        'Rank': 'actual_rank',
        'Pts': 'actual_pts',
        'Team': 'team',
    })

    return calculate_all_play_standings(gw_scores, actual_standings)

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

    # Step 2: Use cached league details
    league_response = get_draft_league_details(draft_league_id)
    if not league_response:
        _logger.warning("Failed to fetch league details for points chart")
        return None

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

    # Customize the layout with dark theme
    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="Total League Points",
        hovermode="x unified",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff", size=14),
        title=dict(font=dict(size=22, color="#ffffff"), x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
        yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
    )

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

    # Step 2: Use cached league details
    league_response = get_draft_league_details(draft_league_id)
    if not league_response:
        _logger.warning("Failed to fetch league details for team points chart")
        return None

    # Step 3: Create a dictionary mapping entry_ids to team names
    league_entries = league_response.get('league_entries', [])
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
        if gameweek > current_gameweek:  # Ignore future matches
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

    # Customize the layout with dark theme
    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="Total Points",
        hovermode="x unified",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff", size=14),
        title=dict(font=dict(size=22, color="#ffffff"), x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
        yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
    )

    # Return the figure
    return(fig)

def show_home_page():
    # st.image('images/fpl_logo_2.png', width=100)  # Adjust path and size as needed
    st.title("My Fantasy Draft Team & League Standings")

    st.subheader("FPL Draft League Table")
    # Toggle to show luck adjusted standing
    show_luck_adjusted_stats = st.checkbox("Show Luck Adjusted Standings")
    if show_luck_adjusted_stats:
        st.caption("**All-Play Record**: simulates every team playing every other team each gameweek. "
                   "Luck +/- shows how much the actual schedule helped or hurt (positive = lucky).")
    # Fetch the league standings based on the toggle
    standings_df = get_fpl_draft_league_standings(config.FPL_DRAFT_LEAGUE_ID,
                                                  show_luck_adjusted_stats=show_luck_adjusted_stats)
    # Display the standings dataframe
    if show_luck_adjusted_stats:
        render_luck_adjusted_table(standings_df)
    else:
        render_standings_table(standings_df, is_h2h=True)

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