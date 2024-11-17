import config
import pandas as pd
import streamlit as st
from scripts.utils import get_current_gameweek, get_league_player_dict_for_gameweek, get_league_teams, \
    get_fpl_player_data, get_transaction_data, get_rotowire_player_projections, get_team_projections, \
    merge_fpl_players_and_projections

def compare_players(my_team_df, top_available_players_df):
    """
    Compares the projected points of my team with the top available players.

    Parameters:
    - my_team_df: DataFrame of my team players and their projected points.
    - top_available_players_df: DataFrame of top available players from waiver wire.

    Returns:
    - DataFrame of players in waiver wire who perform better than my team's players.
    """
    # Ensure both DataFrames contain a 'Team' column before merging
    if 'Team' not in top_available_players_df.columns:
        top_available_players_df['Team'] = ''
    if 'Team' not in my_team_df.columns:
        my_team_df['Team'] = ''

    # Merge the available players with my team based on position
    comparison_df = pd.merge(top_available_players_df, my_team_df, on='Position', suffixes=('_Waiver', '_Team'))

    # Ensure the 'Points' columns are numeric, forcing errors to NaN and filling with 0.0
    comparison_df['Points_Waiver'] = pd.to_numeric(comparison_df['Points_Waiver'], errors='coerce').fillna(0.0)
    comparison_df['Points_Team'] = pd.to_numeric(comparison_df['Points_Team'], errors='coerce').fillna(0.0)

    # Filter to find players in the waiver wire that have higher projected points than my team's players
    better_players_df = comparison_df[comparison_df['Points_Waiver'] > comparison_df['Points_Team']]

    return better_players_df[['Player_Waiver', 'Team_Waiver', 'Position', 'Points_Waiver', 'Player_Team', 'Points_Team']]

def find_top_waivers(fpl_player_projections, league_team_dict, limit=None):
    """
    Analyze and display waiver wire options based on player projections and the current team composition.

    Parameters:
    - fpl_player_projections: DataFrame containing Rotowire player projections.
    - league_team_dict: Dictionary containing teams and their players.
    - limit: Integer representing the number of top players to display.

    Returns:
    - available_players_df: DataFrame containing the top available players.
    """
    # Step 1: Get FPL player data (player name, team, and position)
    fpl_player_df = get_fpl_player_data()

    # Step 2: Flatten the team_dict to create a list of all taken players
    all_taken_players = []

    for team_data in league_team_dict.values():
        if isinstance(team_data, dict):  # Check if the value is a dictionary
            for position_players in team_data.values():
                all_taken_players.extend(position_players)

    # Step 3: Create a DataFrame of the taken players
    taken_players_df = pd.DataFrame({'Player': all_taken_players})

    # Step 4: Merge taken_players_df with fpl_player_data on the 'Player' column
    taken_players_df = pd.merge(taken_players_df, fpl_player_df, on='Player', how='left').set_index('Player_ID')

    # Step 5: Merge the taken players with projections to resolve name discrepancies
    taken_players_projections = merge_fpl_players_and_projections(taken_players_df, fpl_player_projections)

    # Step 6: Create a set of taken players from taken_players_projections
    taken_players_set = set(taken_players_projections['Player'])

    # Step 7: Filter out the taken players from fpl_player_projections
    available_players_df = fpl_player_projections[
        ~fpl_player_projections['Player'].isin(taken_players_set)
    ]

    # Step 8: Set index and sort by points
    available_players_df = available_players_df.set_index('Pos Rank', drop=True)
    available_players_df['Points'] = pd.to_numeric(available_players_df['Points'], errors='coerce')
    available_players_df = available_players_df.sort_values(by='Points', ascending=False)

    # Step 9: Limit the number of players displayed (if limit specified)
    if limit:
        available_players_df = available_players_df.head(limit)

    # Step 10: Return the top available players DataFrame
    return(available_players_df[['Player', 'Team', 'Position', 'Matchup', 'Points']])

def get_waiver_transactions(gameweek, show_non_approved, selected_team=None):
    """
    Fetches waiver transactions and filters by gameweek, approval status, and team (if provided).

    Parameters:
    - gameweek: The gameweek to filter transactions by.
    - show_non_approved: Boolean to toggle showing non-approved transactions.
    - selected_team: Optional team name to filter transactions by.

    Returns:
    - List of filtered transactions for the selected gameweek and team.
    """
    player_dict = get_fpl_player_data()
    team_dict = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    transaction_data = get_transaction_data(config.FPL_DRAFT_LEAGUE_ID)

    # List to hold filtered transactions
    filtered_transactions = []

    for transaction in transaction_data:
        # Check if transaction matches the selected gameweek
        if transaction['event'] == gameweek:
            # Filter by approval status
            if show_non_approved or transaction['result'] == 'a':
                added_player_id = transaction.get('element_in')
                removed_player_id = transaction.get('element_out')
                entry_id = transaction.get('entry')

                # Get player and team names
                added_player_name = player_dict.get(added_player_id, "Unknown Player")
                removed_player_name = player_dict.get(removed_player_id, "Unknown Player")
                team_name = team_dict.get(entry_id, "Unknown Team")

                # If a team is selected, filter by team name
                if selected_team and team_name != selected_team:
                    continue

                # Append transaction details to the list
                filtered_transactions.append({
                    'Team': team_name,
                    'Added Player': added_player_name,
                    'Removed Player': removed_player_name,
                    'Event': transaction['event'],
                    'Result': 'Approved' if transaction['result'] == 'a' else 'Not Approved'
                })

    return filtered_transactions

def get_waiver_transactions_by_team(selected_team, show_non_approved):
    """
    Fetches waiver transactions and filters them by team across all gameweeks.

    Parameters:
    - selected_team: Team name to filter transactions by.
    - show_non_approved: Boolean to toggle showing non-approved transactions.

    Returns:
    - List of filtered transactions for the selected team across all gameweeks.
    """
    player_dict = get_fpl_player_data()
    team_dict = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    transaction_data = get_transaction_data()

    # List to hold filtered transactions
    filtered_transactions = []

    for transaction in transaction_data:
        # Filter by approval status
        if show_non_approved or transaction['result'] == 'a':
            added_player_id = transaction.get('element_in')
            removed_player_id = transaction.get('element_out')
            entry_id = transaction.get('entry')

            # Get player and team names
            added_player_name = player_dict.get(added_player_id, "Unknown Player")
            removed_player_name = player_dict.get(removed_player_id, "Unknown Player")
            team_name = team_dict.get(entry_id, "Unknown Team")

            # Filter by the selected team
            if selected_team and team_name != selected_team:
                continue

            # Append transaction details to the list
            filtered_transactions.append({
                'Transaction ID': transaction['id'],
                'Team': team_name,
                'Added Player': added_player_name,
                'Removed Player': removed_player_name,
                'Event': transaction['event'],
                'Result': 'Approved' if transaction['result'] == 'a' else 'Not Approved'
            })

    return filtered_transactions

def show_waiver_wire_page():
    st.title("Waiver Wire Analysis")

    # Pull FPL player rankings from Rotowire
    player_rankings = get_rotowire_player_projections(config.ROTOWIRE_URL, limit=None)

    # Slider to limit the number of players shown in the waiver wire analysis
    num_players = st.slider("Select the number of available players to display:",
                            min_value=5, max_value=50, value=10, step=5)

    # Get the league player dict for the current gameweek
    league_player_dict = get_league_player_dict_for_gameweek(config.FPL_DRAFT_LEAGUE_ID, config.CURRENT_GAMEWEEK)


    # Pull the top available waivers based on the FPL player rankings and slider value
    top_available_players = find_top_waivers(player_rankings, league_player_dict, num_players)

    # Display the top available players
    st.subheader("Top Available Players:")
    st.dataframe(top_available_players, use_container_width=True)

    # Get my team's projections
    my_team_df = get_team_projections(player_rankings, config.MY_TEAM_NAME, league_player_dict)

    # Compare the waiver wire players with my team's players
    better_players_df = compare_players(my_team_df, top_available_players)

    # Show any waiver wire players that have higher projected points than my team's players
    if not better_players_df.empty:
        st.subheader("Players that could improve your team:")
        st.dataframe(better_players_df, use_container_width=True)
    else:
        st.info("No waiver players perform better than your current team.")

    # Display the league waiver transactions
    st.subheader("Waiver Wire Transactions:")

    # View mode selection (either by gameweek or by team)
    view_mode = st.radio("View Transactions By", options=["Gameweek", "Team"])

    # Checkbox to toggle non-approved transactions
    show_non_approved = st.checkbox("Show Non-Approved Transactions", value=False)

    # Fetch the current gameweek from FPL API
    current_gameweek = get_current_gameweek()

    # Fetch league teams for the team selection dropdown
    league_teams = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    team_options = ["All Teams"] + list(league_teams.values())

    if view_mode == "Gameweek":
        # View by Gameweek mode

        # Dropdown to select the gameweek
        gameweek_options = list(range(1, current_gameweek + 1))
        selected_gameweek_index = min(current_gameweek - 1, len(gameweek_options) - 1)
        selected_gameweek = st.selectbox("Select Gameweek", options=gameweek_options, index=selected_gameweek_index)

        # Get waiver transactions based on the selected gameweek and approval status
        transactions = get_waiver_transactions(selected_gameweek, show_non_approved)

        # Display transactions in a table
        if transactions:
            st.subheader(f"Transactions for Gameweek {selected_gameweek}")
            st.table(transactions)
        else:
            st.info(f"No transactions found for Gameweek {selected_gameweek}")

    elif view_mode == "Team":
        # View by Team mode

        # Dropdown to select a specific team or all teams
        selected_team = st.selectbox("Select Team", options=team_options, index=0)

        # Get waiver transactions for the selected team across all gameweeks
        team_filter = None if selected_team == "All Teams" else selected_team
        transactions = get_waiver_transactions_by_team(team_filter, show_non_approved)

        # Display transactions in a table
        if transactions:
            st.subheader(f"Transactions for {selected_team} across all Gameweeks")
            st.table(transactions)
        else:
            st.info(f"No transactions found for {selected_team}.")
