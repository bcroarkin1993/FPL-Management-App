from bs4 import BeautifulSoup
import config
from fuzzywuzzy import process
import os
import pandas as pd
import re
import requests

def normalize_apostrophes(text):
    """ Normalize different types of apostrophes to a standard single quote """
    return text.replace("’", "'").replace("`", "'")

def remove_duplicate_words(name):
    """
    Function to remove duplicate consecutive words
    """
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', name)

def check_valid_lineup(df):
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
    print("Lineup: ", df)
    print(len(df))

    # Check the total players count
    players = len(df)

    # Count occurrences of each value in the 'Position' column
    position_counts = df['position'].value_counts()
    print("Position Counts: ", position_counts)

    # Perform the checks
    player_check = players == 11
    gk_check = position_counts['G'] == 1
    def_check = position_counts['D'] >= 3 and position_counts['D'] <= 5
    mid_check = position_counts['M'] >= 3 and position_counts['M'] <= 5
    fwd_check = position_counts['F'] >= 1 and position_counts['F'] <= 3

    print("Player Check :", player_check)
    print("GK Check :", gk_check)
    print("DEF Check :", def_check)
    print("MID Check :", mid_check)
    print("FWD Check :", fwd_check)

    # Lineup is valid is all checks are true
    return (player_check & gk_check & def_check & mid_check & fwd_check)

def find_optimal_lineup(df):
    """
    Function to find a team's optimal lineup given their player_projections_df
    :param df: a dataframe of the team's player projections
    :return:
    """
    # 1. Find the top scoring GK
    top_gk = df[df['Position'] == 'G'].sort_values(by='Points', ascending=False).head(1)

    # 2. Find the top 3 scoring DEF
    top_def = df[df['Position'] == 'D'].sort_values(by='Points', ascending=False).head(3)

    # 3. Find the top 3 scoring MID
    top_mid = df[df['Position'] == 'M'].sort_values(by='Points', ascending=False).head(3)

    # 4. Find the top scoring FWD
    top_fwd = df[df['Position'] == 'F'].sort_values(by='Points', ascending=False).head(1)

    # 5. Combine the selected players
    selected_players = pd.concat([top_gk, top_def, top_mid, top_fwd])

    # 6. Find the remaining top 3 scoring players not in the selected players
    remaining_players = df[~df['Player'].isin(selected_players['Player'])]
    top_remaining = remaining_players.sort_values(by='Points', ascending=False).head(3)

    # 7. Combine all selected players
    final_selection = pd.concat([selected_players, top_remaining])

    # 8. Organize the final selection by Position and descending Projected_Points
    final_selection = final_selection.sort_values(
        by=['Position', 'Points'],
        key=lambda x: x.map({'G': 0, 'D': 1, 'M': 2, 'F': 3}),
        ascending=[True, False]
    ).reset_index(drop=True)

    # Display the final selection
    return(final_selection)

def pull_fpl_player_stats():
    # Set FPL Draft API endpoint
    draft_api = 'https://draft.premierleague.com/api/bootstrap-static'

    # Test the endpoint
    data = requests.get(draft_api)

    # extracting data in json format
    data_json = data.json()

    # Create a dataframe for the positions ('element_types')
    position_df = pd.DataFrame.from_records(data_json['element_types'])

    # Format df
    cols = ['id', 'singular_name', 'singular_name_short']
    position_df = position_df[cols]

    # Rename columns
    position_df.columns = ['position_id', 'position_name', 'position_abbrv']

    # Create a dataframe for the teams
    team_df = pd.DataFrame.from_records(data_json['teams'])

    # Format df
    cols = ['id', 'name', 'short_name']
    team_df = team_df[cols]

    # Rename columns
    team_df.columns = ['team_id', 'team_name', 'team_name_abbrv']

    # Create a DataFrame from the Player dictionary ('elements')
    player_df = pd.DataFrame.from_records(data_json['elements'])

    # Create Full Name Column
    player_df['player'] = player_df['first_name'] + ' ' + player_df['web_name']
    # Apply the remove_duplicate_words function to the 'Player' column
    player_df['player'] = player_df['player'].apply(remove_duplicate_words)

    # Merge in team_name
    player_df = pd.merge(player_df, team_df, left_on='team', right_on='team_id')

    # Merge in position name
    player_df = pd.merge(player_df, position_df, left_on='element_type', right_on='position_id')

    # Organize columns
    cols = ['id', 'player', 'position_abbrv', 'team_name', 'clean_sheets', 'goals_scored',
            'assists', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'yellow_cards',
            'starts', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'creativity', 'influence', 'bonus', 'bps', 'form', 'points_per_game', 'total_points',
            'corners_and_indirect_freekicks_order', 'corners_and_indirect_freekicks_text', 'direct_freekicks_order',
            'direct_freekicks_text', 'penalties_order', 'penalties_text', 'chance_of_playing_this_round',
            'chance_of_playing_next_round', 'status', 'added']
    player_df = player_df[cols]

    # Sort dataframe by goals_scored
    player_df = player_df.sort_values(by='total_points', ascending = False)

    # Return df
    player_df

def get_current_gameweek():
    """
    Fetches the current gameweek based on the game status from the FPL Draft API.

    Returns:
    - current_gameweek: An integer representing the current gameweek.
    """
    game_url = "https://draft.premierleague.com/api/game"
    response = requests.get(game_url)
    game_data = response.json()

    # Check if the current event is finished
    if game_data['current_event_finished']:
        current_gameweek = game_data['next_event']
    else:
        current_gameweek = game_data['current_event']

    return current_gameweek

def get_draft_picks(league_id):
    """
    Fetches the draft picks for each team in the league.

    Parameters:
    - league_id: The ID of the league.

    Returns:
    - draft_picks: A dictionary where keys are team names, and values are lists of drafted player IDs.
    """
    draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
    response = requests.get(draft_url)
    draft_data = response.json()

    draft_picks = {}

    # Parse the draft picks
    for choice in draft_data['choices']:
        team_name = choice['entry_name']
        player_id = choice['element']

        if team_name not in draft_picks:
            draft_picks[team_name] = []

        draft_picks[team_name].append(player_id)

    return draft_picks

def get_fpl_player_data():
    """
    Fetches FPL player data from the FPL Draft API and returns it as a DataFrame.

    Returns:
    - fpl_player_data: DataFrame with columns 'Player_ID', 'Player', 'Team', and 'Position'.
    """
    # Fetch data from the FPL Draft API
    player_url = "https://draft.premierleague.com/api/bootstrap-static"
    response = requests.get(player_url)
    player_data = response.json()

    # Extract relevant player information
    players = player_data['elements']

    # Create a DataFrame with necessary columns
    fpl_player_data = pd.DataFrame([{
        'Player_ID': player['id'],
        'Player': f"{player['first_name']} {player['second_name']}",
        'Team': player_data['teams'][player['team'] - 1]['short_name'],  # Map team ID to team short name
        'Position': ['G', 'D', 'M', 'F'][player['element_type'] - 1]  # Map element type to position
    } for player in players])

    return fpl_player_data

def get_league_entries(league_id):
    """
    Fetches the league entries and creates a mapping of entry IDs to team names.

    Parameters:
    - league_id: The ID of the league.

    Returns:
    - A dictionary where keys are entry IDs, and values are team names.
    """
    url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    response = requests.get(url).json()

    return {entry['entry_id']: entry['entry_name'] for entry in response['league_entries']}

def get_league_teams(league_id):
    """
    Fetches league entries from the FPL Draft API and stores them in config.LEAGUE_DATA if not already fetched.
    Returns:
    - team_dict: A dictionary mapping entry IDs to team names.
    """
    if config.LEAGUE_DATA is None:  # Fetch only if not already fetched
        league_url = f"https://draft.premierleague.com/api/league/{league_id}/details"
        league_response = requests.get(league_url)
        league_data = league_response.json()
        # Store in config
        config.LEAGUE_DATA = {entry['entry_id']: entry['entry_name'] for entry in league_data['league_entries']}

    return config.LEAGUE_DATA

def get_player_mapping():
    """
    Fetches player data from the bootstrap-static endpoint and creates a mapping of player IDs to player details.

    Returns:
    - player_map: A dictionary where keys are player IDs and values are player details (name, team, position).
    """
    url = "https://draft.premierleague.com/api/bootstrap-static"
    response = requests.get(url).json()

    # Create a mapping of player IDs to player information
    player_map = {
        player['id']: {
            'Player': f"{player['first_name']} {player['second_name']}",
            'Team': player['team'],  # Use 'team' field to map with team ID
            'Position': ['G', 'D', 'M', 'F'][player['element_type'] - 1]
        }
        for player in response['elements']
    }

    # Fetch team mapping from the same endpoint
    team_map = {team['id']: team['short_name'] for team in response['teams']}

    # Update player_map to use the full team name
    for player_data in player_map.values():
        player_data['Team'] = team_map.get(player_data['Team'], 'Unknown')

    return player_map

def get_transaction_data(league_id):
    """
    Fetches waiver transactions from the FPL Draft API and stores them in config.TRANSACTION_DATA if not already fetched.
    Returns:
    - transaction_data: A list of transactions from the API.
    """
    if config.TRANSACTION_DATA is None:  # Fetch only if not already fetched
        transaction_url = f"https://draft.premierleague.com/api/draft/league/{league_id}/transactions"
        transaction_response = requests.get(transaction_url)
        config.TRANSACTION_DATA = transaction_response.json()['transactions']  # Store transactions in config

    return config.TRANSACTION_DATA

def get_gameweek_fixtures(league_id, gameweek):
    # Draft League API URLs
    league_url = f"https://draft.premierleague.com/api/league/{league_id}/details"

    # Get league details, fixtures, and team details
    league_response = requests.get(league_url).json()

    # Extract the standings and league_entries sections of the JSON
    fixtures = league_response['matches']
    league_entries = league_response['league_entries']

    # Create a dictionary mapping entry_ids to team names and managers
    team_info = {entry['id']: (entry['entry_name'], entry['player_first_name'] + ' ' + entry['player_last_name'])
                 for entry in league_entries}

    # Create an empty list to add gameweek fixtures to
    gameweek_fixtures = []

    # Iterate over the fixtures, filter for the current_gameweek, and then format and add the fixture to list
    for fixture in fixtures:
        if fixture['event'] == gameweek:
            team1_id = fixture['league_entry_1']
            team2_id = fixture['league_entry_2']
            team1_name, team1_manager = team_info.get(team1_id, ("Unknown Team", "Unknown Manager"))
            team2_name, team2_manager = team_info.get(team2_id, ("Unknown Team", "Unknown Manager"))
            gameweek_fixtures.append(f"{team1_name} ({team1_manager}) vs {team2_name} ({team2_manager})")

    return gameweek_fixtures

def get_rotowire_rankings_url():
    """
    Fetches the full article URL for the latest Fantasy Premier League Player Rankings.

    Returns:
    - str: The full URL to the latest Rotowire player rankings article, or None if not found.
    """
    try:
        # Rotowire soccer homepage
        base_url = "https://www.rotowire.com"
        soccer_url = base_url + "/soccer/"
        response = requests.get(soccer_url)
        response.raise_for_status()  # Ensure the request was successful

        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the current gameweek from your helper function
        current_gameweek = get_current_gameweek()

        # Article title to search for
        target_title = f"Fantasy Premier League Player Rankings: Gameweek {current_gameweek}"

        # Find all compact article sections
        articles = soup.find_all('div', class_='compact-article__main')

        # Iterate through the articles to find the matching one
        for article in articles:
            link_tag = article.find('a', class_='compact-article__link')
            if link_tag and target_title in link_tag.text:
                # Construct the correct URL without double '/soccer/' issue
                article_url = base_url + link_tag['href']
                return article_url

        print(f"No article found for Gameweek {current_gameweek}.")
        return None  # Explicitly return None if no matching article is found

    except Exception as e:
        print(f"Error fetching the Rotowire rankings URL: {e}")
        return None

def get_rotowire_player_projections(url, limit=None):
    """
    Fetches fantasy rankings and projected points for players from RotoWire.

    Parameters:
    - url (str): URL to fetch the data from.
    - limit (int, optional): Number of players to display. Defaults to None (displays all players).

    Returns:
    - DataFrame: A DataFrame containing player rankings, projected points, and calculated value.
    """
    # Download the page using the requests library
    website = requests.get(url)
    soup = BeautifulSoup(website.content, 'html.parser')

    # Isolate BeautifulSoup output to the table of interest
    my_classes = soup.find(class_='article-table__tablesorter article-table__standard article-table__figure')
    players = my_classes.find_all("td")

    # Create lists for each field to collect
    overall_rank, fw_rank, mid_rank, def_rank, gk_rank = [], [], [], [], []
    player, team, matchup, position, price, tsb, points = [], [], [], [], [], [], []

    # Loop through the list of players in batches of 12
    batch_size = 12
    for i in range(0, len(players), batch_size):
        overall_rank.append(players[i].text)
        fw_rank.append(players[i + 1].text)
        mid_rank.append(players[i + 2].text)
        def_rank.append(players[i + 3].text)
        gk_rank.append(players[i + 4].text)
        player.append(players[i + 5].text)
        team.append(players[i + 6].text)
        matchup.append(players[i + 7].text)
        position.append(players[i + 8].text)
        price.append(players[i + 9].text)
        tsb.append(players[i + 10].text)
        points.append(players[i + 11].text)

    # Create a DataFrame with formatted column names
    player_rankings = pd.DataFrame(
        list(zip(overall_rank, fw_rank, mid_rank, def_rank, gk_rank, player, team,
                 matchup, position, price, tsb, points)),
        columns=[
            'Overall Rank', 'FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank',
            'Player', 'Team', 'Matchup', 'Position', 'Price', 'TSB %', 'Points'
        ]
    )

    # Replace empty strings with 0 and convert columns to numeric where appropriate
    for col in ['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank', 'Points', 'Price']:
        player_rankings[col] = pd.to_numeric(player_rankings[col], errors='coerce').fillna(0)

    # Create 'Pos Rank' by summing the four position ranks
    player_rankings['Pos Rank'] = (
        player_rankings['FW Rank'] + player_rankings['MID Rank'] +
        player_rankings['DEF Rank'] + player_rankings['GK Rank']
    ).astype(int)

    # Drop individual position rank columns
    player_rankings.drop(columns=['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank'], inplace=True)

    # Ensure 'Price' is numeric
    player_rankings['Price'] = pd.to_numeric(player_rankings['Price'], errors='coerce').fillna(0)

    # Create the 'Value' column by dividing 'Points' by 'Price'
    player_rankings['Value'] = player_rankings.apply(
        lambda row: row['Points'] / row['Price'] if row['Price'] > 0 else float('nan'), axis=1
    )

    # If a limit is provided, return only the top 'limit' players
    if limit:
        player_rankings = player_rankings.head(limit)

    # Format the DataFrame to remove the index and reset it with a starting value of 1
    player_rankings.reset_index(drop=True, inplace=True)
    player_rankings.index = player_rankings.index + 1

    return player_rankings

def get_league_player_dict_for_gameweek(league_id, gameweek):
    """
    Fetches the team compositions for all teams in the specified FPL draft league for a given gameweek.

    Parameters:
    - league_id (int): The ID of the FPL draft league.
    - gameweek (int): The gameweek for which to retrieve team compositions.

    Returns:
    - league_dict (dict): A dictionary where keys are team names, and values are dictionaries of player positions and players.
    """
    # Fetch league details to retrieve team names
    league_url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    league_response = requests.get(league_url).json()

    # Create a dictionary to map entry IDs to team names
    team_name_map = {
        entry['entry_id']: entry['entry_name'] for entry in league_response['league_entries']
    }

    # Fetch all player data from the FPL API
    fpl_players_url = "https://draft.premierleague.com/api/bootstrap-static"
    fpl_data = requests.get(fpl_players_url).json()

    # Create a DataFrame with player information
    players_df = pd.DataFrame(fpl_data['elements'])
    players_df['Player'] = players_df['first_name'] + " " + players_df['second_name']
    players_df['Position'] = players_df['element_type'].map({1: 'G', 2: 'D', 3: 'M', 4: 'F'})

    # Map player IDs to their names and positions
    player_info = players_df.set_index('id')[['Player', 'Position']].to_dict(orient='index')

    # Fetch waiver transactions up to the given gameweek
    transactions = get_waiver_transactions_up_to_gameweek(league_id, gameweek)

    # Initialize the league dictionary with team compositions
    league_dict = {
        team_name: {'G': [], 'D': [], 'M': [], 'F': []} for team_name in team_name_map.values()
    }

    # Process draft picks to populate the initial team compositions
    draft_picks = get_draft_picks(league_id)
    for team_name, player_ids in draft_picks.items():
        for player_id in player_ids:
            player_data = player_info.get(player_id)
            if player_data:
                position = player_data['Position']
                player_name = player_data['Player']
                league_dict[team_name][position].append(player_name)

    # Apply transactions to update team compositions
    for tx in transactions:
        if tx['result'] == 'a':  # Only apply approved transactions
            team_id = tx['entry']
            team_name = team_name_map.get(team_id)

            if team_name:
                player_in = player_info.get(tx['element_in'])
                player_out = player_info.get(tx['element_out'])

                if player_out:
                    position_out = player_out['Position']
                    if player_out['Player'] in league_dict[team_name][position_out]:
                        league_dict[team_name][position_out].remove(player_out['Player'])

                if player_in:
                    position_in = player_in['Position']
                    league_dict[team_name][position_in].append(player_in['Player'])

    return league_dict

def get_team_composition_for_current_gameweek(league_id, team_name):
    """
    Retrieves the current composition of a specific FPL team for the current gameweek.

    Parameters:
    - league_id (int): The ID of the FPL Draft league.
    - team_name (str): The name of the team to retrieve the composition for.

    Returns:
    - team_df (DataFrame): DataFrame with the team's players, sorted by position and points.
    """
    # URLs for API endpoints
    element_status_url = f"https://draft.premierleague.com/api/league/{league_id}/element-status"
    league_details_url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    player_data_url = "https://draft.premierleague.com/api/bootstrap-static"

    # Fetch data from the APIs
    element_status = requests.get(element_status_url).json()['element_status']
    league_details = requests.get(league_details_url).json()
    player_data = requests.get(player_data_url).json()

    # Create a mapping of owner IDs to team names
    owner_to_team = {entry['id']: entry['entry_name'] for entry in league_details['league_entries']}

    # Find the owner ID for the given team name
    team_owner_id = next((owner for owner, name in owner_to_team.items() if name == team_name), None)

    if team_owner_id is None:
        raise ValueError(f"Team '{team_name}' not found in the league.")

    # Filter players belonging to the selected team and not marked 'out'
    team_elements = [
        status['element'] for status in element_status
        if status['owner'] == team_owner_id and status['status'] != 'o'
    ]

    # Create a DataFrame of players from the bootstrap-static data
    players_df = pd.DataFrame(player_data['elements'])

    # Merge to get player details (name, team, position) for the filtered elements
    team_df = players_df[players_df['id'].isin(team_elements)].copy()

    # Create the player name by combining 'first_name' and 'second_name'
    team_df['Player'] = team_df['first_name'] + ' ' + team_df['second_name']

    # Map position codes to readable names
    position_map = {1: 'G', 2: 'D', 3: 'M', 4: 'F'}
    team_df['Position'] = team_df['element_type'].map(position_map)

    # Select relevant columns and sort by position and points
    team_df = team_df[['Player', 'team', 'Position', 'total_points']]
    team_df.columns = ['Player', 'Team', 'Position', 'Points']

    # Ensure 'Points' is numeric and fill missing values with 0
    team_df['Points'] = pd.to_numeric(team_df['Points'], errors='coerce').fillna(0.0)

    # Define position order for sorting
    position_order = ['G', 'D', 'M', 'F']
    team_df['Position'] = pd.Categorical(team_df['Position'], categories=position_order, ordered=True)

    # Sort the DataFrame
    team_df = team_df.sort_values(by=['Position', 'Points'], ascending=[True, False])

    # Reset the index for cleaner display
    team_df.reset_index(drop=True, inplace=True)

    return team_df

def get_team_composition_for_gameweek(league_id, team_name, gameweek):
    """
    Determines the composition of a given FPL team for a specified gameweek.

    Parameters:
    - league_id: The ID of the league.
    - team_name: The name of the team to fetch.
    - gameweek: The gameweek for which to determine the team's composition.

    Returns:
    - DataFrame containing player name, team, and position for the specified gameweek.
    """
    # Fetch player and team mappings
    player_map = get_player_mapping()
    team_map = get_league_entries(league_id)

    # Get the entry ID for the selected team name
    entry_id = next((k for k, v in team_map.items() if v == team_name), None)
    if entry_id is None:
        raise ValueError(f"Team '{team_name}' not found in the league.")

    # Initialize the team composition with the initial draft picks
    draft_picks = get_draft_picks(league_id)
    team_composition = set(draft_picks.get(team_name, []))

    # Apply relevant transactions to the team composition
    transactions = get_waiver_transactions_up_to_gameweek(league_id, gameweek)
    for tx in transactions:
        if tx['entry'] == entry_id and tx['result'] == 'a':
            player_in = tx['element_in']
            player_out = tx['element_out']

            if player_out in team_composition:
                team_composition.remove(player_out)
            team_composition.add(player_in)

    # Convert the team composition to a DataFrame with player details
    player_data = [
        player_map.get(pid, {'Player': f"Unknown ({pid})", 'Team': 'Unknown', 'Position': 'Unknown'})
        for pid in team_composition
    ]

    fpl_players_df = pd.DataFrame(player_data)
    return fpl_players_df

def get_team_lineup(entry_id, gameweek):
    """
    Fetches the lineup for a given team (entry_id) for a given gameweek, linking player IDs to player names.

    Parameters:
    - entry_id: The team ID.
    - gameweek: The gameweek number.

    Returns:
    - lineup: A list of players with their details (name, position, captain status, multiplier, etc.).
    """
    # Fetch the team lineup from the API
    lineup_url = f"https://draft.premierleague.com/api/entry/{entry_id}/event/{gameweek}"
    lineup_response = requests.get(lineup_url)
    lineup_data = lineup_response.json()

    # Get player data (ID to name mapping)
    player_dict = get_player_data()

    # Extract picks (player selections)
    picks = lineup_data['picks']

    # Create a list to store the lineup details
    lineup = []

    for pick in picks:
        player_id = pick['element']
        player_name = player_dict.get(player_id, "Unknown Player")
        position = pick['position']
        is_captain = pick['is_captain']
        is_vice_captain = pick['is_vice_captain']
        multiplier = pick['multiplier']

        # Append player details to the lineup list
        lineup.append({
            'Player Name': player_name,
            'Position': position,
            'Captain': 'Yes' if is_captain else 'No',
            'Vice Captain': 'Yes' if is_vice_captain else 'No',
            'Multiplier': multiplier
        })

    return lineup

def get_team_projections(player_rankings, team_name, team_dict):
    """
    Fetches projected scores for the specified team from player rankings.

    Parameters:
    - player_rankings: DataFrame containing player rankings and projected points.
    - team_name: Name of the team (from config.TEAM_LIST).
    - team_dict: Dictionary of teams and their players.

    Returns:
    - DataFrame of the team's players with their projected points.
    """
    # Get players from the selected team
    team_players = [player for position in team_dict[team_name].values() for player in position]

    # Filter player_rankings for the selected team players
    team_df = player_rankings[player_rankings['Player'].isin(team_players)]

    return team_df[['Player', 'Position', 'Points']]

def get_waiver_transactions_up_to_gameweek(league_id, gameweek):
    """
    Fetches all transactions (waivers, free agent moves, etc.) up to the selected gameweek.

    Parameters:
    - league_id: The league ID for transactions.
    - gameweek: The gameweek to fetch transactions up to.

    Returns:
    - transactions: A list of transactions that occurred up to the given gameweek.
    """
    transaction_url = f"https://draft.premierleague.com/api/draft/league/{league_id}/transactions"
    transaction_response = requests.get(transaction_url)
    transactions = transaction_response.json()['transactions']

    # Filter transactions by gameweek
    filtered_transactions = [tx for tx in transactions if tx['event'] <= gameweek]

    return filtered_transactions

def merge_fpl_players_and_projections(fpl_players_df, projections_df, fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Merges the FPL player data with Rotowire projections based on player name, team, and position.

    Parameters:
    - fpl_players_df: DataFrame with FPL players ['player', 'team', 'position'].
    - projections_df: DataFrame with Rotowire projections.
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.

    Returns:
    - merged_df: DataFrame with players, projections, and any missing players shown with NA values.
    """

    def find_best_match(fpl_player, fpl_team, fpl_position, candidates):
        """Finds the best match for a player using fuzzy matching."""
        match, score = process.extractOne(fpl_player, candidates)

        matched_row = projections_df[projections_df['Player'] == match]
        if not matched_row.empty:
            match_team = matched_row.iloc[0]['Team']
            match_position = matched_row.iloc[0]['Position']

            if match_team == fpl_team and match_position == fpl_position and score >= lower_fuzzy_threshold:
                return match

        if score >= fuzzy_threshold:
            return match

        return None

    merged_data = []

    for _, fpl_row in fpl_players_df.iterrows():
        fpl_player = fpl_row['Player']
        fpl_team = fpl_row['Team']
        fpl_position = fpl_row['Position']

        candidates = projections_df['Player'].tolist()
        best_match = find_best_match(fpl_player, fpl_team, fpl_position, candidates)

        if best_match:
            matched_row = projections_df[projections_df['Player'] == best_match].iloc[0]
            merged_data.append({
                'Player': matched_row['Player'],
                'Team': matched_row['Team'],
                'Matchup': matched_row.get('Matchup', ''),
                'Position': matched_row['Position'],
                'Price': matched_row.get('Price', float('nan')),
                'TSB%': matched_row.get('tsb', float('nan')),
                'Points': matched_row['Points'],
                'Pos Rank': matched_row.get('Pos Rank', 'NA')
            })
        else:
            merged_data.append({
                'Player': fpl_player,
                'Team': fpl_team,
                'Matchup': 'N/A',
                'Position': fpl_position,
                'Price': float('nan'),
                'TSB%': float('nan'),
                'Points': float('nan'),
                'Pos Rank': 'NA'
            })

    # Create the DataFrame
    merged_df = pd.DataFrame(merged_data)

    # Reorder columns
    merged_df = merged_df[['Player', 'Team', 'Matchup', 'Position', 'Price', 'TSB%', 'Points', 'Pos Rank']]

    # Ensure 'Pos Rank' is integer or 'NA'
    merged_df['Pos Rank'] = pd.to_numeric(merged_df['Pos Rank'], errors='coerce').fillna(-1).astype(int)
    merged_df['Pos Rank'] = merged_df['Pos Rank'].replace(-1, 'NA')

    # Set index to represent overall rank, starting at 1
    merged_df.index = pd.RangeIndex(start=1, stop=len(merged_df) + 1, step=1)

    return merged_df

def position_converter(element_type):
    """Converts element type to position name."""
    return {1: 'G', 2: 'D', 3: 'M', 4: 'F'}.get(element_type, 'Unknown')

def team_optimizer(player_rankings):
    # Loop over each team in the team_list
    for team_name, team_data in config.TEAM_LIST.items():
        # Convert the JSON to a DataFrame
        team_player_list = []
        for position, players in team_data.items():
            for player in players:
                team_player_list.append({'player_name': player, 'position': position})

        team_df = pd.DataFrame(team_player_list)

        # Perform a left join, filling missing Pts with 0
        merged_df = team_df.merge(player_rankings, left_on='player_name', right_on='Player', how='left')

        # Fill NaN values in the Pts column with 0
        merged_df['Pts'] = merged_df['Pts'].fillna(0)

        # Select relevant columns for the final output and rename them
        final_df = merged_df[['player_name', 'position', 'Pts']]
        final_df.columns = ['Player', 'Position', 'Projected_Points']

        # Ensure the 'Projected_Points' column is numeric
        final_df.loc[:, 'Projected_Points'] = pd.to_numeric(final_df['Projected_Points'], errors='coerce')

        # Return the DataFrame
        return(final_df)
