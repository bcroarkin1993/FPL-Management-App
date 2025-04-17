from bs4 import BeautifulSoup
from collections import defaultdict
import config
from fuzzywuzzy import process
import pandas as pd
import re
import requests
import unicodedata

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
    # Check the total players count
    players = len(df)

    # Count occurrences of each value in the 'Position' column
    position_counts = df['position'].value_counts()

    # Perform the checks
    player_check = players == 11
    gk_check = position_counts['G'] == 1
    def_check = position_counts['D'] >= 3 and position_counts['D'] <= 5
    mid_check = position_counts['M'] >= 3 and position_counts['M'] <= 5
    fwd_check = position_counts['F'] >= 1 and position_counts['F'] <= 3

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

def format_team_name(name):
    """
    Formats a team name by normalizing apostrophes and capitalizing each word.

    Parameters:
    - name (str): The team name to format.

    Returns:
    - str: The formatted team name.
    """
    if name is None:
        return None
    # Normalize Unicode and replace curly apostrophes with straight apostrophes
    normalized_name = unicodedata.normalize('NFKC', name).replace('’', "'").strip()
    # Capitalize the first letter of each word
    return ' '.join(word.capitalize() for word in normalized_name.split())


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
    player_df['player'] = player_df['first_name'] + ' ' + player_df['second_name']
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

    # Ensure expected_goal_involvements is numeric
    player_df["expected_goal_involvements"] = pd.to_numeric(
        player_df["expected_goal_involvements"], errors="coerce"
    )

    # Create actual goal involvements column and ensure that it is numeric (sum of goals + assists)
    player_df["actual_goal_involvements"] = pd.to_numeric(
        player_df["goals_scored"], errors="coerce"
    ) + pd.to_numeric(player_df["assists"], errors="coerce")

    # Sort dataframe by goals_scored
    player_df = player_df.sort_values(by='total_points', ascending=False)

    # Return df
    return (player_df)

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
    Fetches the draft picks for each team in the league and returns a dictionary with team_id as the key,
    including player names instead of player IDs.

    Parameters:
    - league_id: The ID of the FPL Draft league.

    Returns:
    - draft_picks: A dictionary where keys are team IDs, and values are dictionaries with team name and player names.
    """
    # Endpoints for draft picks and player data
    draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
    league_details_url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    player_data_url = "https://draft.premierleague.com/api/bootstrap-static"

    # Fetch data
    draft_data = requests.get(draft_url).json()
    league_details = requests.get(league_details_url).json()
    player_data = requests.get(player_data_url).json()

    # Create a mapping of player ID to player name
    player_mapping = {
        player['id']: f"{player['first_name']} {player['second_name']}"
        for player in player_data['elements']
    }

    # Initialize the draft picks dictionary
    draft_picks = {}

    # Populate draft picks
    for choice in draft_data['choices']:
        team_id = choice['entry']         # Team ID
        team_name = choice['entry_name']  # Team name
        player_id = choice['element']     # Player ID

        # Ensure the team_id key exists in the dictionary
        if team_id not in draft_picks:
            draft_picks[team_id] = {
                'team_name': team_name,
                'players': []  # Initialize player list
            }

        # Add the player name to the list
        player_name = player_mapping.get(player_id, f"Unknown ({player_id})")
        draft_picks[team_id]['players'].append(player_name)

    return draft_picks

def get_fpl_player_mapping():
    """
    Fetches FPL player data from the FPL Draft API and returns it as a dictionary to link player ids to player names.

    Returns:
    - fpl_player_data: DataFrame with columns 'Player_ID', 'Player', 'Team', and 'Position'.
    """
    # Fetch data from the FPL Draft API
    player_url = "https://draft.premierleague.com/api/bootstrap-static"
    response = requests.get(player_url)

    # Extract relevant player information
    player_data = response.json()
    players = player_data['elements']
    teams = player_data.get('teams', [])

    # Create a mapping of player IDs to player information
    fpl_player_map = {}

    for player in players:
        player_id = player.get('id')
        first_name = player.get('first_name', '')
        second_name = player.get('second_name', '')
        full_name = f"{first_name} {second_name}"

        web_name = player.get('web_name', '').strip()
        if not web_name or web_name == full_name:
            web_name = None  # treat as missing if it's the same as full name or blank

        team_index = player.get('team', 0) - 1  # team index is 1-based
        position_index = player.get('element_type', 1) - 1

        team_short_name = teams[team_index]['short_name'] if 0 <= team_index < len(teams) else 'Unknown'
        position = ['G', 'D', 'M', 'F'][position_index] if 0 <= position_index < 4 else 'Unknown'

        fpl_player_map[player_id] = {
            'Player_Name': full_name,
            'Web_Name': web_name,
            'Team': team_short_name,
            'Position': position
        }

    return fpl_player_map

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

def get_league_player_ownership(league_id):
    """
    Fetches the player ownership in the league for the current gameweek and groups the players by team and position.

    Parameters:
    - league_id: The ID of the FPL Draft league.

    Returns:
    - A dictionary structured as:
      {'Team Name': {'G': [list of goalkeepers], 'D': [list of defenders], 'M': [list of midfielders], 'F': [list of forwards]}}
    """
    # Endpoint URLs
    element_status_url = f"https://draft.premierleague.com/api/league/{league_id}/element-status"
    league_details_url = f"https://draft.premierleague.com/api/league/{league_id}/details"

    # Fetch the element status and league details
    element_status = requests.get(element_status_url).json()['element_status']
    league_details = requests.get(league_details_url).json()

    # Fetch player and owner mappings
    player_map = get_fpl_player_mapping()  # {player_id: {'Player_Name': 'Name', 'Team': 'XYZ', 'Position': 'M'}, ...}
    owner_map = {entry['id']: entry['entry_name'] for entry in league_details['league_entries']}  # {owner_id: 'Team Name'}

    # Initialize a dictionary to group players by team and position
    league_ownership = defaultdict(lambda: {'G': [], 'D': [], 'M': [], 'F': []})

    for status in element_status:
        player_id = status['element']
        owner_id = status['owner']
        if owner_id is None:  # Skip players without an owner
            continue

        # Convert player ID to name and position
        player_info = player_map.get(player_id, {'Player': f"Unknown ({player_id})", 'Position': 'Unknown'})
        player_name = player_info['Player_Name']
        player_position = player_info['Position']

        # Convert owner ID to team name
        team_name = owner_map.get(owner_id, f"Unknown Team ({owner_id})")

        # Add the player to the appropriate team and position group
        if player_position in ['G', 'D', 'M', 'F']:  # Valid positions
            league_ownership[team_name][player_position].append(player_name)

    return dict(league_ownership)

def get_starting_team_composition(league_id):
    """
    Fetches the draft picks for each team in the league and returns a dictionary with team_id as the primary key
    and the team_name field plus a player field with all the player names.

    Parameters:
    - league_id: The ID of the FPL Draft league.

    Returns:
    - draft_picks: A dictionary where keys are team IDs, and values are dictionaries with team name and player names.
    """
    # Endpoints for draft picks and player data
    draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
    league_details_url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    player_data_url = "https://draft.premierleague.com/api/bootstrap-static"

    # Fetch data
    draft_data = requests.get(draft_url).json()
    league_details = requests.get(league_details_url).json()
    player_data = requests.get(player_data_url).json()

    # Create a mapping of entry ID to team name
    team_names = {entry['id']: entry['entry_name'] for entry in league_details['league_entries']}

    # Create a mapping of player ID to player name
    player_mapping = {
        player['id']: f"{player['first_name']} {player['second_name']}"
        for player in player_data['elements']
    }

    # Initialize the draft picks dictionary
    draft_picks = {}

    # Populate draft picks
    for choice in draft_data['choices']:
        team_id = choice['entry']         # Team ID
        team_name = choice['entry_name']  # Team name
        player_id = choice['element']     # Player ID

        # Ensure the team_id key exists in the dictionary
        if team_id not in draft_picks:
            draft_picks[team_id] = {
                'team_name': team_name,
                'players': []  # Initialize player list
            }

        # Add the player name to the list
        player_name = player_mapping.get(player_id, f"Unknown ({player_id})")
        draft_picks[team_id]['players'].append(player_name)

    return draft_picks

def get_team_composition_for_gameweek(league_id, team_id, gameweek):
    """
    Determines the composition of a given FPL team for a specified gameweek.

    Parameters:
    - league_id (int): The ID of the league.
    - team_id (int): The team ID of the team to fetch.
    - gameweek: The gameweek for which to determine the team's composition.

    Returns:
    - DataFrame containing player name, team, and position for the specified gameweek.
    """
    # Fetch player and team mappings
    player_map = get_fpl_player_mapping()

    # Initialize the team composition with the initial draft picks
    draft_picks = get_starting_team_composition(league_id)
    team_composition = set(draft_picks.get(team_id, {}).get('players', []))

    # Apply relevant transactions to the team composition
    transactions = get_waiver_transactions_up_to_gameweek(league_id, gameweek)
    for tx in transactions:
        if tx['entry'] == team_id and tx['result'] == 'a':
            # Convert player IDs to names using player_map
            player_in = player_map.get(tx['element_in'], {}).get('Player', f"Unknown ({tx['element_in']})")
            player_out = player_map.get(tx['element_out'], {}).get('Player', f"Unknown ({tx['element_out']})")

            # Update the team composition
            team_composition.remove(player_out)
            team_composition.add(player_in)

    # Convert the team composition to a DataFrame with player details
    player_data = [
        player_map.get(player_id, {'Player_Name': player_name, 'Team': 'Unknown', 'Position': 'Unknown'})
        for player_name, player_id in [(player, next((k for k, v in player_map.items() if v['Player'] == player), None))
                                       for player in team_composition]
    ]

    fpl_players_df = pd.DataFrame(player_data)
    return(fpl_players_df)

def get_team_id_by_name(league_id, team_name):
    """
    Converts a team name to its corresponding team ID in the specified FPL league.

    Parameters:
    - league_id (int): The ID of the FPL league.
    - team_name (str): The name of the team to search for.

    Returns:
    - team_id (int): The ID of the team with the given name.

    Raises:
    - ValueError: If the team name is not found in the league.
    """
    # Fetch league entries to map team names to IDs
    team_map = dict(get_league_entries(league_id))  # Ensure team_map is a dictionary

    # Normalize the input team name
    normalized_team_name = normalize_apostrophes(team_name)

    # Search for the team ID by normalized team name
    team_id = next((id for id, name in team_map.items() if normalize_apostrophes(name) == normalized_team_name), None)

    if team_id is None:
        raise ValueError(f"Team '{team_name}' not found in the league.")

    return team_id

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
    player_dict = get_fpl_player_mapping()

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

def get_team_projections(player_rankings, league_id, team_id):
    """
    Fetches projected scores for the specified team from player rankings.

    Parameters:
    - player_rankings (df): DataFrame containing player rankings and projected points.
    - league_id (int): The ID of the FPL Draft league.
    - team_id (int): ID of the team (unique identifier for the FPL team).

    Returns:
    - DataFrame of the team's players with their projected points.
    """
    # Step 1: Fetch team composition using the team ID
    team_players_df = get_team_composition_for_gameweek(int(league_id), int(team_id), config.CURRENT_GAMEWEEK)

    # Step 2: Merge team players with player rankings using fuzzy matching
    team_projections_df = merge_fpl_players_and_projections(team_players_df, player_rankings)

    # Step 3: Return the relevant columns with proper formatting
    return team_projections_df[['Player', 'Position', 'Points']].sort_values(by='Points', ascending=False)

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

def clean_fpl_player_names(fpl_players_df, projections_df, fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Cleans the player names in the FPL DataFrame by replacing them with their best matches from the projections DataFrame.

    Parameters:
    - fpl_players_df: DataFrame with FPL players ['Player', 'Team', 'Position'].
    - projections_df: DataFrame with Rotowire projections ['Player', 'Team', 'Position'].
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.

    Returns:
    - fpl_players_df: Updated FPL DataFrame with cleaned player names.
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

        return fpl_player  # Return original name if no good match is found

    # Extract candidate names from projections
    projection_names = projections_df['Player'].tolist()

    # Update FPL DataFrame with cleaned player names
    fpl_players_df['Player'] = fpl_players_df.apply(
        lambda row: find_best_match(row['Player_Name'], row['Team'], row['Position'], projection_names), axis=1
    )

    return fpl_players_df

def find_best_match(fpl_player, fpl_team, fpl_position, candidates, projections_df, fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Finds the best match for a player using fuzzy matching.

    Parameters:
    - projections_df: DataFrame with Rotowire projections.
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.
    """
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

def merge_fpl_players_and_projections(fpl_players_df, projections_df):
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

    merged_data = []

    for _, fpl_row in fpl_players_df.iterrows():
        fpl_player = fpl_row['Player']
        fpl_team = fpl_row['Team']
        fpl_position = fpl_row['Position']

        candidates = projections_df['Player'].tolist()
        best_match = find_best_match(fpl_player, fpl_team, fpl_position, candidates, projections_df)

        if best_match:
            matched_row = projections_df[projections_df['Player'] == best_match].iloc[0]
            merged_data.append({
                'Player': matched_row['Player'],
                'Team': matched_row['Team'],
                'Matchup': matched_row.get('Matchup', ''),
                'Position': matched_row['Position'],
                'Points': matched_row['Points'],
                'Pos Rank': matched_row.get('Pos Rank', 'NA')
            })
        else:
            merged_data.append({
                'Player': fpl_player,
                'Team': fpl_team,
                'Matchup': 'N/A',
                'Position': fpl_position,
                'Points': float('nan'),
                'Pos Rank': 'NA'
            })

    # Create the DataFrame
    merged_df = pd.DataFrame(merged_data)

    # Reorder columns
    merged_df = merged_df[['Player', 'Team', 'Matchup', 'Position', 'Points', 'Pos Rank']]

    # Ensure 'Pos Rank' is integer or 'NA'
    merged_df['Pos Rank'] = pd.to_numeric(merged_df['Pos Rank'], errors='coerce').fillna(-1).astype(int)
    merged_df['Pos Rank'] = merged_df['Pos Rank'].replace(-1, 'NA')

    # Set index to represent overall rank, starting at 1
    merged_df.index = pd.RangeIndex(start=1, stop=len(merged_df) + 1, step=1)

    return merged_df

def normalize_apostrophes(text):
    """
    Normalizes text by converting different apostrophe types to a standard straight apostrophe.

    Parameters:
    - text (str): The text to normalize.

    Returns:
    - str: The normalized text.
    """
    if text is None:
        return None
    # Normalize Unicode and replace curly apostrophes with straight apostrophes
    return unicodedata.normalize('NFKC', text).replace('’', "'").strip().lower()

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
