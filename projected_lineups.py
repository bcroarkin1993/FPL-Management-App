from bs4 import BeautifulSoup
from collections import defaultdict
import config
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

def extract_players(section, team_type, team_name):
    """
    Extracts valid players from the given section, excluding those listed in the Injuries section.

    Parameters:
    - section (BeautifulSoup): The section containing the team’s lineup.
    - team_type (str): 'home' or 'visit' to determine the team type.
    - team_name (str): The name of the team.

    Returns:
    - list: A list of tuples containing (Team, Position, Player) for valid players.
    """
    player_list = []
    injuries_section_reached = False  # Track if the Injuries section has been reached

    # Find the correct lineup section for the team
    players_section = section.find('ul', class_=f'lineup__list is-{team_type}')

    if players_section:
        for item in players_section.find_all('li', class_='lineup__player'):
            # Check if we reached the Injuries section
            if item.find_previous_sibling('li', class_='lineup__title is-middle'):
                injuries_section_reached = True

            # Skip players if we are in the Injuries section
            if not injuries_section_reached:
                try:
                    position = item.find('div', class_='lineup__pos').text.strip()
                    player_name = item.find('a').text.strip()
                    player_list.append((team_name, position, player_name))
                except AttributeError:
                    continue  # Skip if position or player name is missing

    return player_list

def scrape_rotowire_lineups(url):
    """
    Scrapes the Rotowire Soccer Lineups page to extract the projected lineups for all matchups,
    excluding players listed in the Injuries section.

    Parameters:
    - url (str): The URL of the Rotowire lineups page.

    Returns:
    - DataFrame containing the team names, player names, and positions.
    """
    # Send a request to the Rotowire lineups page
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Initialize an empty list to store match data
    matchups = []

    # Find all lineup sections (home and away matchups)
    lineup_sections = soup.find_all('div', class_='lineup__main')

    # Iterate through each section to extract team and player data
    for section in lineup_sections:
        try:
            # Extract home and away team names
            home_team = section.find_previous('div', class_='lineup__mteam is-home').text.strip()
            away_team = section.find_previous('div', class_='lineup__mteam is-visit').text.strip()

            # Extract players while excluding those listed in the Injuries section
            home_players = extract_players(section, 'home', home_team)
            away_players = extract_players(section, 'visit', away_team)

            # Add players to the matchups list
            matchups.extend(home_players + away_players)

        except AttributeError as e:
            print(f"Error parsing section: {e}")

    # Convert the data to a pandas DataFrame
    lineups_df = pd.DataFrame(matchups, columns=['Team', 'Position', 'Player'])

    return lineups_df

def scrape_matchups(url):
    """Scrapes the matchups from the Rotowire page."""
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    matchups_section = soup.find_all('div', class_='lineup__matchup')

    matchups = []
    for matchup in matchups_section:
        try:
            home_team = matchup.find('div', class_='lineup__mteam is-home').text.strip()
            away_team = matchup.find('div', class_='lineup__mteam is-visit').text.strip()
            matchups.append((home_team, away_team))
        except:
            continue

    return matchups

def plot_soccer_field(player_df, team_name):
    """
    Plots players on a soccer field based on their positions for a specific team.

    Parameters:
    - player_df (pd.DataFrame): A DataFrame containing 'Position' and 'Player' columns.
    - team_name (str): The name of the team, displayed as the title above the field.
    """
    # Define positions with (x, y) coordinates
    position_mapping = {
        'GK': (5, 0),  # Goalkeeper
        'DL': (1.5, 1.5), 'DC': (5, 1), 'DR': (8.5, 1.5),  # Defenders
        'DML': (2, 2), 'DMC': (5, 2), 'DMR': (8, 2),  # Defensive Midfielders
        'ML': (1.5, 2.75), 'MC': (5, 2.5), 'MR': (8.5, 2.75),  # Midfielders
        'AML': (1.4, 3.25), 'AMC': (5, 3.25), 'AMR': (8.6, 3.25),  # Attacking Midfielders
        'FL': (2, 4.25), 'FWL': (2, 4.25), 'FC': (5, 4.5), 'FW': (5, 4.25), 'FWR': (8, 4.25), 'FR': (8, 4.25)  # Forwards
    }

    # Get the team's primary and secondary colors or use default values (black and white)
    colors = config.TEAM_COLORS.get(team_name, {"primary": "#000000", "secondary": "#FFFFFF"})
    primary_color = colors["primary"]
    secondary_color = colors["secondary"]

    # Group players by position
    grouped_players = defaultdict(list)
    for _, row in player_df.iterrows():
        grouped_players[row['Position']].append(row['Player'])

    # Create a Plotly figure
    fig = go.Figure()

    # Draw field boundaries
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=10, y1=6.5, line=dict(color="green", width=2))

    # Add a horizontal white line at the top (y = 5)
    fig.add_shape(type="line", x0=0, y0=5, x1=10, y1=5, line=dict(color="white", dash="dash"))

    # Add players to the field
    for position, players in grouped_players.items():
        for i, player_name in enumerate(players):
            x, y = position_mapping.get(position, (5, 3))  # Default to MC if not found

            # Adjust for two players in the same central position
            if len(players) == 2 and position in ['DC', 'MC', 'FC', 'FW', 'DMC', 'AMC']:
                x += -1.25 if i == 0 else 1.25  # Shift left and right

            # Adjust for three players in the same central position
            elif len(players) == 3 and position in ['DC', 'MC', 'FC', 'DMC', 'AMC']:
                if i == 0:
                    x -= 1.75  # Shift left
                elif i == 2:
                    x += 1.75  # Shift right

            # Add player marker with team-specific color and hover info
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=15, color=primary_color),
                text=player_name,
                textposition="top center",
                textfont=dict(color=secondary_color, size=16),
                hovertemplate=f"Name: {player_name}<br>Position: {position}<br>Coordinates: ({x}, {y})<extra></extra>"
            ))

    fig.update_layout(
        width=500, height=600,
        xaxis=dict(range=[0, 10], visible=False),
        yaxis=dict(range=[-0.2, 5.2], visible=False),  # Adjust y-axis range to zoom in
        plot_bgcolor="green",
        showlegend = False # Remove the legend
    )

    # Return the plot
    return fig

def show_projected_lineups():
    st.title("FPL Matchups and Projected Lineups")

    # Scrape the EPL matchups from Rotowire
    matchups = scrape_matchups(config.ROTOWIRE_LINEUPS_URL)

    # Create a drop-down to choose the matchup to view
    selected_matchup = st.selectbox(
        "Select a Matchup", matchups, format_func=lambda x: f"{x[0]} vs {x[1]}"
    )

    if selected_matchup:
        home_team, away_team = selected_matchup
        lineups_df = scrape_rotowire_lineups(config.ROTOWIRE_LINEUPS_URL)

        home_team_df = lineups_df[lineups_df['Team'] == home_team]
        away_team_df = lineups_df[lineups_df['Team'] == away_team]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{home_team} Lineup")
            st.plotly_chart(plot_soccer_field(home_team_df, home_team), use_container_width=True)
            st.dataframe(
                home_team_df[['Position', 'Player']].set_index('Player'),
                use_container_width=True,
                height=422  # Adjust the height to ensure the entire table shows
            )

        with col2:
            st.subheader(f"{away_team} Lineup")
            st.plotly_chart(plot_soccer_field(away_team_df, away_team), use_container_width=True)
            st.dataframe(
                away_team_df[['Position', 'Player']].set_index('Player'),
                use_container_width=True,
                height=422  # Adjust the height to ensure the entire table shows
            )


