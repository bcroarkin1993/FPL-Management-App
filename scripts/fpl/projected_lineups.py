from bs4 import BeautifulSoup
from collections import defaultdict
import config
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scripts.common.error_helpers import get_logger
from scripts.fpl.injuries import get_fpl_availability_df
from scripts.common.utils import get_classic_bootstrap_static
from scripts.common.player_matching import canonical_normalize

_logger = get_logger("fpl_app.projected_lineups")

def extract_players(section, team_type, team_name, matchup_index):
    """
    Extracts valid players from the given section, excluding those listed in the Injuries section.

    Parameters:
    - section (BeautifulSoup): The section containing the team's lineup.
    - team_type (str): 'home' or 'visit' to determine the team type.
    - team_name (str): The name of the team.
    - matchup_index (int): Index of the matchup to track which game this lineup belongs to.

    Returns:
    - list: A list of tuples containing (Team, Position, Player, MatchupIndex) for valid players.
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
                    player_list.append((team_name, position, player_name, matchup_index))
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
    - DataFrame containing the team names, player names, positions, and matchup index.
    """
    # Send a request to the Rotowire lineups page
    try:
        page = requests.get(url, timeout=30)
    except Exception as e:
        _logger.warning("Failed to fetch Rotowire lineups from %s: %s", url, e)
        return pd.DataFrame(columns=['Team', 'Position', 'Player', 'MatchupIndex'])
    soup = BeautifulSoup(page.content, 'html.parser')

    # Initialize an empty list to store match data
    all_players = []

    # Find all lineup sections (home and away matchups)
    lineup_sections = soup.find_all('div', class_='lineup__main')

    # Iterate through each section to extract team and player data
    for matchup_index, section in enumerate(lineup_sections):
        try:
            # Extract home and away team names
            home_team = section.find_previous('div', class_='lineup__mteam is-home').text.strip()
            away_team = section.find_previous('div', class_='lineup__mteam is-visit').text.strip()

            # Extract players while excluding those listed in the Injuries section
            home_players = extract_players(section, 'home', home_team, matchup_index)
            away_players = extract_players(section, 'visit', away_team, matchup_index)

            # Add players to the list
            all_players.extend(home_players + away_players)

        except AttributeError as e:
            _logger.warning("Error parsing lineup section (HTML structure may have changed): %s", e)

    # Convert the data to a pandas DataFrame
    lineups_df = pd.DataFrame(all_players, columns=['Team', 'Position', 'Player', 'MatchupIndex'])

    return lineups_df

def scrape_matchups(url):
    """
    Scrapes the matchups from the Rotowire page.

    Returns:
    - List of tuples: (home_team, away_team, matchup_index)
    """
    try:
        page = requests.get(url, timeout=30)
    except Exception as e:
        _logger.warning("Failed to fetch Rotowire matchups from %s: %s", url, e)
        return []
    soup = BeautifulSoup(page.content, 'html.parser')
    matchups_section = soup.find_all('div', class_='lineup__matchup')

    matchups = []
    for idx, matchup in enumerate(matchups_section):
        try:
            home_team = matchup.find('div', class_='lineup__mteam is-home').text.strip()
            away_team = matchup.find('div', class_='lineup__mteam is-visit').text.strip()
            matchups.append((home_team, away_team, idx))
        except AttributeError as e:
            _logger.warning("Error parsing matchup (HTML structure may have changed): %s", e)
            continue

    if not matchups and matchups_section:
        _logger.warning("Rotowire: Found %d matchup sections but failed to parse any", len(matchups_section))

    return matchups


def get_player_data_map():
    """
    Fetches player data from FPL API and creates a lookup map by player name.
    Uses multiple keys (web_name, full_name, normalized versions) for better matching.

    Returns:
    - dict: {player_name: {form, points_per_game, chance_of_playing, status, news, ...}}
    """
    player_map = {}
    norm_to_data = {}  # Normalized name -> player data (for fuzzy lookup)

    try:
        # Get availability data
        avail_df = get_fpl_availability_df()
        avail_lookup = {}
        if not avail_df.empty:
            for _, row in avail_df.iterrows():
                web_name = row.get('Web_Name', '')
                if web_name:
                    avail_lookup[web_name] = {
                        'play_pct': row.get('PlayPct', 100),
                        'status_bucket': row.get('StatusBucket', 'Available'),
                        'news': row.get('News', '')
                    }

        # Get bootstrap data for form and other stats
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap:
            return player_map

        elements = bootstrap.get('elements', [])
        teams = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}

        for elem in elements:
            web_name = elem.get('web_name', '')
            first_name = elem.get('first_name', '')
            second_name = elem.get('second_name', '')
            full_name = f"{first_name} {second_name}".strip()

            if not web_name:
                continue

            # Build player data
            pdata = {
                'form': float(elem.get('form', 0) or 0),
                'points_per_game': float(elem.get('points_per_game', 0) or 0),
                'total_points': elem.get('total_points', 0),
                'minutes': elem.get('minutes', 0),
                'goals_scored': elem.get('goals_scored', 0),
                'assists': elem.get('assists', 0),
                'clean_sheets': elem.get('clean_sheets', 0),
                'chance_of_playing': elem.get('chance_of_playing_this_round'),
                'status': elem.get('status', 'a'),
                'news': elem.get('news', ''),
                'team': teams.get(elem.get('team'), ''),
                'play_pct': 100,
                'status_bucket': 'Available',
            }

            # Merge availability data
            if web_name in avail_lookup:
                pdata.update(avail_lookup[web_name])

            # Store under multiple keys for better matching
            player_map[web_name] = pdata
            if full_name and full_name != web_name:
                player_map[full_name] = pdata
            if second_name and second_name != web_name:
                player_map[second_name] = pdata

            # Store normalized version for fuzzy lookup
            norm_key = canonical_normalize(full_name)
            if norm_key:
                norm_to_data[norm_key] = pdata

        # Store the normalized lookup for use in matching
        player_map['_norm_lookup'] = norm_to_data

    except Exception as e:
        _logger.warning("Failed to fetch player data for lineup enhancement: %s", e)

    return player_map


def lookup_player_data(player_name, player_data_map):
    """
    Look up player data with fallback to normalized name matching.
    Handles cases where Rotowire uses:
    - Shortened names: 'Bruno Fernandes' -> 'Bruno Borges Fernandes'
    - Abbreviated names: 'R. Sanchez' -> 'Robert Sanchez'
    """
    # Direct lookup
    if player_name in player_data_map:
        return player_data_map[player_name]

    # Try normalized lookup (exact match)
    norm_lookup = player_data_map.get('_norm_lookup', {})
    norm_name = canonical_normalize(player_name)
    if norm_name in norm_lookup:
        return norm_lookup[norm_name]

    # Handle abbreviated first names like "R. Sanchez" or "H. Maguire"
    # Pattern: single letter followed by period and space, then last name
    import re
    abbrev_match = re.match(r'^([A-Z])\.\s+(.+)$', player_name)
    if abbrev_match:
        first_initial = abbrev_match.group(1).lower()
        last_name = abbrev_match.group(2)
        norm_last = canonical_normalize(last_name)

        # Search for players whose first name starts with the initial and last name matches
        for key, value in norm_lookup.items():
            key_parts = key.split()
            if len(key_parts) >= 2:
                if key_parts[0].startswith(first_initial) and key_parts[-1] == norm_last:
                    return value

        # Also try direct last name match in player_map
        if last_name in player_data_map:
            return player_data_map[last_name]

    # Try partial normalized match: find names that start and end with our words
    # e.g., "bruno fernandes" should match "bruno borges fernandes"
    if norm_name and len(norm_name.split()) >= 2:
        norm_parts = norm_name.split()
        first_word = norm_parts[0]
        last_word = norm_parts[-1]

        for key, value in norm_lookup.items():
            key_parts = key.split()
            if len(key_parts) >= 2:
                # Match if first and last words align
                if key_parts[0] == first_word and key_parts[-1] == last_word:
                    return value

    # Try partial match on last name only (for single-name lookups like "Casemiro")
    parts = player_name.split()
    if len(parts) >= 1:
        last_name = parts[-1]
        if last_name in player_data_map:
            return player_data_map[last_name]

    return {}


def get_availability_color(status_bucket, play_pct=None):
    """Returns color based on player availability status."""
    if status_bucket == 'Out':
        return '#e74c3c'  # Red
    elif status_bucket == 'Doubtful':
        return '#e67e22'  # Orange
    elif status_bucket == 'Questionable':
        return '#f1c40f'  # Yellow
    elif status_bucket == 'Likely':
        return '#2ecc71'  # Light green
    else:  # Available
        return '#27ae60'  # Green


def get_form_color(form):
    """Returns color based on player form rating."""
    if form >= 6:
        return '#27ae60'  # Green - excellent
    elif form >= 4:
        return '#2ecc71'  # Light green - good
    elif form >= 2:
        return '#f1c40f'  # Yellow - average
    else:
        return '#e74c3c'  # Red - poor

def plot_soccer_field(player_df, team_name, player_data_map=None):
    """
    Plots players on a soccer field based on their positions for a specific team.
    Enhanced with player form and availability indicators.

    Parameters:
    - player_df (pd.DataFrame): A DataFrame containing 'Position' and 'Player' columns.
    - team_name (str): The name of the team, displayed as the title above the field.
    - player_data_map (dict): Optional player data for form/availability enhancement.
    """
    if player_data_map is None:
        player_data_map = {}

    # Map Rotowire team names to TEAM_COLORS keys
    team_name_map = {
        'Manchester United': 'Man Utd',
        'Manchester City': 'Man City',
        'Tottenham Hotspur': 'Spurs',
        'Nottingham Forest': "Nott'm Forest",
        'Wolverhampton': 'Wolves',
        'West Ham United': 'West Ham',
        'Leicester City': 'Leicester',
        'Ipswich Town': 'Ipswich',
        'AFC Bournemouth': 'Bournemouth',
        'Brighton and Hove Albion': 'Brighton',
        'Newcastle United': 'Newcastle',
    }
    mapped_team_name = team_name_map.get(team_name, team_name)

    # Define positions with (x, y) coordinates
    position_mapping = {
        'GK': (5, 0.3),  # Goalkeeper
        'DL': (1.5, 1.5), 'DC': (5, 1.2), 'DR': (8.5, 1.5),  # Defenders
        'DML': (2, 2.1), 'DMC': (5, 2.1), 'DMR': (8, 2.1),  # Defensive Midfielders
        'ML': (1.5, 2.9), 'MC': (5, 2.7), 'MR': (8.5, 2.9),  # Midfielders
        'AML': (1.4, 3.5), 'AMC': (5, 3.5), 'AMR': (8.6, 3.5),  # Attacking Midfielders
        'FL': (2, 4.4), 'FWL': (2, 4.4), 'FC': (5, 4.6), 'FW': (5, 4.4), 'FWR': (8, 4.4), 'FR': (8, 4.4)  # Forwards
    }

    # Get the team's primary and secondary colors (fallback to visible colors, not background color)
    colors = config.TEAM_COLORS.get(mapped_team_name, {"primary": "#3498db", "secondary": "#FFFFFF"})
    primary_color = colors["primary"]
    secondary_color = colors["secondary"]

    # Group players by position
    grouped_players = defaultdict(list)
    for _, row in player_df.iterrows():
        grouped_players[row['Position']].append(row['Player'])

    # Create a Plotly figure
    fig = go.Figure()

    # Draw simple field lines (original style)
    # Field boundary
    fig.add_shape(type="rect", x0=0, y0=-0.3, x1=10, y1=5.2,
                  line=dict(color="white", width=2))

    # Center line
    fig.add_shape(type="line", x0=0, y0=2.5, x1=10, y1=2.5,
                  line=dict(color="white", dash="dash"))

    # Add players to the field
    for position, players in grouped_players.items():
        for i, player_name in enumerate(players):
            x, y = position_mapping.get(position, (5, 2.75))  # Default to center if not found

            # Adjust for multiple players in the same central position
            if len(players) == 2 and position in ['DC', 'MC', 'FC', 'FW', 'DMC', 'AMC']:
                x += -1.5 if i == 0 else 1.5
            elif len(players) == 3 and position in ['DC', 'MC', 'FC', 'DMC', 'AMC']:
                if i == 0:
                    x -= 2.0
                elif i == 2:
                    x += 2.0

            # Get player data for enhanced display
            pdata = lookup_player_data(player_name, player_data_map)
            form = pdata.get('form', 0)
            status_bucket = pdata.get('status_bucket', 'Available')
            play_pct = pdata.get('play_pct', 100)
            news = pdata.get('news', '')
            total_points = pdata.get('total_points', 0)
            goals = pdata.get('goals_scored', 0)
            assists = pdata.get('assists', 0)

            # Determine marker border color based on availability
            border_color = get_availability_color(status_bucket, play_pct)

            # Build hover text with player details
            hover_lines = [
                f"<b>{player_name}</b>",
                f"Position: {position}",
                f"Form: {form:.1f}" if form else "Form: N/A",
                f"Total Points: {total_points}",
            ]
            if goals or assists:
                hover_lines.append(f"G: {goals} | A: {assists}")
            if status_bucket != 'Available':
                hover_lines.append(f"Status: {status_bucket} ({play_pct:.0f}%)" if play_pct else f"Status: {status_bucket}")
            if news:
                # Truncate long news
                news_short = news[:50] + "..." if len(news) > 50 else news
                hover_lines.append(f"News: {news_short}")

            hover_text = "<br>".join(hover_lines)

            # Add player marker
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=primary_color,
                    line=dict(color=border_color, width=2)
                ),
                text=player_name,
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=12),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False
            ))

            # Add form indicator below player name (small colored dot)
            if form > 0:
                form_color = get_form_color(form)
                fig.add_trace(go.Scatter(
                    x=[x], y=[y - 0.35],
                    mode='markers+text',
                    marker=dict(size=8, color=form_color, symbol='circle'),
                    text=f"{form:.1f}",
                    textposition="bottom center",
                    textfont=dict(color="#FFFFFF", size=9),
                    hoverinfo='skip',
                    showlegend=False
                ))

    fig.update_layout(
        width=500, height=600,
        xaxis=dict(range=[0, 10], visible=False),
        yaxis=dict(range=[-0.5, 5.5], visible=False),
        plot_bgcolor="#228B22",
        paper_bgcolor="#1a1a2e",
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
    )

    return fig

def render_player_cards_html(player_df, player_data_map):
    """Renders all player cards as a single HTML block."""
    cards = []

    for _, row in player_df.iterrows():
        player_name = row['Player']
        position = row['Position']
        pdata = lookup_player_data(player_name, player_data_map)

        form = pdata.get('form', 0)
        status_bucket = pdata.get('status_bucket', 'Available')
        total_points = pdata.get('total_points', 0)
        goals = pdata.get('goals_scored', 0)
        assists = pdata.get('assists', 0)
        news = pdata.get('news', '')

        # Colors
        status_colors = {'Out': '#e74c3c', 'Doubtful': '#e67e22', 'Questionable': '#f1c40f', 'Likely': '#2ecc71', 'Available': '#27ae60'}
        status_color = status_colors.get(status_bucket, '#27ae60')
        form_color = get_form_color(form) if form > 0 else '#888'

        # Status badge
        status_badge = ''
        if status_bucket != 'Available':
            status_badge = f'<span style="background:{status_color};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-left:8px;">{status_bucket}</span>'

        # News line
        news_line = ''
        if news:
            news_short = (news[:50] + '...') if len(news) > 50 else news
            news_line = f'<div style="color:#aaa;font-size:0.8em;margin-top:2px;font-style:italic;">{news_short}</div>'

        # Stats display - show N/A if no data found
        if pdata:
            stats_html = f'<span style="color:{form_color};font-weight:600;">Form: {form:.1f}</span><span style="color:#2ecc71;margin-left:15px;">Pts: {total_points}</span><span style="color:#fff;margin-left:15px;">G:{goals} A:{assists}</span>'
        else:
            stats_html = '<span style="color:#999;">Stats unavailable</span>'

        # Build card with solid background for better readability
        card = f'<div style="display:flex;align-items:center;padding:12px 14px;margin-bottom:8px;background:#2c3e50;border-radius:8px;border-left:4px solid {form_color};box-shadow:0 2px 4px rgba(0,0,0,0.2);"><div style="min-width:40px;text-align:center;background:#34495e;padding:6px 10px;border-radius:4px;margin-right:12px;"><span style="color:#fff;font-weight:bold;">{position}</span></div><div style="flex:1;"><div style="color:#fff;font-weight:bold;font-size:1.05em;">{player_name}{status_badge}</div><div style="margin-top:6px;font-size:0.9em;">{stats_html}</div>{news_line}</div></div>'
        cards.append(card)

    return ''.join(cards)


def show_projected_lineups():
    st.title("Projected Lineups")
    st.write("View projected starting lineups with player form and availability status.")

    # Scrape the EPL matchups from Rotowire
    matchups = scrape_matchups(config.ROTOWIRE_LINEUPS_URL)

    if not matchups:
        st.warning("No matchups available. Rotowire may not have published lineups yet.")
        return

    # Create a drop-down to choose the matchup to view
    selected_matchup = st.selectbox(
        "Select a Matchup",
        matchups,
        format_func=lambda x: f"{x[0]} vs {x[1]}"
    )

    if selected_matchup:
        home_team, away_team, matchup_index = selected_matchup

        # Fetch lineup data
        lineups_df = scrape_rotowire_lineups(config.ROTOWIRE_LINEUPS_URL)

        # Filter by BOTH team name AND matchup index to fix duplicate team bug
        home_team_df = lineups_df[
            (lineups_df['Team'] == home_team) &
            (lineups_df['MatchupIndex'] == matchup_index)
        ]
        away_team_df = lineups_df[
            (lineups_df['Team'] == away_team) &
            (lineups_df['MatchupIndex'] == matchup_index)
        ]

        # Fetch player data for enhancements
        with st.spinner("Loading player data..."):
            player_data_map = get_player_data_map()

        # Add legend for availability colors
        st.markdown("""
        <div style="display:flex;gap:15px;margin-bottom:15px;flex-wrap:wrap;">
            <span style="font-size:0.85em;color:#888;">Availability:</span>
            <span style="font-size:0.85em;"><span style="color:#27ae60;">●</span> Available</span>
            <span style="font-size:0.85em;"><span style="color:#2ecc71;">●</span> Likely</span>
            <span style="font-size:0.85em;"><span style="color:#f1c40f;">●</span> Questionable</span>
            <span style="font-size:0.85em;"><span style="color:#e67e22;">●</span> Doubtful</span>
            <span style="font-size:0.85em;"><span style="color:#e74c3c;">●</span> Out</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{home_team}")
            home_fig = plot_soccer_field(home_team_df, home_team, player_data_map)
            st.plotly_chart(home_fig, use_container_width=True, key=f"home_{matchup_index}")

            # Enhanced player list
            st.markdown("##### Squad Details")
            if not home_team_df.empty:
                cards_html = render_player_cards_html(home_team_df, player_data_map)
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.info("No lineup data available for this team.")

        with col2:
            st.subheader(f"{away_team}")
            away_fig = plot_soccer_field(away_team_df, away_team, player_data_map)
            st.plotly_chart(away_fig, use_container_width=True, key=f"away_{matchup_index}")

            # Enhanced player list
            st.markdown("##### Squad Details")
            if not away_team_df.empty:
                cards_html = render_player_cards_html(away_team_df, player_data_map)
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.info("No lineup data available for this team.")


