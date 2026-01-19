import config
import pandas as pd
import plotly.express as px
import streamlit as st
from scripts.common.utils import clean_fpl_player_names, get_rotowire_player_projections, pull_fpl_player_stats

def display_top_goal_scorers(player_statistics, position_filter, top_n=10):
    """
    Displays a horizontal bar chart of the top goal scorers in the league using Streamlit.

    Parameters:
    - player_statistics: DataFrame containing player statistics.
    - position_filter (str, optional): Filter based on player position.
    - top_n (int, optional): Number of top players to display. Default is 10.
    """
    # Ensure goals_scored is numeric
    player_statistics['goals_scored'] = pd.to_numeric(player_statistics['goals_scored'], errors='coerce')

    # Filters data based on app filters
    filtered_data = player_statistics[player_statistics['position_abbrv'].isin(position_filter)]

    # Select top goal scorers and sort in descending order
    top_scorers = filtered_data.nlargest(top_n, 'goals_scored').sort_values(by='goals_scored', ascending=False)

    # Create the horizontal bar plot
    fig = px.bar(
        top_scorers,
        y=top_scorers['player'],  # Player names on y-axis
        x='goals_scored',
        color='team_name',
        text='goals_scored',  # Display Goals at the end of the bars
        title="🏆 Top Goal Scorers",
        labels={'player': 'Player', 'goals_scored': 'Goals', 'team_name': 'Team'},
        orientation='h'  # Horizontal bar chart
    )

    # Update text and hover template
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>Player</b> = %{y}<br>' +
                      '<b>Team</b> = %{customdata[0]}<br>' +
                      '<b>Goals</b> = %{x}<extra></extra>',
        customdata=top_scorers[['team_name']].values,
        showlegend=False
    )

    # Reverse the order so the highest scorer is at the top
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending')  # Ensures correct descending order
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def display_top_assisters(player_statistics, position_filter, top_n=10):
    """
    Displays a horizontal bar chart of the top assisters in the league using Streamlit.

    Parameters:
    - player_statistics: DataFrame containing player statistics.
    - position_filter (str, optional): Filter based on player position.
    - top_n (int, optional): Number of top players to display. Default is 10.
    """
    # Filters data based on app filters
    filtered_data = player_statistics[player_statistics['position_abbrv'].isin(position_filter)]

    # Get top 10 assists
    top_assisters = filtered_data.nlargest(top_n, 'assists')

    # Create the horizontal bar plot
    fig = px.bar(
        top_assisters,
        y='player',
        x='assists',
        color='team_name',
        text='assists',  # Display Assists at the end of the bars
        title="🎯 Top Assist Providers",
        labels={'player': 'Player', 'assists': 'Assists', 'team_name': 'Team'},
        orientation='h'  # Horizontal bar chart
    )

    # Update text and hover template
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>Player</b> = %{y}<br>' +
                      '<b>Team</b> = %{customdata[0]}<br>' +
                      '<b>Assists</b> = %{x}<extra></extra>',
        customdata=top_assisters[['team_name']].values,
        showlegend=False
    )

    # Reverse the order so the highest scorer is at the top
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending')  # Ensures correct descending order
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def display_top_clean_sheets(player_statistics, clean_sheets_filter, top_n=10):
    """
    Displays a horizontal bar chart of the top clean sheets in the league using Streamlit. This list is limited to just
    FPL defenders and goalkeepers.

    Parameters:
    - player_statistics: DataFrame containing player statistics.
    - position_filter (str, optional): Filter based on player position, limited to DEF and GK as options
    - top_n (int, optional): Number of top players to display. Default is 10.
    """
    # Filter for goalkeepers and/or defenders
    filtered_data = player_statistics[player_statistics['position_abbrv'].isin(clean_sheets_filter)]

    # Get top 10 clean sheets
    top_clean_sheets = filtered_data.nlargest(top_n, 'clean_sheets')

    # Create the horizontal bar plot
    fig = px.bar(
        top_clean_sheets,
        y='player',
        x='clean_sheets',
        color='team_name',
        text='clean_sheets',  # Display Clean Sheets at the end of the bars
        title="🧤 Clean Sheet Leaders",
        labels={'player': 'Player', 'clean_sheets': 'Clean Sheets', 'team_name': 'Team'},
        orientation='h'  # Horizontal bar chart
    )

    # Update text and hover template
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>Player</b> = %{y}<br>' +
                      '<b>Team</b> = %{customdata[0]}<br>' +
                      '<b>Clean Sheets</b> = %{x}<extra></extra>',
        customdata=top_clean_sheets[['team_name']].values,
        showlegend=False
    )

    # Reverse the order so the highest scorer is at the top
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending')  # Ensures correct descending order
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def display_expected_vs_actual_goals(player_statistics, position_filter, team_filter, top_n):
    """

    Parameters:
    - player_statistics: DataFrame containing player statistics.
    - position_filter (str, optional): Filter based on player position.
    - team_filter (str, optional): Filter based on team.
    - top_n (int, optional): Number of top players to display. Default is 10.
    """
    # Filter data by Streamlit app input
    filtered_data = player_statistics[
        (player_statistics['position_abbrv'].isin(position_filter)) &
        (player_statistics['team_name'].isin(team_filter))
    ]

    # Replace NaN or negative values in total_points to avoid invalid sizes
    filtered_data["total_points"] = filtered_data["total_points"].fillna(0)
    filtered_data["total_points"] = filtered_data["total_points"].apply(lambda x: max(x, 1))  # Ensure positive size

    # Drop rows where expected_goal_involvements is NaN after conversion
    filtered_data = filtered_data.dropna(subset=["expected_goal_involvements"])

    # Get top_n rows
    filtered_data = filtered_data.nlargest(top_n, 'expected_goal_involvements')

    # Create hover text with bold labels
    filtered_data["hover_text"] = (
        "<b>Player:</b> " + filtered_data["player"] +
        "<br><b>Position:</b> " + filtered_data["position_abbrv"] +
        "<br><b>Team Name:</b> " + filtered_data["team_name"] +
        "<br><b>Expected Goals:</b> " + filtered_data["expected_goal_involvements"].astype(str) +
        "<br><b>Actual Goal Involvement:</b> " + filtered_data["actual_goal_involvements"].astype(str) +
        "<br><b>Total Points:</b> " + filtered_data["total_points"].astype(str)
    )

    # Create scatter plot
    fig = px.scatter(
        filtered_data,
        x="expected_goal_involvements",
        y="actual_goal_involvements",
        color="position_abbrv",
        hover_data={"hover_text": True},  # Use formatted hover text
        size="total_points",
        title="Expected vs. Actual Goal Involvements",
        labels={
            "expected_goal_involvements": "Expected Goal Involvements (xGI)",
            "actual_goal_involvements": "Actual Goal Involvements (Goals + Assists)",
        },
    )

    # Remove other labels and only display hover_text
    fig.update_traces(hovertemplate="%{customdata[0]}", marker=dict(opacity=0.8))

    # Update x-axis to ensure ascending order and consistent intervals
    fig.update_layout(
        xaxis_title="Expected Goal Involvements (xGI)",
        yaxis_title="Actual Goal Involvements (Goals + Assists)",
        legend_title="Position",
        template="plotly_dark",
        xaxis=dict(
            tickmode="linear",
            dtick=5,  # Adjust tick intervals for readability
            showgrid=True,
            zeroline=True,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
        ),
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def display_boxplot_point_distribution(player_statistics, position_filter, team_filter):
    """

    Parameters:
    - player_statistics: DataFrame containing player statistics.
    - position_filter (str, optional): Filter based on player position.
    - team_filter (str, optional): Filter based on team.
    """
    # Filter data by Streamlit app input
    filtered_data = player_statistics[
        (player_statistics['position_abbrv'].isin(position_filter)) &
        (player_statistics['team_name'].isin(team_filter))
    ]

    # Create the box plot
    fig = px.box(
        filtered_data,
        x='position_abbrv',
        y='total_points',
        color='position_abbrv',  # Different colors for each position
        title="Distribution of Total Points by Position",
        labels={'position_abbrv': 'Position', 'total_points': 'Total Points'},
        hover_data=['player', 'team_name']
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Player Position",
        yaxis_title="Total Points",
        boxmode="group",  # Groups positions together
        template="plotly_white"
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def show_player_stats_page():
    # Page Title and Description with Emojis and Image
    st.title("📊 FPL Player Statistics Dashboard ⚽")
    st.markdown("""
    #### Track key performance metrics for Fantasy Premier League players!  
    Use the filters to explore **top goal scorers, assist providers, clean sheet leaders, and advanced stats.**  
    """)

    # Pull FPL player stats
    fpl_player_statistics = pull_fpl_player_stats()

    # Pull FPL player projections from Rotowire (if available), for player name cleaning
    if config.ROTOWIRE_URL:
        fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

        # Clean FPL player names
        fpl_player_statistics = clean_fpl_player_names(fpl_player_statistics, fpl_player_projections)

    # Print out FPL player stats, if desired
    print("FPL Player Statistics: \n", fpl_player_statistics)

    # Filters Section
    st.subheader("🔍 Filters")

    col_filter_1, col_filter_2, col_filter_3 = st.columns(3)

    # Number of players to display
    top_n = col_filter_1.slider("Select Number of Players", min_value=5, max_value=20, value=10)

    # Position filter (For Goals & Assists)
    available_positions = fpl_player_statistics['position_abbrv'].unique().tolist()
    position_filter = col_filter_2.multiselect(
        "Filter by Position", options=available_positions, default=available_positions,
        placeholder="Select Positions"
    )

    # Clean sheets filter (GK + DEF only)
    clean_sheets_filter = col_filter_3.multiselect(
        "Filter Clean Sheets by Position", options=['GK', 'DEF'], default=['GK', 'DEF'],
        placeholder="Select GK or DEF"
    )

    # Team filter for Expected vs Actual Goal Involvement (Dropdown stays collapsed until clicked)
    team_options = sorted(fpl_player_statistics['team_name'].unique().tolist())
    team_filter = st.multiselect(
        "Filter by Team (For Expected Goals Chart)", options=team_options, default=team_options,
        placeholder="Select Teams"
    )

    # Create Streamlit visuals
    if not fpl_player_statistics.empty:

        ### --- Layout of Visuals ---
        # Top row: Goal Scorers, Assist Leaders, Clean Sheets
        col1, col2, col3 = st.columns(3)
        with col1:
            display_top_goal_scorers(fpl_player_statistics, position_filter, top_n)
        with col2:
            display_top_assisters(fpl_player_statistics, position_filter, top_n)
        with col3:
            display_top_clean_sheets(fpl_player_statistics, clean_sheets_filter, top_n)

        # Bottom row: Expected vs. Actual Goals and Box Plot
        col4, col5 = st.columns(2)
        with col4:
            display_expected_vs_actual_goals(fpl_player_statistics, position_filter, team_filter, top_n)
        with col5:
            display_boxplot_point_distribution(fpl_player_statistics, position_filter, team_filter)

    else:
        st.error("No data available at the URL provided.")
