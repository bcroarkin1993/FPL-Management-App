import config
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

from scripts.common.api import (
    pull_fpl_player_stats,
    get_rotowire_player_projections,
    get_bootstrap_static
)
from scripts.common.utils import (
    clean_fpl_player_names,
    merge_fpl_players_and_projections
)


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def display_top_goal_scorers(player_statistics, position_filter, top_n=10):
    """Displays a horizontal bar chart of the top goal scorers."""
    filtered_data = player_statistics[player_statistics['position_abbrv'].isin(position_filter)]
    top_scorers = filtered_data.nlargest(top_n, 'goals_scored').sort_values(by='goals_scored', ascending=False)

    fig = px.bar(
        top_scorers,
        y=top_scorers['player'],
        x='goals_scored',
        color='team_name',
        text='goals_scored',
        title="🏆 Top Goal Scorers",
        labels={'player': 'Player', 'goals_scored': 'Goals', 'team_name': 'Team'},
        orientation='h'
    )

    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>Player</b> = %{y}<br><b>Team</b> = %{customdata[0]}<br><b>Goals</b> = %{x}<extra></extra>',
        customdata=top_scorers[['team_name']].values,
        showlegend=False
    )
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)


def display_top_assisters(player_statistics, position_filter, top_n=10):
    """Displays a horizontal bar chart of the top assisters."""
    filtered_data = player_statistics[player_statistics['position_abbrv'].isin(position_filter)]
    top_assisters = filtered_data.nlargest(top_n, 'assists')

    fig = px.bar(
        top_assisters,
        y='player',
        x='assists',
        color='team_name',
        text='assists',
        title="🎯 Top Assist Providers",
        labels={'player': 'Player', 'assists': 'Assists', 'team_name': 'Team'},
        orientation='h'
    )

    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>Player</b> = %{y}<br><b>Team</b> = %{customdata[0]}<br><b>Assists</b> = %{x}<extra></extra>',
        customdata=top_assisters[['team_name']].values,
        showlegend=False
    )
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)


def display_top_clean_sheets(player_statistics, clean_sheets_filter, top_n=10):
    """Displays top clean sheets (Defenders/Goalkeepers only)."""
    filtered_data = player_statistics[player_statistics['position_abbrv'].isin(clean_sheets_filter)]
    top_clean_sheets = filtered_data.nlargest(top_n, 'clean_sheets')

    fig = px.bar(
        top_clean_sheets,
        y='player',
        x='clean_sheets',
        color='team_name',
        text='clean_sheets',
        title="🧤 Clean Sheet Leaders",
        labels={'player': 'Player', 'clean_sheets': 'Clean Sheets', 'team_name': 'Team'},
        orientation='h'
    )

    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>Player</b> = %{y}<br><b>Team</b> = %{customdata[0]}<br><b>Clean Sheets</b> = %{x}<extra></extra>',
        customdata=top_clean_sheets[['team_name']].values,
        showlegend=False
    )
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)


def display_expected_vs_actual_goals(player_statistics, position_filter, team_filter, top_n):
    """Scatter plot for Expected vs Actual Goal Involvement."""
    filtered_data = player_statistics[
        (player_statistics['position_abbrv'].isin(position_filter)) &
        (player_statistics['team_name'].isin(team_filter))
        ].copy()

    # Data is already converted to numeric in main(), so we can safely filter/sort
    filtered_data["total_points"] = filtered_data["total_points"].apply(lambda x: max(x, 1))
    filtered_data = filtered_data.dropna(subset=["expected_goal_involvements"])
    filtered_data = filtered_data.nlargest(top_n, 'expected_goal_involvements')

    filtered_data["hover_text"] = (
            "<b>Player:</b> " + filtered_data["player"] +
            "<br><b>Position:</b> " + filtered_data["position_abbrv"] +
            "<br><b>Team Name:</b> " + filtered_data["team_name"] +
            "<br><b>Expected Goals:</b> " + filtered_data["expected_goal_involvements"].astype(str) +
            "<br><b>Actual Goal Involvement:</b> " + filtered_data["actual_goal_involvements"].astype(str) +
            "<br><b>Total Points:</b> " + filtered_data["total_points"].astype(str)
    )

    fig = px.scatter(
        filtered_data,
        x="expected_goal_involvements",
        y="actual_goal_involvements",
        color="position_abbrv",
        hover_data={"hover_text": True},
        size="total_points",
        title="Expected vs. Actual Goal Involvements",
        labels={
            "expected_goal_involvements": "Expected Goal Involvements (xGI)",
            "actual_goal_involvements": "Actual Goal Involvements (Goals + Assists)",
        },
    )

    fig.update_traces(hovertemplate="%{customdata[0]}", marker=dict(opacity=0.8))
    fig.update_layout(
        xaxis_title="Expected Goal Involvements (xGI)",
        yaxis_title="Actual Goal Involvements (Goals + Assists)",
        legend_title="Position",
        template="plotly_dark",
        xaxis=dict(tickmode="linear", dtick=5, showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
    )
    st.plotly_chart(fig, use_container_width=True)


def display_boxplot_point_distribution(player_statistics, position_filter, team_filter):
    """Box plot of points distribution."""
    filtered_data = player_statistics[
        (player_statistics['position_abbrv'].isin(position_filter)) &
        (player_statistics['team_name'].isin(team_filter))
        ]

    fig = px.box(
        filtered_data,
        x='position_abbrv',
        y='total_points',
        color='position_abbrv',
        title="Distribution of Total Points by Position",
        labels={'position_abbrv': 'Position', 'total_points': 'Total Points'},
        hover_data=['player', 'team_name']
    )

    fig.update_layout(
        xaxis_title="Player Position",
        yaxis_title="Total Points",
        boxmode="group",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# MAIN PAGE LOGIC
# ==============================================================================

def show_player_stats_page():
    st.title("📊 FPL Player Statistics Dashboard ⚽")
    st.markdown("""
    #### Track key performance metrics for Fantasy Premier League players!  
    Use the filters to explore **top goal scorers, assist providers, clean sheet leaders, and advanced stats.** """)

    # 1. FETCH RAW STATS
    fpl_player_statistics = pull_fpl_player_stats()

    if fpl_player_statistics.empty:
        st.error("Unable to load player statistics.")
        return

    # 2. DATA TYPE ENFORCEMENT (This fixes the 'object' dtype errors)
    numeric_cols = [
        'goals_scored', 'assists', 'clean_sheets',
        'expected_goal_involvements', 'total_points'
    ]
    for col in numeric_cols:
        if col in fpl_player_statistics.columns:
            fpl_player_statistics[col] = pd.to_numeric(fpl_player_statistics[col], errors='coerce').fillna(0)

    # 3. ENRICH DATA (Team Names & Positions)
    static_data = get_bootstrap_static()
    teams = static_data.get("teams", [])

    # Map Team IDs
    if teams and 'team' in fpl_player_statistics.columns:
        id_to_team = {t['id']: t['name'] for t in teams}
        fpl_player_statistics['team_name'] = fpl_player_statistics['team'].map(id_to_team)
    else:
        fpl_player_statistics['team_name'] = "Unknown"

    # Map Position IDs
    if 'element_type' in fpl_player_statistics.columns:
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        fpl_player_statistics['position_abbrv'] = fpl_player_statistics['element_type'].map(pos_map)
    else:
        fpl_player_statistics['position_abbrv'] = "UNK"

    # Ensure 'player' column
    if 'web_name' in fpl_player_statistics.columns:
        fpl_player_statistics['player'] = fpl_player_statistics['web_name']

    # Calculate actual involvement
    g = fpl_player_statistics['goals_scored']
    a = fpl_player_statistics['assists']
    fpl_player_statistics['actual_goal_involvements'] = g + a

    # 4. MERGE PROJECTIONS (Optional)
    if config.ROTOWIRE_URL:
        fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

        if 'player' in fpl_player_statistics.columns:
            fpl_player_statistics['player'] = fpl_player_statistics['player'].apply(clean_fpl_player_names)

        if not fpl_player_projections.empty and 'Player' in fpl_player_projections.columns:
            fpl_player_projections['Player'] = fpl_player_projections['Player'].apply(clean_fpl_player_names)

            fpl_player_statistics = pd.merge(
                fpl_player_statistics,
                fpl_player_projections[['Player', 'Price']],
                left_on='player',
                right_on='Player',
                how='left'
            ).rename(columns={'Price': 'price'})

            if 'Player' in fpl_player_statistics.columns:
                fpl_player_statistics.drop(columns=['Player'], inplace=True)

    # 5. FILTERS
    st.subheader("🔍 Filters")
    col_filter_1, col_filter_2, col_filter_3 = st.columns(3)

    top_n = col_filter_1.slider("Select Number of Players", 5, 20, 10)

    available_positions = sorted(fpl_player_statistics['position_abbrv'].dropna().unique().tolist())
    position_filter = col_filter_2.multiselect(
        "Filter by Position", options=available_positions, default=available_positions
    )

    clean_sheets_filter = col_filter_3.multiselect(
        "Filter Clean Sheets by Position", options=['GK', 'DEF'], default=['GK', 'DEF']
    )

    fpl_player_statistics['team_name'] = fpl_player_statistics['team_name'].astype(str)
    team_options = sorted(fpl_player_statistics['team_name'].unique().tolist())
    team_filter = st.multiselect(
        "Filter by Team (For Expected Goals Chart)", options=team_options, default=team_options
    )

    # 6. RENDER CHARTS
    if not fpl_player_statistics.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            display_top_goal_scorers(fpl_player_statistics, position_filter, top_n)
        with col2:
            display_top_assisters(fpl_player_statistics, position_filter, top_n)
        with col3:
            display_top_clean_sheets(fpl_player_statistics, clean_sheets_filter, top_n)

        col4, col5 = st.columns(2)
        with col4:
            display_expected_vs_actual_goals(fpl_player_statistics, position_filter, team_filter, top_n)
        with col5:
            display_boxplot_point_distribution(fpl_player_statistics, position_filter, team_filter)
    else:
        st.error("No data available to display.")