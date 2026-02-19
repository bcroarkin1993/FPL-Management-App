import config
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scripts.common.utils import (
    clean_fpl_player_names,
    get_fixture_difficulty_grid,
    get_rotowire_player_projections,
    pull_fpl_player_stats,
)
from scripts.common.styled_tables import render_styled_table

# -----------------------------------------------------------------------------
# Column Configuration for Advanced Stats Table
# -----------------------------------------------------------------------------

# key: (display_name, high_is_good, format_type, category)
# high_is_good: True = green for high, False = green for low, None = no coloring
STAT_COLUMNS = {
    # Info columns (no coloring)
    "player": ("Player", None, "text", "info"),
    "team_name": ("Team", None, "text", "info"),
    "position_abbrv": ("Pos", None, "text", "info"),
    "fdr_avg_3": ("FDR 3GW", False, ".2f", "info"),  # Low is good
    "price": ("Price", None, "price", "info"),
    "ownership": ("Own%", None, ".1f", "info"),

    # Playing Time
    "minutes": ("Mins", True, ",d", "playing"),
    "starts": ("Starts", True, ",d", "playing"),
    "avg_mins_game": ("Avg Min/Gm", True, ".1f", "playing"),
    "avg_mins_start": ("Avg Min/St", True, ".1f", "playing"),
    "start_pct": ("Start%", True, ".1f", "playing"),

    # Points
    "total_points": ("Pts", True, ",d", "points"),
    "points_per_game": ("PPG", True, ".2f", "points"),
    "bonus": ("Bonus", True, ",d", "points"),
    "bps": ("BPS", True, ",d", "points"),

    # Attacking - Goals
    "goals_scored": ("Goals", True, ",d", "attacking"),
    "goals_per_90": ("G/90", True, ".2f", "attacking"),
    "expected_goals": ("xG", True, ".2f", "attacking"),
    "xg_per_90": ("xG/90", True, ".2f", "attacking"),

    # Attacking - Assists
    "assists": ("Ast", True, ",d", "attacking"),
    "assists_per_90": ("A/90", True, ".2f", "attacking"),
    "expected_assists": ("xA", True, ".2f", "attacking"),
    "xa_per_90": ("xA/90", True, ".2f", "attacking"),

    # Attacking - Goal Involvements
    "goal_involvements": ("GI", True, ",d", "attacking"),
    "gi_per_90": ("GI/90", True, ".2f", "attacking"),
    "expected_goal_involvements": ("xGI", True, ".2f", "attacking"),
    "xgi_per_90": ("xGI/90", True, ".2f", "attacking"),

    # Regression/Over-Under Performance (positive = over-performing, may regress down)
    "g_minus_xg": ("G-xG", None, "+.2f", "regression"),
    "a_minus_xa": ("A-xA", None, "+.2f", "regression"),
    "gi_minus_xgi": ("GI-xGI", None, "+.2f", "regression"),

    # Defensive
    "clean_sheets": ("CS", True, ",d", "defensive"),
    "goals_conceded": ("GC", False, ",d", "defensive"),  # Low is good
    "expected_goals_conceded": ("xGC", False, ".2f", "defensive"),  # Low is good
    "saves": ("Saves", True, ",d", "defensive"),

    # ICT
    "ict_index": ("ICT", True, ".1f", "ict"),
    "influence": ("Infl", True, ".1f", "ict"),
    "creativity": ("Crea", True, ".1f", "ict"),
    "threat": ("Threat", True, ".1f", "ict"),

    # Misc
    "own_goals": ("OG", False, ",d", "misc"),  # Low is good
    "penalties_saved": ("PS", True, ",d", "misc"),
    "penalties_missed": ("PM", False, ",d", "misc"),  # Low is good
    "form": ("Form", True, ".1f", "misc"),
}

COLUMN_PRESETS = {
    "Essential": ["player", "team_name", "position_abbrv", "total_points", "minutes", "goals_scored", "assists", "clean_sheets"],
    "Attacking": ["player", "team_name", "position_abbrv", "goals_scored", "goals_per_90", "expected_goals", "xg_per_90", "assists", "assists_per_90", "expected_assists", "xa_per_90"],
    "Defensive": ["player", "team_name", "position_abbrv", "clean_sheets", "goals_conceded", "expected_goals_conceded", "saves"],
    "Per 90": ["player", "team_name", "position_abbrv", "goals_per_90", "xg_per_90", "assists_per_90", "xa_per_90", "gi_per_90", "xgi_per_90"],
    "ICT Focus": ["player", "team_name", "position_abbrv", "ict_index", "influence", "creativity", "threat", "total_points"],
    "Fixture Focus": ["player", "team_name", "position_abbrv", "fdr_avg_3", "form", "total_points", "expected_goal_involvements"],
    "GK Stats": ["player", "team_name", "clean_sheets", "saves", "goals_conceded", "expected_goals_conceded", "penalties_saved", "bonus"],
    "Regression": ["player", "team_name", "position_abbrv", "goals_scored", "expected_goals", "g_minus_xg", "assists", "expected_assists", "a_minus_xa", "gi_minus_xgi"],
}


# -----------------------------------------------------------------------------
# Helper Functions for Advanced Stats
# -----------------------------------------------------------------------------

def calculate_per_90(value, minutes, min_minutes=90):
    """Calculate per-90 statistic."""
    if pd.isna(value) or pd.isna(minutes) or minutes < min_minutes:
        return np.nan
    return (value / minutes) * 90


def get_team_fdr_avg(team_short: str, fdr_avg_dict: dict) -> float:
    """Get average FDR for next 3 gameweeks for a team."""
    return fdr_avg_dict.get(team_short, 3.0)  # 3.0 = neutral default


def prepare_advanced_stats_df(player_df: pd.DataFrame, min_minutes: int = 90) -> pd.DataFrame:
    """
    Prepare the advanced stats DataFrame with all calculated columns.
    """
    df = player_df.copy()

    # Convert numeric columns
    numeric_cols = ['minutes', 'starts', 'goals_scored', 'assists', 'expected_goals',
                    'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
                    'goals_conceded', 'saves', 'clean_sheets', 'total_points', 'points_per_game',
                    'bonus', 'bps', 'creativity', 'influence', 'threat', 'ict_index',
                    'own_goals', 'penalties_saved', 'penalties_missed', 'form',
                    'now_cost', 'selected_by_percent']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate price from now_cost (in tenths)
    if 'now_cost' in df.columns:
        df['price'] = df['now_cost'] / 10
    else:
        df['price'] = np.nan

    # Calculate ownership from selected_by_percent
    if 'selected_by_percent' in df.columns:
        df['ownership'] = df['selected_by_percent']
    else:
        df['ownership'] = np.nan

    # Calculate goal involvements
    df['goal_involvements'] = df['goals_scored'].fillna(0) + df['assists'].fillna(0)

    # Calculate average minutes per game (using current gameweek as games played proxy)
    try:
        from config import CURRENT_GAMEWEEK
        games_played = int(CURRENT_GAMEWEEK)
    except Exception:
        games_played = 1

    # Safer calculation: use starts to estimate games
    df['avg_mins_game'] = df.apply(
        lambda row: row['minutes'] / games_played if games_played > 0 and pd.notna(row['minutes']) else np.nan,
        axis=1
    )

    # Average minutes per start
    df['avg_mins_start'] = df.apply(
        lambda row: row['minutes'] / row['starts'] if pd.notna(row['starts']) and row['starts'] > 0 else np.nan,
        axis=1
    )

    # Start percentage
    df['start_pct'] = df.apply(
        lambda row: (row['starts'] / games_played * 100) if games_played > 0 and pd.notna(row['starts']) else np.nan,
        axis=1
    )

    # Per-90 calculations
    df['goals_per_90'] = df.apply(lambda row: calculate_per_90(row['goals_scored'], row['minutes'], min_minutes), axis=1)
    df['xg_per_90'] = df.apply(lambda row: calculate_per_90(row['expected_goals'], row['minutes'], min_minutes), axis=1)
    df['assists_per_90'] = df.apply(lambda row: calculate_per_90(row['assists'], row['minutes'], min_minutes), axis=1)
    df['xa_per_90'] = df.apply(lambda row: calculate_per_90(row['expected_assists'], row['minutes'], min_minutes), axis=1)
    df['gi_per_90'] = df.apply(lambda row: calculate_per_90(row['goal_involvements'], row['minutes'], min_minutes), axis=1)
    df['xgi_per_90'] = df.apply(lambda row: calculate_per_90(row['expected_goal_involvements'], row['minutes'], min_minutes), axis=1)

    # Regression metrics (actual - expected): positive = over-performing, negative = under-performing
    df['g_minus_xg'] = df['goals_scored'] - df['expected_goals']
    df['a_minus_xa'] = df['assists'] - df['expected_assists']
    df['gi_minus_xgi'] = df['goal_involvements'] - df['expected_goal_involvements']

    # Get FDR averages for each team
    try:
        _, _, fdr_avg = get_fixture_difficulty_grid(weeks=3)
        fdr_avg_dict = fdr_avg.to_dict()
    except Exception:
        fdr_avg_dict = {}

    # Add FDR average for next 3 GWs
    df['fdr_avg_3'] = df['team_name_abbrv'].apply(lambda x: get_team_fdr_avg(x, fdr_avg_dict))

    return df


def build_stats_display(df: pd.DataFrame, selected_columns: list):
    """
    Prepare display DataFrame and color column lists for the stats table.

    Returns (display_df, col_formats, positive_color_cols, negative_color_cols).
    """
    display_df = df[selected_columns].copy()

    # Rename columns to display names
    rename_map = {col: STAT_COLUMNS[col][0] for col in selected_columns if col in STAT_COLUMNS}
    display_df = display_df.rename(columns=rename_map)

    # Build format dict and color column lists
    col_formats = {}
    positive_cols = []
    negative_cols = []

    for col in selected_columns:
        if col not in STAT_COLUMNS:
            continue
        display_name, high_is_good, fmt, _ = STAT_COLUMNS[col]

        # Format spec
        if fmt == "text":
            pass
        elif fmt == "price":
            col_formats[display_name] = "£{:.1f}m"
        elif fmt == ",d":
            col_formats[display_name] = "{:,.0f}"
        elif fmt.startswith("+"):
            col_formats[display_name] = "{:" + fmt + "}"
        elif fmt.startswith("."):
            col_formats[display_name] = "{:" + fmt + "}"

        # Color direction
        if high_is_good is True:
            positive_cols.append(display_name)
        elif high_is_good is False:
            negative_cols.append(display_name)

    return display_df, col_formats, positive_cols, negative_cols


# Dark theme layout for Plotly charts
_DARK_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title=dict(font=dict(size=20, color="#ffffff"), x=0.5, xanchor="center"),
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=12)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=12)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=12)),
)


PRESET_DESCRIPTIONS = {
    "Essential": "Core stats: Points, Minutes, Goals, Assists, Clean Sheets",
    "Attacking": "Offensive output: Goals, Assists, xG, xA with per-90 rates",
    "Defensive": "Defensive stats: Clean Sheets, Goals Conceded, xGC, Saves",
    "Per 90": "Rate stats normalized to per-90 minutes for fair comparison",
    "ICT Focus": "FPL's Influence, Creativity, Threat index breakdown",
    "Fixture Focus": "Form + upcoming Fixture Difficulty Rating (lower = easier)",
    "GK Stats": "Goalkeeper-specific: Saves, Clean Sheets, Penalties Saved",
    "Regression": "G-xG, A-xA, GI-xGI: positive = over-performing (may regress), negative = under-performing (may improve)",
}


def display_advanced_stats_table(player_df: pd.DataFrame):
    """
    Display the advanced statistics table with filters and column selection.
    """
    st.subheader("Advanced Statistics Table")

    # Filters row
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        preset_options = ["Custom"] + list(COLUMN_PRESETS.keys())
        selected_preset = st.selectbox("Column Preset", preset_options, index=1)
        # Show preset description
        if selected_preset in PRESET_DESCRIPTIONS:
            st.caption(PRESET_DESCRIPTIONS[selected_preset])

    with col2:
        min_minutes = st.number_input("Min Minutes", min_value=0, max_value=3000, value=90, step=90)

    with col3:
        available_positions = player_df['position_abbrv'].unique().tolist()
        position_filter = st.multiselect(
            "Position Filter",
            options=available_positions,
            default=available_positions,
            key="adv_position_filter"
        )

    # Team filter in expander to keep UI clean
    available_teams = sorted(player_df['team_name'].unique().tolist())
    with st.expander("Filter by Team", expanded=False):
        team_filter = st.multiselect(
            "Select Teams",
            options=available_teams,
            default=available_teams,
            key="adv_team_filter",
            label_visibility="collapsed"
        )

    # Get default columns based on preset
    if selected_preset != "Custom":
        default_columns = COLUMN_PRESETS[selected_preset]
    else:
        default_columns = COLUMN_PRESETS["Essential"]

    # Column selection
    all_columns = list(STAT_COLUMNS.keys())
    selected_columns = st.multiselect(
        "Select Columns",
        options=all_columns,
        default=default_columns,
        format_func=lambda x: STAT_COLUMNS[x][0],
        key="adv_columns"
    )

    if not selected_columns:
        st.warning("Please select at least one column to display.")
        return

    # Prepare the data
    stats_df = prepare_advanced_stats_df(player_df, min_minutes)

    # Apply filters
    filtered_df = stats_df[
        (stats_df['position_abbrv'].isin(position_filter)) &
        (stats_df['team_name'].isin(team_filter))
    ]

    # Filter by minimum minutes
    filtered_df = filtered_df[filtered_df['minutes'] >= min_minutes]

    # Sort by total points by default
    filtered_df = filtered_df.sort_values('total_points', ascending=False)

    if filtered_df.empty:
        st.info("No players match the current filters.")
        return

    # Build display table and render with styled dark theme
    display_df, col_formats, positive_cols, negative_cols = build_stats_display(filtered_df, selected_columns)

    render_styled_table(
        display_df,
        col_formats=col_formats,
        positive_color_cols=positive_cols,
        negative_color_cols=negative_cols,
        max_height=600,
    )

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
        textfont=dict(color="#ffffff"),
        hovertemplate='<b>Player</b> = %{y}<br>' +
                      '<b>Team</b> = %{customdata[0]}<br>' +
                      '<b>Goals</b> = %{x}<extra></extra>',
        customdata=top_scorers[['team_name']].values,
        showlegend=False
    )

    # Reverse the order so the highest scorer is at the top
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        **_DARK_LAYOUT,
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
        textfont=dict(color="#ffffff"),
        hovertemplate='<b>Player</b> = %{y}<br>' +
                      '<b>Team</b> = %{customdata[0]}<br>' +
                      '<b>Assists</b> = %{x}<extra></extra>',
        customdata=top_assisters[['team_name']].values,
        showlegend=False
    )

    # Reverse the order so the highest scorer is at the top
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        **_DARK_LAYOUT,
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
        textfont=dict(color="#ffffff"),
        hovertemplate='<b>Player</b> = %{y}<br>' +
                      '<b>Team</b> = %{customdata[0]}<br>' +
                      '<b>Clean Sheets</b> = %{x}<extra></extra>',
        customdata=top_clean_sheets[['team_name']].values,
        showlegend=False
    )

    # Reverse the order so the highest scorer is at the top
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        **_DARK_LAYOUT,
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
        xaxis=dict(
            tickmode="linear",
            dtick=5,
            showgrid=True,
            zeroline=True,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
        ),
        **_DARK_LAYOUT,
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

    # Update layout with dark theme
    fig.update_layout(
        xaxis_title="Player Position",
        yaxis_title="Total Points",
        boxmode="group",
        **_DARK_LAYOUT,
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

    # Create tabs for different views
    tab_advanced, tab_charts = st.tabs(["Advanced Stats Table", "Visual Charts"])

    # Tab 1: Advanced Stats Table (main view)
    with tab_advanced:
        if not fpl_player_statistics.empty:
            display_advanced_stats_table(fpl_player_statistics)
        else:
            st.error("No data available at the URL provided.")

    # Tab 2: Visual Charts
    with tab_charts:
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
