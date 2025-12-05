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

def gk_xgc_vs_saves_chart(players_df: pd.DataFrame, min_minutes: int = 180):
    df = players_df.copy()

    # flexible column resolver
    rename_map = {c: c.strip().lower() for c in df.columns}
    inv = {v: k for k, v in rename_map.items()}
    def col(*cands):
        for c in cands:
            if c in inv: return inv[c]
        return None

    pos_col   = col("position", "position_abbrv", "pos")
    min_col   = col("minutes", "mins")
    xgc_col   = col("expected_goals_conceded", "xgc")
    saves_col = col("saves")
    name_col  = col("player", "web_name")
    team_col  = col("team", "team_name_abbrv", "team_name", "team_short")

    # keep only GKs
    df = df[df[pos_col].isin(["G", "GK", "GKP"])].copy()
    print(df)

    # numeric coercions
    for c in (min_col, xgc_col, saves_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop null/blank names
    df[name_col] = df[name_col].astype(str).str.strip()
    df = df[df[name_col].ne("") & df[name_col].ne("None") & df[name_col].notna()]

    # minutes filter & per-90s
    df = df[(df[min_col] >= min_minutes) & (df[min_col] > 0)].copy()
    df["xGC/90"]   = (df[xgc_col] / df[min_col]) * 90
    df["Saves/90"] = (df[saves_col] / df[min_col]) * 90

    # thresholds (medians)
    x_thr = float(df["xGC/90"].median())
    s_thr = float(df["Saves/90"].median())

    # quadrant buckets
    def quad(r):
        if r["xGC/90"] <= x_thr and r["Saves/90"] >= s_thr: return "⭐ Bottom-Right (ideal)"
        if r["xGC/90"] >= x_thr and r["Saves/90"] <= s_thr: return "⚠ Top-Left (worst)"
        if r["xGC/90"] >= x_thr and r["Saves/90"] >= s_thr: return "🧤 Top-Right (high xG + high saves)"
        return "👍 Bottom-Left (low xG + low saves)"
    df["Quadrant"] = df.apply(quad, axis=1)

    # build figure (uniform, larger bubbles; add name labels)
    fig = px.scatter(
        df,
        x="xGC/90",
        y="Saves/90",
        color="Quadrant",
        title="Goalkeepers: Saves vs xG Conceded per 90",
        hover_name=df[name_col],
        hover_data={team_col: True, "xGC/90":":.2f", "Saves/90":":.2f", "Quadrant": True},
        template="plotly_white",
        text=df[name_col],  # <-- show names on chart
    )

    # bigger uniform markers + readable labels
    fig.update_traces(
        mode="markers+text",
        marker=dict(size=18, opacity=0.85, line=dict(width=0.5, color="white")),
        textposition="top center",
        textfont=dict(size=10),
        selector=dict(mode="markers+text")
    )
    fig.update_layout(hovermode="closest", legend_title="Quadrant", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(title="xG Conceded / 90 (lower is better)", zeroline=False)
    fig.update_yaxes(title="Saves / 90 (higher is better)", zeroline=False)

    # quadrant lines + annotations
    fig.add_hline(y=s_thr, line_dash="dash", line_color="#999")
    fig.add_vline(x=x_thr, line_dash="dash", line_color="#999")
    fig.add_annotation(x=x_thr*0.5, y=s_thr*0.5, text="👍 Low xG / Low Saves", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=x_thr*1.5, y=s_thr*1.5, text="🧤 High xG / High Saves", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=x_thr*0.5, y=s_thr*1.5, text="⭐ Low xG / High Saves", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=x_thr*1.5, y=s_thr*0.5, text="⚠ High xG / Low Saves", showarrow=False, font=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)

def def_xgi_vs_xgc_chart(players_df: pd.DataFrame, min_minutes: int = 180):
    df = players_df.copy()

    # Flexible column resolver (case-insensitive)
    low = {c.lower(): c for c in df.columns}
    def col(*cands):
        for c in cands:
            if c in low: return low[c]
        return None

    name_col  = col("player", "web_name")
    pos_col   = col("position", "position_abbrv", "pos")
    mins_col  = col("minutes", "mins")
    xg_col    = col("expected_goals", "xg")
    xa_col    = col("expected_assists", "xa")
    xgc_col   = col("expected_goals_conceded", "xgc")

    # Defensive stats (optional, many data sources won't have all)
    tac_col   = col("tackles")
    int_col   = col("interceptions")
    clr_col   = col("clearances")
    blk_col   = col("blocks")
    cs_col    = col("clean_sheets")  # fallback signal

    # Keep only midfielders; basic cleaning
    if not (name_col and pos_col and mins_col and xg_col and xa_col and xgc_col):
        st.warning("Missing required columns for midfielder chart.")
        return

    df = df[df[pos_col].isin(["D", "DEF"])].copy()
    for c in [mins_col, xg_col, xa_col, xgc_col, tac_col, int_col, clr_col, blk_col, cs_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop null/blank names
    df[name_col] = df[name_col].astype(str).str.strip()
    df = df[df[name_col].ne("") & df[name_col].ne("None") & df[name_col].notna()]

    # Minutes & per-90s
    df = df[(df[mins_col] >= min_minutes) & (df[mins_col] > 0)].copy()
    df["xGI/90"] = ((df[xg_col] + df[xa_col]) / df[mins_col]) * 90
    df["xGC/90"] = (df[xgc_col] / df[mins_col]) * 90

    # Defensive contribution per 90 (primary: tackles+interceptions+clearances+blocks)
    if any(c is None for c in [tac_col, int_col, clr_col, blk_col]) or \
       any(k not in df.columns for k in [tac_col, int_col, clr_col, blk_col]):
        # Fallback: clean sheets per 90 (coarse but available)
        if cs_col and cs_col in df.columns:
            df["Def/90"] = (df[cs_col] / df[mins_col]) * 90
        else:
            # final fallback: zeros to avoid crash
            df["Def/90"] = 0.0
    else:
        df["Def/90"] = (
            df[tac_col].fillna(0) + df[int_col].fillna(0) +
            df[clr_col].fillna(0) + df[blk_col].fillna(0)
        ) / df[mins_col] * 90

    # Bucket into High / Medium / Low by tertiles
    q_low, q_high = df["Def/90"].quantile([0.33, 0.67]).tolist()
    def bucket(v):
        if v <= q_low:  return "Low defensive contribution"
        if v >= q_high: return "High defensive contribution"
        return "Medium defensive contribution"
    df["Def Bucket"] = df["Def/90"].apply(bucket)

    # Filter by Top-25 in Total Points
    top_df = df.nlargest(25, 'total_points')

    # Build scatter
    fig = px.scatter(
        top_df,
        x="xGI/90",
        y="xGC/90",
        color="Def Bucket",
        size=mins_col,
        size_max=18,
        hover_name=top_df[name_col],
        hover_data={
            "xGI/90":":.2f",
            "xGC/90":":.2f",
            "Def/90":":.2f",
            mins_col: True,
            "Def Bucket": True
        },
        template="plotly_white",
        text=top_df[name_col],   # show names on points
    )

    # improve readability
    fig.update_traces(
        mode="markers+text",
        marker=dict(opacity=0.85, line=dict(width=0.5, color="white")),
        textposition="top center",
        textfont=dict(size=9)
    )
    fig.update_layout(
        title="Top 25 Defenders: xGI per 90 vs xGC per 90",
        legend_title="Defensive contribution",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig.update_xaxes(title="xGI / 90 (xG + xA)", zeroline=False)
    fig.update_yaxes(title="xGC / 90 (lower is better)", zeroline=False)

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

    if config.ROTOWIRE_URL:
        # Pull FPL player projections from Rotowire (if available), for player name cleaning and player price
        fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

        # Clean FPL player names
        fpl_player_statistics = clean_fpl_player_names(fpl_player_statistics, fpl_player_projections)

        # Merge in player price
        fpl_player_statistics = pd.merge(fpl_player_statistics, fpl_player_projections[['Player', 'Price']],
                                          left_on='player', right_on='Player', how='left').rename(columns={'Price':'price'}).drop(columns=['Player'], axis=1)

    # Print out FPL player stats, if desired
    print("FPL Player Statistics: \n", fpl_player_statistics)
    print("Columns: \n", fpl_player_statistics.columns)

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

        # 2nd Row - Expected vs. Actual Goals and Box Plot
        col4, col5 = st.columns(2)
        with col4:
            display_expected_vs_actual_goals(fpl_player_statistics, position_filter, team_filter, top_n)
        with col5:
            display_boxplot_point_distribution(fpl_player_statistics, position_filter, team_filter)

        # 3rd Row - Goalkeepers Expected Goals vs. Saves
        gk_xgc_vs_saves_chart(fpl_player_statistics, min_minutes=180)

        # Bottom row - Defenders
        def_xgi_vs_xgc_chart(fpl_player_statistics, min_minutes=180)

    else:
        st.error("No data available at the URL provided.")
