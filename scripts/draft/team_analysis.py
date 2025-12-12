import config
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Imports from common modules
from scripts.common.api import (
    get_rotowire_player_projections,
    get_draft_team_composition_for_gameweek,
    get_draft_league_teams,
    get_league_details
)
from scripts.common.utils import (
    get_team_id_by_name,
    merge_fpl_players_and_projections,
    compute_team_record,
    compute_h2h_breakdown
)

WIN_COLOR = "#22c55e"  # green
DRAW_COLOR = "#ffffff"  # white
LOSS_COLOR = "#ef4444"  # red


def _h2h_stacked_bar(df: pd.DataFrame, title: str = "Head-to-Head (Win/Draw/Loss %)"):
    """
    Plotly horizontal stacked bars: Wins left (green), Draws middle (white), Losses right (red).
    Kept here as it is a specific visual for this page.
    """
    if df.empty:
        return go.Figure()
    y = df["Opponent"].tolist()
    win = df["Win%"].tolist()
    draw = df["Draw%"].tolist()
    loss = df["Loss%"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Win %", x=win, y=y, orientation="h", marker_color=WIN_COLOR))
    fig.add_trace(go.Bar(name="Draw %", x=draw, y=y, orientation="h", marker_color=DRAW_COLOR,
                         marker_line=dict(color="#ddd", width=1)))
    fig.add_trace(go.Bar(name="Loss %", x=loss, y=y, orientation="h", marker_color=LOSS_COLOR))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(title="Percentage", tickformat=".0%", range=[0, 1]),
        yaxis=dict(autorange="reversed"),
        legend_title=None,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    return fig


def show_team_projections(team_id, fpl_player_projections, gameweek):
    """Helper to merge team composition with projections."""
    team_composition_df = get_draft_team_composition_for_gameweek(team_id, gameweek)
    team_player_projections = merge_fpl_players_and_projections(team_composition_df, fpl_player_projections)

    if not team_player_projections.empty:
        cols_needed = ['Player', 'Team', 'Matchup', 'Position', 'Points', 'Pos Rank']
        cols = [c for c in cols_needed if c in team_player_projections.columns]
        team_player_projections = team_player_projections[cols]

    return team_player_projections


def show_team_stats_page():
    st.title("Team Analysis")
    st.write("Displaying detailed statistics and projections for selected team.")

    # 1. Fetch Core Data
    player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)
    team_dict = get_draft_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    league_details = get_league_details(config.FPL_DRAFT_LEAGUE_ID)

    # 2. Select Team
    team_list = list(team_dict.values())
    selected_team = st.selectbox("Select a Team", team_list)
    team_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, selected_team)

    # 3. Season Record
    w, d, l, pts = compute_team_record(team_id, league_details, up_to_gw=config.CURRENT_GAMEWEEK)
    st.subheader("Season Record")
    st.markdown(f"**{w}W – {d}D – {l}L = {pts} points**")

    # 4. Head-to-Head Analysis
    matches = league_details.get("matches", [])
    id_to_name = {e["id"]: e["entry_name"] for e in league_details.get("league_entries", [])}

    h2h_df = compute_h2h_breakdown(team_id, matches, id_to_name, up_to_gw=config.CURRENT_GAMEWEEK)
    st.subheader("Head-to-Head vs League Teams")
    st.plotly_chart(_h2h_stacked_bar(h2h_df), use_container_width=True)

    with st.expander("Show head-to-head table"):
        st.dataframe(h2h_df.assign(**{
            "Win%": (h2h_df["Win%"] * 100).round(1),
            "Draw%": (h2h_df["Draw%"] * 100).round(1),
            "Loss%": (h2h_df["Loss%"] * 100).round(1),
        }), use_container_width=True)

    # 5. Projections for Current Gameweek
    st.subheader(f"{selected_team} Projected Player Stats for Gameweek {config.CURRENT_GAMEWEEK}")
    st.dataframe(show_team_projections(team_id, player_projections, config.CURRENT_GAMEWEEK),
                 use_container_width=True, height=560)

    # 6. Team Composition Inspector
    st.subheader("Display Team Composition by Gameweek")
    gameweek = st.selectbox("Select Gameweek", list(range(1, config.CURRENT_GAMEWEEK + 1)))

    if st.button("Show Team Composition"):
        team_composition = get_draft_team_composition_for_gameweek(team_id, gameweek)
        st.subheader(f"Team Composition for {selected_team} in Gameweek {gameweek}")
        st.write(team_composition)