import config
import pandas as pd
import plotly.express as px
import streamlit as st
from scripts.common.utils import (
    get_league_player_ownership,
    get_league_teams,
    get_rotowire_player_projections,
    get_team_composition_for_gameweek,
    get_team_id_by_name,
    merge_fpl_players_and_projections,
    get_draft_all_h2h_records,
    get_draft_points_by_position,
    get_draft_team_players_with_points,
    get_classic_bootstrap_static,
)
from scripts.common.team_analysis_helpers import render_season_highlights


def show_team_projections(team_id, fpl_player_projections, gameweek):
    # Get the team composition for the team in the current gameweek
    team_composition_df = get_team_composition_for_gameweek(config.FPL_DRAFT_LEAGUE_ID, team_id, gameweek)

    # Merge the FPL team df with the fpl_player_projections
    team_player_projections = merge_fpl_players_and_projections(team_composition_df, fpl_player_projections)

    # Format columns
    team_player_projections = team_player_projections[['Player', 'Team', 'Matchup', 'Position', 'Points', 'Pos Rank']]

    # Return the df
    return(team_player_projections)


def show_team_stats_page():
    st.title("Team Analysis")
    st.write("Detailed statistics, projections, and season insights for your team.")

    # Pull FPL player projections from Rotowire
    player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    # Get FPL team names
    team_dict = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    team_list = list(team_dict.values())

    # Create a dropdown to select the team
    selected_team = st.selectbox("Select a Team", team_list)

    # Get the team ids based on the team names
    team_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, selected_team)

    # Get player data for the selected team (used across multiple sections)
    player_data_dict = get_draft_team_players_with_points(config.FPL_DRAFT_LEAGUE_ID)
    team_players = player_data_dict.get(selected_team, [])

    # Get bootstrap data for detailed stats
    bootstrap = get_classic_bootstrap_static()

    st.divider()

    # ---------------------------
    # SEASON HIGHLIGHTS (Top Row)
    # ---------------------------
    st.header("Season Highlights")

    render_season_highlights(team_players, bootstrap_data=bootstrap, is_classic=False)

    st.divider()

    # ---------------------------
    # PROJECTED STATS (Current GW)
    # ---------------------------
    st.header(f"Gameweek {config.CURRENT_GAMEWEEK} Projections")

    st.dataframe(
        show_team_projections(team_id, player_projections, config.CURRENT_GAMEWEEK),
        use_container_width=True,
        height=400
    )

    st.divider()

    # ---------------------------
    # HEAD-TO-HEAD RECORDS
    # ---------------------------
    st.header("Head-to-Head Records")

    h2h_records = get_draft_all_h2h_records(config.FPL_DRAFT_LEAGUE_ID, team_id)

    if h2h_records:
        # Calculate overall record
        total_wins = sum(r["wins"] for r in h2h_records)
        total_draws = sum(r["draws"] for r in h2h_records)
        total_losses = sum(r["losses"] for r in h2h_records)
        total_pf = sum(r["points_for"] for r in h2h_records)
        total_pa = sum(r["points_against"] for r in h2h_records)

        # Display overall record
        st.markdown(f"**Overall Record: {total_wins}-{total_draws}-{total_losses}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Wins", total_wins)
        with col2:
            st.metric("Draws", total_draws)
        with col3:
            st.metric("Losses", total_losses)
        with col4:
            st.metric("Points For", total_pf)
        with col5:
            st.metric("Points Against", total_pa)

        st.markdown("")  # Spacer

        # Display H2H record against each opponent
        h2h_df = pd.DataFrame([
            {
                "Opponent": r["opponent_name"],
                "W": r["wins"],
                "D": r["draws"],
                "L": r["losses"],
                "Record": r["record_str"],
                "PF": r["points_for"],
                "PA": r["points_against"],
                "Diff": r["points_for"] - r["points_against"]
            }
            for r in h2h_records
        ])

        st.dataframe(
            h2h_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Opponent": st.column_config.TextColumn("Opponent"),
                "W": st.column_config.NumberColumn("W", width="small"),
                "D": st.column_config.NumberColumn("D", width="small"),
                "L": st.column_config.NumberColumn("L", width="small"),
                "Record": st.column_config.TextColumn("Record"),
                "PF": st.column_config.NumberColumn("PF", help="Points For"),
                "PA": st.column_config.NumberColumn("PA", help="Points Against"),
                "Diff": st.column_config.NumberColumn("+/-", help="Point Differential"),
            }
        )
    else:
        st.info("No head-to-head data available yet. The season may not have started.")

    st.divider()

    # ---------------------------
    # POINTS BY POSITION
    # ---------------------------
    st.header("Points by Position")

    POSITION_COLORS = {
        "GK": "#f39c12",
        "DEF": "#3498db",
        "MID": "#2ecc71",
        "FWD": "#e74c3c",
    }

    pos_df = get_draft_points_by_position(config.FPL_DRAFT_LEAGUE_ID)

    team_row = pos_df[pos_df["Team"] == selected_team] if not pos_df.empty else pd.DataFrame()

    if not team_row.empty:
        row = team_row.iloc[0]
        pos_cols = ["GK", "DEF", "MID", "FWD"]
        total = row["Total"]

        # Pie chart + metrics side by side
        col_pie, col_metrics = st.columns([1, 1])

        with col_pie:
            pie_data = pd.DataFrame({
                "Position": pos_cols,
                "Points": [row[c] for c in pos_cols]
            })
            fig = px.pie(
                pie_data,
                values="Points",
                names="Position",
                color="Position",
                color_discrete_map=POSITION_COLORS,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_metrics:
            for pos in pos_cols:
                pts = int(row[pos])
                pct = f"{pts / total * 100:.1f}%" if total > 0 else "0%"
                st.metric(pos, f"{pts} pts ({pct})")

        # Player detail table
        if team_players:
            st.markdown("**Player Breakdown**")
            players_df = pd.DataFrame(team_players)
            players_df.columns = ["Player", "Position", "Total Points", "Team"]

            # Sort by position order then points desc
            pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
            players_df["_pos_order"] = players_df["Position"].map(pos_order)
            players_df = players_df.sort_values(
                ["_pos_order", "Total Points"], ascending=[True, False]
            ).drop(columns=["_pos_order"])

            st.dataframe(
                players_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player"),
                    "Position": st.column_config.TextColumn("Pos"),
                    "Total Points": st.column_config.NumberColumn("Points"),
                    "Team": st.column_config.TextColumn("Team"),
                }
            )
    else:
        st.info("No position data available for this team.")

    st.divider()

    # ---------------------------
    # HISTORICAL TEAM COMPOSITION
    # ---------------------------
    with st.expander("View Team Composition by Gameweek"):
        # Create dropdown to select Gameweek
        current_gameweek = config.CURRENT_GAMEWEEK
        gameweek = st.selectbox("Select Gameweek", list(range(1, current_gameweek + 1)))

        # Get and display the team composition for the selected gameweek
        if st.button("Show Team Composition"):
            # Get the team composition
            team_composition = get_team_composition_for_gameweek(config.FPL_DRAFT_LEAGUE_ID, team_id, gameweek)

            # Display the team's players for selected gameweek
            st.subheader(f"Team Composition for {selected_team} in Gameweek {gameweek}")
            st.write(team_composition)
