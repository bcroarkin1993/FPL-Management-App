import config
import pandas as pd
import streamlit as st
from scripts.common.utils import get_league_player_ownership, get_league_teams, get_rotowire_player_projections, \
    get_team_composition_for_gameweek, get_team_id_by_name, merge_fpl_players_and_projections, \
    get_draft_all_h2h_records

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
    st.write("Displaying detailed statistics and projections for selected team.")

    # Pull FPL player projections from Rotowire
    player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    # Get FPL team names
    team_dict = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)
    team_list = list(team_dict.values())

    # Create a dropdown to select the team
    selected_team = st.selectbox("Select a Team", team_list)

    # Get the team ids based on the team names
    team_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, selected_team)

    # Display the team's player projected stats
    st.subheader(f"{selected_team} Projected Player Stats for Gameweek {config.CURRENT_GAMEWEEK}")
    st.dataframe(show_team_projections(team_id, player_projections, config.CURRENT_GAMEWEEK),
                 use_container_width=True, height=560)

    st.divider()

    # ---------------------------
    # HEAD-TO-HEAD RECORDS
    # ---------------------------
    st.subheader("Head-to-Head Records")

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

    # Display the team's players by gameweek
    st.subheader("Display Team Composition by Gameweek")

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

