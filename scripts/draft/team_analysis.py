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
    get_draft_league_details,
)
from scripts.common.team_analysis_helpers import render_season_highlights
from scripts.common.styled_tables import render_styled_table


def _stat_card(label: str, value: str, accent: str = "#00ff87") -> str:
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'</div>'
    )


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
    # LEAGUE STANDING
    # ---------------------------
    league_details = get_draft_league_details(config.FPL_DRAFT_LEAGUE_ID)
    if league_details:
        standings = league_details.get("standings", [])
        league_entries = league_details.get("league_entries", [])

        # Build entry_id → id mapping (get_team_id_by_name returns entry_id,
        # but standings use league_entry which maps to id in league_entries)
        entry_id_to_id = {e["entry_id"]: e["id"] for e in league_entries if "entry_id" in e and "id" in e}
        league_entry_id = entry_id_to_id.get(team_id)

        # Find this team's standing
        team_standing = None
        if league_entry_id is not None:
            for s in standings:
                if s.get("league_entry") == league_entry_id:
                    team_standing = s
                    break

        if team_standing:
            st.header("League Standing")

            total_teams = len(standings)
            rank = team_standing.get("rank", "?")
            wins = team_standing.get("matches_won", 0)
            draws = team_standing.get("matches_drawn", 0)
            losses = team_standing.get("matches_lost", 0)
            points_for = team_standing.get("points_for", 0)
            points_against = team_standing.get("points_against", 0)
            league_points = team_standing.get("total", 0)

            def _standing_card(label: str, value: str) -> str:
                return (
                    f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
                    f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
                    f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;'
                    f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
                    f'<div style="color:#00ff87;font-size:22px;font-weight:700;">{value}</div>'
                    f'</div>'
                )

            cols = st.columns(4)
            with cols[0]:
                st.markdown(_standing_card("League Position", f"{rank} / {total_teams}"), unsafe_allow_html=True)
            with cols[1]:
                st.markdown(_standing_card("Record", f"{wins}W - {draws}D - {losses}L"), unsafe_allow_html=True)
            with cols[2]:
                st.markdown(_standing_card("Points For / Against", f"{points_for} / {points_against}"), unsafe_allow_html=True)
            with cols[3]:
                st.markdown(_standing_card("League Points", str(league_points)), unsafe_allow_html=True)

            st.divider()

    # ---------------------------
    # PROJECTED STATS (Current GW)
    # ---------------------------
    st.header(f"Gameweek {config.CURRENT_GAMEWEEK} Projections")

    proj_df = show_team_projections(team_id, player_projections, config.CURRENT_GAMEWEEK)
    if not proj_df.empty:
        _pos_sort = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        proj_df["_pos_order"] = proj_df["Position"].map(_pos_sort)
        proj_df["_pts"] = pd.to_numeric(proj_df["Points"], errors="coerce").fillna(0)
        proj_df = proj_df.sort_values(["_pos_order", "_pts"], ascending=[True, False]).drop(columns=["_pos_order", "_pts"])
    render_styled_table(
        proj_df,
        col_formats={"Points": "{:.1f}"},
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
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(_stat_card("Wins", str(total_wins)), unsafe_allow_html=True)
        with col2:
            st.markdown(_stat_card("Draws", str(total_draws), accent="#9ca3af"), unsafe_allow_html=True)
        with col3:
            st.markdown(_stat_card("Losses", str(total_losses), accent="#f87171"), unsafe_allow_html=True)
        with col4:
            st.markdown(_stat_card("Points For", str(total_pf)), unsafe_allow_html=True)
        with col5:
            st.markdown(_stat_card("Points Against", str(total_pa), accent="#f87171"), unsafe_allow_html=True)

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

        h2h_df = h2h_df.rename(columns={"Diff": "+/-"})
        render_styled_table(
            h2h_df,
            text_align={"W": "center", "D": "center", "L": "center", "Record": "center"},
            positive_color_cols=["+/-"],
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
                paper_bgcolor="#1a1a2e",
                font=dict(color="#ffffff"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_metrics:
            for pos in pos_cols:
                pts = int(row[pos])
                pct = f"{pts / total * 100:.1f}%" if total > 0 else "0%"
                st.markdown(_stat_card(pos, f"{pts} pts ({pct})", accent=POSITION_COLORS.get(pos, "#00ff87")), unsafe_allow_html=True)

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

            render_styled_table(
                players_df,
                positive_color_cols=["Total Points"],
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
