"""
Draft FPL - League Analysis Page

Provides advanced league-wide analytics including:
- Head-to-head records matrix
- Scoring distribution and consistency
- Weekly performance trends
- Strength of schedule analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from typing import Optional, Dict, List, Tuple

import config
from scripts.common.error_helpers import get_logger, show_api_error
from scripts.common.utils import get_current_gameweek, get_draft_points_by_position
from scripts.common.styled_tables import render_styled_table

_logger = get_logger("fpl_app.draft.league_analysis")


# ---------------------------
# DATA FETCHING
# ---------------------------

@st.cache_data(ttl=600)
def fetch_league_data(league_id: int) -> Optional[dict]:
    """Fetch full league details including matches and entries."""
    try:
        url = f"https://draft.premierleague.com/api/league/{league_id}/details"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        _logger.warning("Failed to fetch league data for league %s", league_id, exc_info=True)
    return None


def get_team_names(league_data: dict) -> Dict[int, str]:
    """Extract team ID to name mapping."""
    entries = league_data.get("league_entries", [])
    return {entry["id"]: entry["entry_name"] for entry in entries}


def get_matches_df(league_data: dict, team_names: Dict[int, str]) -> pd.DataFrame:
    """Convert matches to DataFrame with team names."""
    matches = league_data.get("matches", [])
    if not matches:
        return pd.DataFrame()

    rows = []
    for match in matches:
        gw = match.get("event")
        team1_id = match.get("league_entry_1")
        team2_id = match.get("league_entry_2")
        team1_pts = match.get("league_entry_1_points", 0)
        team2_pts = match.get("league_entry_2_points", 0)

        # Skip unplayed matches
        if team1_pts == 0 and team2_pts == 0:
            continue

        rows.append({
            "Gameweek": gw,
            "Team1_ID": team1_id,
            "Team1": team_names.get(team1_id, f"Team {team1_id}"),
            "Team1_Pts": team1_pts,
            "Team2_ID": team2_id,
            "Team2": team_names.get(team2_id, f"Team {team2_id}"),
            "Team2_Pts": team2_pts,
        })

    return pd.DataFrame(rows)


# ---------------------------
# HEAD-TO-HEAD ANALYSIS
# ---------------------------

def build_h2h_matrix(matches_df: pd.DataFrame, team_names: Dict[int, str]) -> pd.DataFrame:
    """
    Build head-to-head record matrix.
    Cell (i, j) shows Team i's record vs Team j as "W-D-L".
    """
    if matches_df.empty:
        return pd.DataFrame()

    teams = sorted(set(matches_df["Team1"].tolist() + matches_df["Team2"].tolist()))

    # Initialize records dict
    records = {team: {opp: {"W": 0, "D": 0, "L": 0} for opp in teams} for team in teams}

    for _, row in matches_df.iterrows():
        team1, team2 = row["Team1"], row["Team2"]
        pts1, pts2 = row["Team1_Pts"], row["Team2_Pts"]

        if pts1 > pts2:
            records[team1][team2]["W"] += 1
            records[team2][team1]["L"] += 1
        elif pts1 < pts2:
            records[team1][team2]["L"] += 1
            records[team2][team1]["W"] += 1
        else:
            records[team1][team2]["D"] += 1
            records[team2][team1]["D"] += 1

    # Build matrix
    matrix_data = []
    for team in teams:
        row = {"Team": team}
        for opp in teams:
            if team == opp:
                row[opp] = "-"
            else:
                r = records[team][opp]
                row[opp] = f"{r['W']}-{r['D']}-{r['L']}"
        matrix_data.append(row)

    return pd.DataFrame(matrix_data).set_index("Team")


def build_h2h_points_matrix(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build head-to-head total points matrix.
    Cell (i, j) shows total points Team i scored against Team j.
    """
    if matches_df.empty:
        return pd.DataFrame()

    teams = sorted(set(matches_df["Team1"].tolist() + matches_df["Team2"].tolist()))

    # Initialize points dict
    points = {team: {opp: 0 for opp in teams} for team in teams}

    for _, row in matches_df.iterrows():
        team1, team2 = row["Team1"], row["Team2"]
        pts1, pts2 = row["Team1_Pts"], row["Team2_Pts"]

        points[team1][team2] += pts1
        points[team2][team1] += pts2

    # Build matrix
    matrix_data = []
    for team in teams:
        row = {"Team": team}
        for opp in teams:
            if team == opp:
                row[opp] = "-"
            else:
                row[opp] = points[team][opp]
        matrix_data.append(row)

    return pd.DataFrame(matrix_data).set_index("Team")


# ---------------------------
# SCORING ANALYSIS
# ---------------------------

def get_team_weekly_scores(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Get each team's score for each gameweek."""
    if matches_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in matches_df.iterrows():
        rows.append({
            "Team": row["Team1"],
            "Gameweek": row["Gameweek"],
            "Points": row["Team1_Pts"],
            "Opponent": row["Team2"],
            "Opp_Points": row["Team2_Pts"],
        })
        rows.append({
            "Team": row["Team2"],
            "Gameweek": row["Gameweek"],
            "Points": row["Team2_Pts"],
            "Opponent": row["Team1"],
            "Opp_Points": row["Team1_Pts"],
        })

    return pd.DataFrame(rows).sort_values(["Team", "Gameweek"])


def calculate_scoring_stats(weekly_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate scoring statistics for each team."""
    if weekly_scores.empty:
        return pd.DataFrame()

    stats = weekly_scores.groupby("Team").agg(
        Games=("Points", "count"),
        Total_Pts=("Points", "sum"),
        Avg_Pts=("Points", "mean"),
        Std_Dev=("Points", "std"),
        Min_Pts=("Points", "min"),
        Max_Pts=("Points", "max"),
        Pts_Against=("Opp_Points", "sum"),
        Avg_Against=("Opp_Points", "mean"),
    ).reset_index()

    # Calculate consistency score (lower std dev = more consistent)
    stats["Consistency"] = stats["Avg_Pts"] / (stats["Std_Dev"] + 1)

    # Round values
    for col in ["Avg_Pts", "Std_Dev", "Avg_Against", "Consistency"]:
        stats[col] = stats[col].round(2)

    return stats.sort_values("Avg_Pts", ascending=False)


def plot_scoring_distribution(weekly_scores: pd.DataFrame) -> Optional[go.Figure]:
    """Create box plot of scoring distribution by team."""
    if weekly_scores.empty:
        return None

    # Sort teams by median score
    team_order = weekly_scores.groupby("Team")["Points"].median().sort_values(ascending=False).index.tolist()

    fig = px.box(
        weekly_scores,
        x="Team",
        y="Points",
        color="Team",
        title="Scoring Distribution by Team",
        labels={"Points": "Gameweek Points", "Team": ""},
        category_orders={"Team": team_order}
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        yaxis_title="Points",
        hovermode="x unified"
    )

    return fig


def plot_points_vs_against(scoring_stats: pd.DataFrame) -> Optional[go.Figure]:
    """Create scatter plot of points for vs points against."""
    if scoring_stats.empty:
        return None

    fig = px.scatter(
        scoring_stats,
        x="Total_Pts",
        y="Pts_Against",
        text="Team",
        title="Points Scored vs Points Conceded",
        labels={
            "Total_Pts": "Total Points For",
            "Pts_Against": "Total Points Against"
        }
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(size=12)
    )

    # Add quadrant lines at averages
    avg_for = scoring_stats["Total_Pts"].mean()
    avg_against = scoring_stats["Pts_Against"].mean()

    fig.add_hline(y=avg_against, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=avg_for, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    fig.add_annotation(
        x=scoring_stats["Total_Pts"].max(), y=scoring_stats["Pts_Against"].min(),
        text="High scoring, good defense", showarrow=False,
        font=dict(size=10, color="green"), opacity=0.7
    )
    fig.add_annotation(
        x=scoring_stats["Total_Pts"].min(), y=scoring_stats["Pts_Against"].max(),
        text="Low scoring, poor defense", showarrow=False,
        font=dict(size=10, color="red"), opacity=0.7
    )

    fig.update_layout(hovermode="closest")

    return fig


# ---------------------------
# WEEKLY TRENDS
# ---------------------------

def plot_weekly_rank_trends(weekly_scores: pd.DataFrame) -> Optional[go.Figure]:
    """Plot weekly rank trends for all teams."""
    if weekly_scores.empty:
        return None

    # Calculate weekly rank
    weekly_scores = weekly_scores.copy()
    weekly_scores["GW_Rank"] = weekly_scores.groupby("Gameweek")["Points"].rank(
        ascending=False, method="min"
    ).astype(int)

    fig = px.line(
        weekly_scores,
        x="Gameweek",
        y="GW_Rank",
        color="Team",
        title="Weekly Rank Trends",
        labels={"GW_Rank": "Gameweek Rank", "Gameweek": "Gameweek"}
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        yaxis_title="Rank (1 = Best)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    return fig


def plot_cumulative_points(weekly_scores: pd.DataFrame) -> Optional[go.Figure]:
    """Plot cumulative points over time."""
    if weekly_scores.empty:
        return None

    # Calculate cumulative points
    weekly_scores = weekly_scores.copy().sort_values(["Team", "Gameweek"])
    weekly_scores["Cumulative_Pts"] = weekly_scores.groupby("Team")["Points"].cumsum()

    fig = px.line(
        weekly_scores,
        x="Gameweek",
        y="Cumulative_Pts",
        color="Team",
        title="Cumulative Points Over Season",
        labels={"Cumulative_Pts": "Total Points", "Gameweek": "Gameweek"}
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    return fig


# ---------------------------
# STRENGTH OF SCHEDULE
# ---------------------------

def calculate_strength_of_schedule(weekly_scores: pd.DataFrame, scoring_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate strength of schedule for each team.
    Based on average points of opponents faced.
    """
    if weekly_scores.empty or scoring_stats.empty:
        return pd.DataFrame()

    # Get opponent average scores
    opp_avg = scoring_stats.set_index("Team")["Avg_Pts"].to_dict()

    # Calculate SOS for each team
    sos_data = []
    for team in weekly_scores["Team"].unique():
        team_matches = weekly_scores[weekly_scores["Team"] == team]
        opponents = team_matches["Opponent"].tolist()

        # Average of opponents' season average
        opp_strengths = [opp_avg.get(opp, 0) for opp in opponents]
        avg_opp_strength = sum(opp_strengths) / len(opp_strengths) if opp_strengths else 0

        # Actual points scored by opponents against this team
        avg_opp_actual = team_matches["Opp_Points"].mean()

        sos_data.append({
            "Team": team,
            "Opp_Avg_Season": round(avg_opp_strength, 2),
            "Opp_Avg_vs_Team": round(avg_opp_actual, 2),
            "Difficulty_Delta": round(avg_opp_actual - avg_opp_strength, 2),
        })

    sos_df = pd.DataFrame(sos_data)

    # Rank by opponent strength (higher = harder schedule)
    sos_df["SOS_Rank"] = sos_df["Opp_Avg_Season"].rank(ascending=False).astype(int)

    return sos_df.sort_values("SOS_Rank")


# ---------------------------
# RECORDS & STREAKS
# ---------------------------

def calculate_records(matches_df: pd.DataFrame, weekly_scores: pd.DataFrame) -> dict:
    """Calculate various league records."""
    if matches_df.empty or weekly_scores.empty:
        return {}

    records = {}

    # Highest single gameweek score
    max_idx = weekly_scores["Points"].idxmax()
    max_row = weekly_scores.loc[max_idx]
    records["highest_score"] = {
        "team": max_row["Team"],
        "points": max_row["Points"],
        "gameweek": max_row["Gameweek"],
        "opponent": max_row["Opponent"]
    }

    # Lowest single gameweek score
    min_idx = weekly_scores["Points"].idxmin()
    min_row = weekly_scores.loc[min_idx]
    records["lowest_score"] = {
        "team": min_row["Team"],
        "points": min_row["Points"],
        "gameweek": min_row["Gameweek"],
        "opponent": min_row["Opponent"]
    }

    # Biggest win margin
    matches_df = matches_df.copy()
    matches_df["Margin"] = abs(matches_df["Team1_Pts"] - matches_df["Team2_Pts"])
    max_margin_idx = matches_df["Margin"].idxmax()
    max_margin = matches_df.loc[max_margin_idx]

    if max_margin["Team1_Pts"] > max_margin["Team2_Pts"]:
        winner, loser = max_margin["Team1"], max_margin["Team2"]
        w_pts, l_pts = max_margin["Team1_Pts"], max_margin["Team2_Pts"]
    else:
        winner, loser = max_margin["Team2"], max_margin["Team1"]
        w_pts, l_pts = max_margin["Team2_Pts"], max_margin["Team1_Pts"]

    records["biggest_win"] = {
        "winner": winner,
        "loser": loser,
        "score": f"{w_pts}-{l_pts}",
        "margin": max_margin["Margin"],
        "gameweek": max_margin["Gameweek"]
    }

    # Closest match
    min_margin_idx = matches_df[matches_df["Margin"] > 0]["Margin"].idxmin()
    min_margin = matches_df.loc[min_margin_idx]
    records["closest_match"] = {
        "team1": min_margin["Team1"],
        "team2": min_margin["Team2"],
        "score": f"{min_margin['Team1_Pts']}-{min_margin['Team2_Pts']}",
        "margin": min_margin["Margin"],
        "gameweek": min_margin["Gameweek"]
    }

    return records


def calculate_streaks(weekly_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate current and longest win/loss streaks."""
    if weekly_scores.empty:
        return pd.DataFrame()

    streak_data = []

    for team in weekly_scores["Team"].unique():
        team_df = weekly_scores[weekly_scores["Team"] == team].sort_values("Gameweek")

        # Determine W/L/D for each game
        results = []
        for _, row in team_df.iterrows():
            if row["Points"] > row["Opp_Points"]:
                results.append("W")
            elif row["Points"] < row["Opp_Points"]:
                results.append("L")
            else:
                results.append("D")

        # Calculate streaks
        current_streak = 1
        current_type = results[-1] if results else None
        longest_win = 0
        longest_loss = 0

        # Current streak
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current_type:
                current_streak += 1
            else:
                break

        # Longest streaks
        streak = 1
        for i in range(1, len(results)):
            if results[i] == results[i-1]:
                streak += 1
            else:
                if results[i-1] == "W":
                    longest_win = max(longest_win, streak)
                elif results[i-1] == "L":
                    longest_loss = max(longest_loss, streak)
                streak = 1

        # Check final streak
        if results:
            if results[-1] == "W":
                longest_win = max(longest_win, streak)
            elif results[-1] == "L":
                longest_loss = max(longest_loss, streak)

        streak_data.append({
            "Team": team,
            "Current_Streak": f"{current_streak}{current_type}" if current_type else "-",
            "Longest_Win_Streak": longest_win,
            "Longest_Loss_Streak": longest_loss,
        })

    return pd.DataFrame(streak_data).sort_values("Longest_Win_Streak", ascending=False)


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_draft_league_analysis_page():
    """Display the Draft League Analysis page."""

    st.title("Draft League Analysis")

    league_id = config.FPL_DRAFT_LEAGUE_ID
    if not league_id:
        st.warning("No Draft league configured. Add `FPL_DRAFT_LEAGUE_ID` to your `.env` file.")
        return

    # Fetch data
    with st.spinner("Loading league data..."):
        league_data = fetch_league_data(league_id)

    if not league_data:
        show_api_error("loading league data", hint_key="league_id")
        return

    team_names = get_team_names(league_data)
    matches_df = get_matches_df(league_data, team_names)

    if matches_df.empty:
        st.info("No match data available yet. The season may not have started.")
        return

    weekly_scores = get_team_weekly_scores(matches_df)
    scoring_stats = calculate_scoring_stats(weekly_scores)

    # League name header
    league_name = league_data.get("league", {}).get("name", "Draft League")
    st.markdown(f"### {league_name}")

    current_gw = get_current_gameweek()
    if current_gw:
        st.caption(f"Analysis through Gameweek {current_gw - 1}")

    st.divider()

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Head-to-Head",
        "Scoring Analysis",
        "Weekly Trends",
        "Strength of Schedule",
        "Records & Streaks",
        "Points by Position"
    ])

    # ---------------------------
    # TAB 1: HEAD-TO-HEAD
    # ---------------------------
    with tab1:
        st.subheader("Head-to-Head Records")

        h2h_type = st.radio(
            "View",
            ["Win-Draw-Loss Records", "Total Points Scored"],
            horizontal=True
        )

        if h2h_type == "Win-Draw-Loss Records":
            h2h_matrix = build_h2h_matrix(matches_df, team_names)
            if not h2h_matrix.empty:
                st.caption("Each cell shows the row team's record (W-D-L) against the column team")
                render_styled_table(h2h_matrix.reset_index(), text_align={col: "center" for col in h2h_matrix.columns})
            else:
                st.info("Not enough data to build head-to-head matrix.")
        else:
            h2h_pts_matrix = build_h2h_points_matrix(matches_df)
            if not h2h_pts_matrix.empty:
                st.caption("Each cell shows total points the row team scored against the column team")
                render_styled_table(h2h_pts_matrix.reset_index(), text_align={col: "center" for col in h2h_pts_matrix.columns})
            else:
                st.info("Not enough data to build points matrix.")

    # ---------------------------
    # TAB 2: SCORING ANALYSIS
    # ---------------------------
    with tab2:
        st.subheader("Scoring Statistics")

        if not scoring_stats.empty:
            # Display stats table
            display_cols = ["Team", "Games", "Total_Pts", "Avg_Pts", "Std_Dev", "Min_Pts", "Max_Pts", "Consistency"]
            stats_display = scoring_stats[display_cols].copy()
            stats_display.columns = ["Team", "GP", "Total", "Avg", "Std Dev", "Min", "Max", "Consistency"]
            render_styled_table(
                stats_display,
                col_formats={"Avg": "{:.1f}", "Std Dev": "{:.1f}", "Consistency": "{:.2f}"},
                positive_color_cols=["Total", "Avg", "Consistency"],
            )

            st.divider()

            # Scoring distribution box plot
            fig = plot_scoring_distribution(weekly_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Points for vs against scatter
            fig = plot_points_vs_against(scoring_stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for scoring analysis.")

    # ---------------------------
    # TAB 3: WEEKLY TRENDS
    # ---------------------------
    with tab3:
        st.subheader("Weekly Performance Trends")

        chart_type = st.radio(
            "Chart Type",
            ["Weekly Rank", "Cumulative Points"],
            horizontal=True
        )

        if chart_type == "Weekly Rank":
            fig = plot_weekly_rank_trends(weekly_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Track how each team ranks each gameweek (1 = highest scorer that week)")
        else:
            fig = plot_cumulative_points(weekly_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # TAB 4: STRENGTH OF SCHEDULE
    # ---------------------------
    with tab4:
        st.subheader("Strength of Schedule")

        sos_df = calculate_strength_of_schedule(weekly_scores, scoring_stats)

        if not sos_df.empty:
            st.caption("Based on average points scored by opponents throughout the season")

            sos_display = sos_df.rename(columns={
                "SOS_Rank": "SOS Rank",
                "Opp_Avg_Season": "Opp Season Avg",
                "Opp_Avg_vs_Team": "Opp Avg vs Team",
                "Difficulty_Delta": "Delta",
            })
            render_styled_table(
                sos_display,
                col_formats={"Opp Season Avg": "{:.1f}", "Opp Avg vs Team": "{:.1f}", "Delta": "{:+.1f}"},
                negative_color_cols=["Delta"],
            )

            st.markdown("""
            **How to read this:**
            - **SOS Rank**: 1 = faced the toughest opponents on average
            - **Opp Season Avg**: The average gameweek score of teams you've faced
            - **Delta**: Positive means opponents scored more against you than their season average (unlucky or poor defense)
            """)
        else:
            st.info("Not enough data for strength of schedule analysis.")

    # ---------------------------
    # TAB 5: RECORDS & STREAKS
    # ---------------------------
    with tab5:
        st.subheader("League Records")

        records = calculate_records(matches_df, weekly_scores)

        if records:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Highest Single GW Score**")
                r = records["highest_score"]
                st.metric(
                    label=f"{r['team']} (GW{r['gameweek']})",
                    value=f"{r['points']} pts",
                    delta=f"vs {r['opponent']}"
                )

                st.markdown("**Biggest Win**")
                r = records["biggest_win"]
                st.metric(
                    label=f"{r['winner']} (GW{r['gameweek']})",
                    value=r["score"],
                    delta=f"+{r['margin']} margin vs {r['loser']}"
                )

            with col2:
                st.markdown("**Lowest Single GW Score**")
                r = records["lowest_score"]
                st.metric(
                    label=f"{r['team']} (GW{r['gameweek']})",
                    value=f"{r['points']} pts",
                    delta=f"vs {r['opponent']}",
                    delta_color="inverse"
                )

                st.markdown("**Closest Match**")
                r = records["closest_match"]
                st.metric(
                    label=f"GW{r['gameweek']}",
                    value=r["score"],
                    delta=f"{r['team1']} vs {r['team2']} ({r['margin']} pt margin)"
                )

        st.divider()

        st.subheader("Streaks")
        streaks_df = calculate_streaks(weekly_scores)

        if not streaks_df.empty:
            streaks_display = streaks_df.rename(columns={
                "Current_Streak": "Current",
                "Longest_Win_Streak": "Best Win Streak",
                "Longest_Loss_Streak": "Worst Loss Streak",
            })
            render_styled_table(
                streaks_display,
                text_align={"Current": "center", "Best Win Streak": "center", "Worst Loss Streak": "center"},
                positive_color_cols=["Best Win Streak"],
                negative_color_cols=["Worst Loss Streak"],
            )
        else:
            st.info("Not enough data for streak analysis.")

    # ---------------------------
    # TAB 6: POINTS BY POSITION
    # ---------------------------
    with tab6:
        st.subheader("Points by Position")

        POSITION_COLORS = {
            "GK": "#f39c12",
            "DEF": "#3498db",
            "MID": "#2ecc71",
            "FWD": "#e74c3c",
        }

        with st.spinner("Loading position data..."):
            pos_df = get_draft_points_by_position(league_id)

        if pos_df.empty:
            st.info("No position data available yet.")
        else:
            # Toggle between raw points and percentage
            view_mode = st.radio(
                "View",
                ["Raw Points", "Percentage"],
                horizontal=True,
                key="draft_league_pos_view"
            )

            pos_cols = ["GK", "DEF", "MID", "FWD"]

            # Calculate league averages
            avg_row = {"Team": "League Average"}
            for col in pos_cols:
                avg_row[col] = round(pos_df[col].mean(), 1)
            avg_row["Total"] = round(pos_df["Total"].mean(), 1)

            display_df = pos_df.copy()

            if view_mode == "Percentage":
                for _, row in display_df.iterrows():
                    total = row["Total"]
                    if total > 0:
                        for col in pos_cols:
                            display_df.at[_, col] = round(row[col] / total * 100, 1)
                    else:
                        for col in pos_cols:
                            display_df.at[_, col] = 0.0
                display_df["Total"] = 100.0

                # Recalculate avg for percentage view
                avg_total = avg_row["Total"]
                if avg_total > 0:
                    for col in pos_cols:
                        avg_row[col] = round(avg_row[col] / avg_total * 100, 1)
                avg_row["Total"] = 100.0

            # Append league average row
            avg_df = pd.DataFrame([avg_row])
            table_df = pd.concat([display_df, avg_df], ignore_index=True)

            # Style: gradient red-white-green based on distance from average
            def gradient_vs_avg(row):
                styles = [""] * len(row)
                if row["Team"] == "League Average":
                    styles = ["font-weight: bold; background-color: #f0f0f0"] * len(row)
                else:
                    for i, col in enumerate(row.index):
                        if col in pos_cols:
                            avg_val = avg_row[col]
                            # Compute column range for scaling
                            col_vals = table_df[table_df["Team"] != "League Average"][col]
                            col_min = col_vals.min()
                            col_max = col_vals.max()
                            val = row[col]

                            if val > avg_val and col_max > avg_val:
                                # Scale 0..1 from avg to max (white to green)
                                t = min((val - avg_val) / (col_max - avg_val), 1.0)
                                r = int(255 - t * (255 - 72))   # 255 -> 72
                                g = int(255 - t * (255 - 199))  # 255 -> 199
                                b = int(255 - t * (255 - 142))  # 255 -> 142
                                styles[i] = f"background-color: rgb({r},{g},{b})"
                            elif val < avg_val and col_min < avg_val:
                                # Scale 0..1 from avg to min (white to red)
                                t = min((avg_val - val) / (avg_val - col_min), 1.0)
                                r = int(255 - t * (255 - 248))  # 255 -> 248
                                g = int(255 - t * (255 - 105))  # 255 -> 105
                                b = int(255 - t * (255 - 107))  # 255 -> 107
                                styles[i] = f"background-color: rgb({r},{g},{b})"
                return styles

            styled = table_df.style.apply(gradient_vs_avg, axis=1)

            suffix = "%" if view_mode == "Percentage" else " pts"
            styled = styled.format(
                {col: f"{{:.1f}}{suffix}" for col in pos_cols + ["Total"]},
                subset=pos_cols + ["Total"]
            )

            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.divider()

            # Stacked bar chart
            bar_df = pos_df.melt(
                id_vars=["Team"],
                value_vars=pos_cols,
                var_name="Position",
                value_name="Points"
            )

            fig_bar = px.bar(
                bar_df,
                x="Team",
                y="Points",
                color="Position",
                title="Points Distribution by Position",
                barmode="stack",
                color_discrete_map=POSITION_COLORS,
                category_orders={"Position": pos_cols}
            )
            fig_bar.update_layout(
                xaxis_title="",
                yaxis_title="Total Points",
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # League-wide pie chart
            league_totals = {col: pos_df[col].sum() for col in pos_cols}
            pie_df = pd.DataFrame({
                "Position": list(league_totals.keys()),
                "Points": list(league_totals.values())
            })

            fig_pie = px.pie(
                pie_df,
                values="Points",
                names="Position",
                title="League-Wide Points Distribution",
                color="Position",
                color_discrete_map=POSITION_COLORS,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
