import config
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple
import streamlit as st
from scripts.common.utils import get_league_teams, get_rotowire_player_projections, \
    get_team_composition_for_gameweek, get_team_id_by_name, merge_fpl_players_and_projections

WIN_COLOR  = "#22c55e"   # green
DRAW_COLOR = "#ffffff"   # white
LOSS_COLOR = "#ef4444"   # red

def _fetch_league_details(league_id: int) -> dict:
    """Return the league details JSON (standings, matches, entries)."""
    url = f"https://draft.premierleague.com/api/league/{league_id}/details"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _compute_team_record(team_id: int, league_id: int, up_to_gw: Optional[int] = None) -> Tuple[int, int, int, int]:
    """
    Return (W, D, L, Pts) using the official standings in league details.
    If standings are missing, fall back to scanning matches.
    """
    details = _fetch_league_details(league_id)
    standings = details.get("standings", []) or []

    # Preferred: read straight from standings (same source your league table uses)
    row = next((s for s in standings if s.get("league_entry") == team_id), None)
    if row is not None:
        w = int(row.get("matches_won", 0) or 0)
        d = int(row.get("matches_drawn", 0) or 0)
        l = int(row.get("matches_lost", 0) or 0)
        pts = w * 3 + d
        return w, d, l, pts

    # Fallback: compute from matches (if standings unavailable early season)
    w = d = l = 0
    matches = details.get("matches", []) or []
    for m in matches:
        if up_to_gw is not None and int(m.get("event", 0) or 0) > int(up_to_gw):
            continue
        if team_id not in (m.get("league_entry_1"), m.get("league_entry_2")):
            continue
        s1 = int(m.get("league_entry_1_points", 0) or 0)
        s2 = int(m.get("league_entry_2_points", 0) or 0)
        t1 = m.get("league_entry_1")
        # treat any equal points as a draw; otherwise compare for W/L
        if s1 == s2:
            d += 1
        else:
            won = (team_id == t1 and s1 > s2) or (team_id != t1 and s2 > s1)
            if won: w += 1
            else:   l += 1
    pts = w * 3 + d
    return w, d, l, pts

def _compute_h2h_breakdown(team_id: int, matches: list, id_to_name: dict, up_to_gw: int) -> pd.DataFrame:
    """Per-opponent W/D/L and percentages vs each opponent."""
    rows = []
    # get unique opponents the team has faced
    opp_ids = set()
    for m in matches:
        if team_id in (m.get("league_entry_1"), m.get("league_entry_2")):
            opp_ids.add(m["league_entry_2"] if m["league_entry_1"] == team_id else m["league_entry_1"])

    for opp in opp_ids:
        w=d=l=0
        total=0
        for m in matches:
            if up_to_gw is not None and int(m.get("event", 0)) > int(up_to_gw):
                continue
            if not m.get("finished"):
                continue
            t1, t2 = m.get("league_entry_1"), m.get("league_entry_2")
            if {t1, t2} != {team_id, opp}:
                continue
            s1 = m.get("league_entry_1_points", 0) or 0
            s2 = m.get("league_entry_2_points", 0) or 0
            total += 1
            if s1 == s2:
                d += 1
            else:
                our_win = (team_id == t1 and s1 > s2) or (team_id == t2 and s2 > s1)
                if our_win: w += 1
                else: l += 1
        if total == 0:
            continue
        win_pct  = w/total
        draw_pct = d/total
        loss_pct = l/total
        rows.append({
            "Opponent": id_to_name.get(opp, f"Team {opp}"),
            "W": w, "D": d, "L": l, "GP": total,
            "Win%": win_pct, "Draw%": draw_pct, "Loss%": loss_pct
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Win%", ascending=False).reset_index(drop=True)
    return df

def _h2h_stacked_bar(df: pd.DataFrame, title: str = "Head-to-Head (Win/Draw/Loss %)"):
    """Plotly horizontal stacked bars: Wins left (green), Draws middle (white), Losses right (red)."""
    if df.empty:
        return go.Figure()
    y = df["Opponent"].tolist()
    win = df["Win%"].tolist()
    draw = df["Draw%"].tolist()
    loss = df["Loss%"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Win %",  x=win,  y=y, orientation="h", marker_color=WIN_COLOR))
    fig.add_trace(go.Bar(name="Draw %", x=draw, y=y, orientation="h", marker_color=DRAW_COLOR, marker_line=dict(color="#ddd", width=1)))
    fig.add_trace(go.Bar(name="Loss %", x=loss, y=y, orientation="h", marker_color=LOSS_COLOR))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(title="Percentage", tickformat=".0%", range=[0,1]),
        yaxis=dict(autorange="reversed"),  # top = highest Win%
        legend_title=None,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
    )
    # subtle grid
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    return fig

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

    # Overall record + H2H chart
    details = _fetch_league_details(config.FPL_DRAFT_LEAGUE_ID)
    matches = details.get("matches", [])
    id_to_name = {e["id"]: e["entry_name"] for e in details.get("league_entries", [])}

    # Overall record up to current GW
    w, d, l, pts = _compute_team_record(team_id, matches, up_to_gw=config.CURRENT_GAMEWEEK)
    st.subheader("Season Record")
    st.markdown(f"**{w}W – {d}D – {l}L = {pts} points**")

    # Head-to-head breakdown (sorted by Win%)
    h2h_df = _compute_h2h_breakdown(team_id, matches, id_to_name, up_to_gw=config.CURRENT_GAMEWEEK)
    st.subheader("Head-to-Head vs League Teams")
    st.plotly_chart(_h2h_stacked_bar(h2h_df), use_container_width=True)
    # optional: show raw table beneath
    with st.expander("Show head-to-head table"):
        st.dataframe(h2h_df.assign(**{
            "Win%": (h2h_df["Win%"]*100).round(1),
            "Draw%": (h2h_df["Draw%"]*100).round(1),
            "Loss%": (h2h_df["Loss%"]*100).round(1),
        }), use_container_width=True)


    # Display the team's player projected stats
    st.subheader(f"{selected_team} Projected Player Stats for Gameweek {config.CURRENT_GAMEWEEK}")
    st.dataframe(show_team_projections(team_id, player_projections, config.CURRENT_GAMEWEEK),
                 use_container_width=True, height=560)

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

