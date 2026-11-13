"""
Classic FPL - Home Page

Displays league standings, points progression charts, and rank trends
for Classic FPL leagues. Supports multiple leagues via dropdown.
"""

import config
import pandas as pd
import plotly.express as px
import random
import streamlit as st
from typing import Optional, Dict, List

from scripts.classic.my_week import (
    build_week_view,
    render_week_actions,
    render_week_detail,
    week_card_html,
)
from scripts.common import cards
from scripts.common.error_helpers import get_logger
from scripts.common.luck_analysis import (
    extract_classic_h2h_gw_scores,
    calculate_all_play_standings,
    calculate_h2h_streaks,
    build_classic_h2h_weekly_scores,
    format_streak_html,
    render_luck_adjusted_table,
    render_standings_table,
)
from scripts.common.error_helpers import show_api_error
from scripts.common.season_odds import (
    render_classic_h2h_season_odds,
    render_classic_points_season_odds,
)
from scripts.common.styled_tables import render_styled_table
from scripts.common.utils import (
    get_league_standings,
    get_classic_team_history,
    get_entry_details,
    get_current_gameweek,
    get_all_h2h_league_matches,
)


# ---------------------------
# LEAGUE DATA FUNCTIONS
# ---------------------------

_logger = get_logger("fpl_app.classic.home")


def get_league_display_options() -> list:
    """
    Build list of league options for the dropdown.
    Returns list of dicts with 'id', 'name', 'display' keys.
    """
    leagues = config.FPL_CLASSIC_LEAGUE_IDS or []
    options = []

    for league in leagues:
        league_id = league["id"]
        # Use configured name or fetch from API
        if league.get("name"):
            name = league["name"]
        else:
            # Try to fetch name from API
            data = get_league_standings(league_id)
            if data and "league" in data:
                name = data["league"].get("name", f"League {league_id}")
            else:
                name = f"League {league_id}"

        options.append({
            "id": league_id,
            "name": name,
            "display": f"{name} ({league_id})"
        })

    return options


def fetch_standings_data(league_id: int) -> Optional[dict]:
    """
    Fetch league standings and metadata.
    Returns dict with 'league_info', 'standings', 'scoring_type'.
    """
    data = get_league_standings(league_id)

    if not data:
        return None

    league_info = data.get("league", {})
    standings_data = data.get("standings", {})

    # Determine scoring type
    league_type = league_info.get("league_type", "x")
    scoring = league_info.get("scoring", "c")

    if league_type == "s" or scoring == "h":
        scoring_type = "H2H"
    else:
        scoring_type = "Classic"

    return {
        "league_info": league_info,
        "standings": standings_data,
        "scoring_type": scoring_type
    }


def format_classic_standings(standings_data: dict) -> pd.DataFrame:
    """Format classic (total points) league standings."""
    results = standings_data.get("results", [])

    if not results:
        return pd.DataFrame()

    rows = []
    for entry in results:
        rows.append({
            "Rank": entry.get("rank", "-"),
            "Team": entry.get("entry_name", "Unknown"),
            "Manager": entry.get("player_name", "Unknown"),
            "GW Pts": entry.get("event_total", 0),
            "Total Pts": entry.get("total", 0),
            "entry_id": entry.get("entry", 0),
        })

    return pd.DataFrame(rows)


def format_h2h_standings(standings_data: dict) -> pd.DataFrame:
    """Format H2H league standings."""
    results = standings_data.get("results", [])

    if not results:
        return pd.DataFrame()

    rows = []
    for entry in results:
        rows.append({
            "Rank": entry.get("rank", "-"),
            "Team": entry.get("entry_name", "Unknown"),
            "Manager": entry.get("player_name", "Unknown"),
            "W": entry.get("matches_won", 0),
            "D": entry.get("matches_drawn", 0),
            "L": entry.get("matches_lost", 0),
            "PF": entry.get("points_for", 0),
            "PA": entry.get("points_against", 0),
            "Pts": entry.get("total", 0),
            "entry_id": entry.get("entry", 0),
        })

    return pd.DataFrame(rows)


def get_classic_h2h_luck_adjusted(league_id: int, standings_data: dict) -> pd.DataFrame:
    """
    Calculate luck-adjusted standings for a Classic H2H league using All-Play Record.

    Parameters:
    - league_id: The H2H FPL league ID.
    - standings_data: The standings dict from the league API (passed to avoid re-fetching).

    Returns:
    - DataFrame with All-Play standings.
    """
    matches = get_all_h2h_league_matches(league_id)
    if not matches:
        return pd.DataFrame()

    gw_scores = extract_classic_h2h_gw_scores(matches)
    if gw_scores.empty:
        return pd.DataFrame()

    # Build actual standings from formatted H2H standings
    h2h_df = format_h2h_standings(standings_data)
    if h2h_df.empty:
        return calculate_all_play_standings(gw_scores)

    actual_standings = h2h_df[['entry_id', 'Team', 'Rank', 'Pts']].rename(columns={
        'entry_id': 'team_id',
        'Team': 'team',
        'Rank': 'actual_rank',
        'Pts': 'actual_pts',
    })

    return calculate_all_play_standings(gw_scores, actual_standings)


# ---------------------------
# CHART FUNCTIONS
# ---------------------------

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title=dict(font=dict(size=22, color="#ffffff"), x=0.5, xanchor="center"),
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
)

@st.cache_data(ttl=600)
def fetch_team_histories(team_ids: List[int], team_names: Dict[int, str]) -> pd.DataFrame:
    """
    Fetch gameweek history for multiple teams.
    Returns DataFrame with columns: Team, Gameweek, GW_Points, Total_Points, Overall_Rank
    """
    all_data = []

    for team_id in team_ids:
        history = get_classic_team_history(team_id)
        if not history or not history.get("current"):
            continue

        team_name = team_names.get(team_id, f"Team {team_id}")

        for gw_data in history["current"]:
            all_data.append({
                "Team": team_name,
                "Gameweek": gw_data.get("event"),
                "GW_Points": gw_data.get("points", 0),
                "Total_Points": gw_data.get("total_points", 0),
                "Overall_Rank": gw_data.get("overall_rank"),
                "GW_Rank": gw_data.get("rank"),
            })

    return pd.DataFrame(all_data)


def plot_total_points_over_time(history_df: pd.DataFrame) -> Optional[px.line]:
    """Create line chart of total points over time."""
    if history_df.empty:
        return None

    fig = px.line(
        history_df,
        x="Gameweek",
        y="Total_Points",
        color="Team",
        title="Total Points Over Time",
        labels={
            "Total_Points": "Total Points",
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="Total Points",
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )

    return fig


def plot_total_points_vs_avg(history_df: pd.DataFrame) -> Optional[px.line]:
    """Create line chart of total points relative to league average per gameweek."""
    if history_df.empty:
        return None

    # Calculate league average total points per gameweek
    avg_by_gw = history_df.groupby("Gameweek")["Total_Points"].mean().reset_index()
    avg_by_gw.rename(columns={"Total_Points": "Avg_Total_Points"}, inplace=True)

    df = history_df.merge(avg_by_gw, on="Gameweek")
    df["Points_vs_Avg"] = df["Total_Points"] - df["Avg_Total_Points"]

    fig = px.line(
        df,
        x="Gameweek",
        y="Points_vs_Avg",
        color="Team",
        title="Total Points vs League Average",
        labels={
            "Points_vs_Avg": "Points vs Average",
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="Points vs Average",
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )

    # Add a zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.6)

    return fig


def plot_gw_points_over_time(history_df: pd.DataFrame) -> Optional[px.line]:
    """Create line chart of gameweek points over time."""
    if history_df.empty:
        return None

    fig = px.line(
        history_df,
        x="Gameweek",
        y="GW_Points",
        color="Team",
        title="Gameweek Points Over Time",
        labels={
            "GW_Points": "GW Points",
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="GW Points",
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )

    return fig


def plot_gw_points_vs_avg(history_df: pd.DataFrame) -> Optional[px.line]:
    """Create line chart of gameweek points relative to league average."""
    if history_df.empty:
        return None

    avg_by_gw = history_df.groupby("Gameweek")["GW_Points"].mean().reset_index()
    avg_by_gw.rename(columns={"GW_Points": "Avg_GW_Points"}, inplace=True)

    df = history_df.merge(avg_by_gw, on="Gameweek")
    df["GW_Pts_vs_Avg"] = df["GW_Points"] - df["Avg_GW_Points"]

    fig = px.line(
        df,
        x="Gameweek",
        y="GW_Pts_vs_Avg",
        color="Team",
        title="Gameweek Points vs League Average",
        labels={
            "GW_Pts_vs_Avg": "GW Points vs Average",
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title="GW Points vs Average",
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.6)

    return fig


def plot_rank_progression(history_df: pd.DataFrame, use_overall_rank: bool = False) -> Optional[px.line]:
    """Create line chart of rank over time (inverted - lower is better)."""
    if history_df.empty:
        return None

    rank_col = "Overall_Rank" if use_overall_rank else "GW_Rank"
    title = "Overall Rank Progression" if use_overall_rank else "Gameweek Rank Progression"
    y_label = "Overall Rank" if use_overall_rank else "GW Rank"

    # Filter out rows without rank data
    plot_df = history_df.dropna(subset=[rank_col])
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="Gameweek",
        y=rank_col,
        color="Team",
        title=title,
        labels={
            rank_col: y_label,
            "Gameweek": "Gameweek",
            "Team": "Team"
        }
    )

    # Invert y-axis (rank 1 at top)
    fig.update_layout(
        xaxis_title="Gameweek",
        yaxis_title=y_label,
        hovermode="x unified",
        **_DARK_CHART_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")

    return fig


# ---------------------------
# MAIN PAGE
# ---------------------------

def _league_card(league_name: str, scoring_type: str, standings: dict,
                 current_gw: int) -> str:
    """The selected league: what it is, how big, and where you are in it.

    Was a bare `### heading` plus a `st.metric("Format")`, which said less and
    did not line up with anything beside it.
    """
    results = (standings or {}).get("results") or []
    mine = None
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if team_id:
        mine = next((r for r in results
                     if str(r.get("entry")) == str(team_id)), None)
    body = (
        cards.eyebrow("League")
        + f'<div style="display:flex;align-items:center;gap:9px;flex-wrap:wrap;">'
          f'{cards.headline(league_name)}'
          f'{cards.badge(scoring_type, cards.ACCENT_ALT if scoring_type == "H2H" else cards.ACCENT)}'
          f'</div>'
        + cards.note("%d manager%s &middot; gameweek %d"
                     % (len(results), "" if len(results) == 1 else "s", current_gw))
    )
    if mine is not None:
        body += cards.push(cards.stats([
            cards.stat(_ordinal(mine.get("rank")), "your rank", cards.ACCENT),
            cards.stat("%s" % mine.get("total", 0), "total pts"),
        ]))
    return cards.card(body)


def _my_team_card(standings: dict) -> str:
    """Your own team, at the top of the page rather than the bottom of it.

    This was four `st.metric`s inside a collapsed expander below three
    dividers -- the most personal data on the page in the least reachable place,
    in the one widget style the rest of the app replaced with cards. Returns ""
    when no team is configured, so the band closes up rather than holding an
    empty card.
    """
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        return ""
    entry = get_entry_details(team_id) or {}
    history = get_classic_team_history(team_id) or {}
    weeks = history.get("current") or []
    latest = weeks[-1] if weeks else {}

    manager = ("%s %s" % (entry.get("player_first_name", ""),
                          entry.get("player_last_name", ""))).strip()
    body = cards.eyebrow("My team") + cards.headline(
        entry.get("name") or "Your team")
    if manager:
        body += cards.note(manager, cards.MUTED)

    if latest:
        overall = latest.get("overall_rank")
        # A rank means little without the field it is out of -- "2.1m" reads as
        # bad until you know it is 2.1m of 10.8m. FPL publishes the count on the
        # bootstrap, which is already cached for this page.
        total_players = _total_players()
        rank_text = _compact_rank(overall) if overall else "—"
        if overall and total_players:
            rank_text += " / %s" % _compact_rank(total_players)
        body += cards.stats([
            cards.stat("%s" % latest.get("total_points", "—"), "total pts",
                       cards.ACCENT),
            cards.stat(("%s" % latest.get("points", "—")),
                       "GW%s pts" % latest.get("event", "")),
            cards.stat(rank_text, "overall rank"),
        ])
        chips = history.get("chips") or []
        if chips:
            names = {"wildcard": "WC", "freehit": "FH", "bboost": "BB",
                     "3xc": "TC"}
            body += cards.push(cards.note("Chips: " + ", ".join(
                "%s GW%s" % (names.get(c.get("name"), c.get("name")),
                             c.get("event", "?")) for c in chips)))
        else:
            body += cards.push(cards.note("No chips played yet."))
    else:
        body += cards.push(cards.note("Could not load your team history."))
    return cards.card(body)


def _top_band(league_html: str, team_html: str, week_html: str) -> str:
    """The three top cards as one CSS grid.

    **Grid items stretch to the tallest in the row by default**, which is what
    makes these level however their text wraps -- the thing estimating pixel
    heights could not do, because whether a verdict wraps onto a second line is
    a browser's decision and not Python's. One block of markup rather than three
    Streamlit columns, for the same reason: columns are independent, a grid row
    is not.

    The controls that belong to these cards -- the league selector, the week's
    buttons -- render underneath in a matching `st.columns`, since a Streamlit
    widget cannot live inside one block of markup.
    """
    present = [html for html in (league_html, team_html, week_html) if html]
    if len(present) == 3:
        columns = "1fr 1fr 2fr"
    elif len(present) == 2:
        columns = "1fr 2fr"
    else:
        columns = "1fr"
    return (
        '<style>'
        '.fpl-band{display:grid;gap:16px;align-items:stretch;'
        'grid-template-columns:%s;margin-bottom:10px;}'
        '@media (max-width:900px){.fpl-band{grid-template-columns:1fr;}}'
        '</style>'
        '<div class="fpl-band">%s</div>' % (columns, "".join(present)))


def _total_players() -> int:
    """How many managers are playing, for the overall rank to be out of.

    Returns 0 when the bootstrap cannot be read, which renders the rank alone
    rather than a rank out of nothing.
    """
    try:
        from scripts.common.fpl_classic_api import get_classic_bootstrap_static
        return int((get_classic_bootstrap_static() or {}).get("total_players") or 0)
    except Exception:
        _logger.warning("Could not read the player count", exc_info=True)
        return 0


def _compact_rank(value) -> str:
    """412,345 -> 412k. An overall rank is read for its order of magnitude."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "—"
    if n >= 1_000_000:
        return "%.1fm" % (n / 1_000_000)
    if n >= 10_000:
        return "%dk" % (n // 1_000)
    return "{:,}".format(n)


def _ordinal(value) -> str:
    """1 -> 1st. Rendered beside a league position, so the suffix matters."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "—"
    if 10 <= n % 100 <= 20:
        return "%dth" % n
    return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def show_classic_home_page():
    """Display the Classic FPL Home page with standings and charts."""

    st.title("Classic FPL League")

    # Get configured leagues
    league_options = get_league_display_options()

    if not league_options:
        st.warning("No Classic leagues configured.")
        st.info(
            "Add leagues to `FPL_CLASSIC_LEAGUE_IDS` in your `.env` file:\n\n"
            "```\nFPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League\n```\n\n"
            "The format is `league_id:League Name` with multiple leagues separated by commas."
        )
        return

    # Initialize random league selection on first load
    if "classic_home_league_index" not in st.session_state:
        st.session_state.classic_home_league_index = 0

    # ---------------------------
    # THE TOP: which league, your team, and your week in it
    # ---------------------------
    # The three cards are one CSS grid so they stretch to each other; their
    # controls sit in a matching column row underneath, because a widget cannot
    # live inside a block of markup. The grid is written into a slot reserved
    # *before* the controls, so the selector -- whose league id everything below
    # fetches -- still runs first.
    band_slot = st.empty()
    has_team = bool(getattr(config, "FPL_CLASSIC_TEAM_ID", None))
    control_cols = st.columns([1, 1, 2] if has_team else [1, 2], gap="medium")

    display_options = [opt["display"] for opt in league_options]
    with control_cols[0]:
        selected_display = st.selectbox(
            "Select League",
            options=display_options,
            index=st.session_state.classic_home_league_index,
            key="classic_home_league_selector",
            label_visibility="collapsed",
        )

    new_index = display_options.index(selected_display)
    if new_index != st.session_state.classic_home_league_index:
        st.session_state.classic_home_league_index = new_index

    selected_league = league_options[st.session_state.classic_home_league_index]
    league_id = selected_league["id"]

    with st.spinner("Loading league data..."):
        data = fetch_standings_data(league_id)

    if not data:
        show_api_error(f"loading standings for league {league_id}", hint_key="classic_league_stale")
        return

    league_info = data["league_info"]
    scoring_type = data["scoring_type"]
    standings = data["standings"]
    # The league card, the odds paths and the week all need it, and
    # `get_current_gameweek` is cached.
    current_gw = get_current_gameweek() or 1

    try:
        week_view = build_week_view(league_id=league_id,
                                    scoring_type=scoring_type,
                                    current_gw=current_gw)
    except Exception:
        # Guarded as well as inside the section: `page_error_boundary` would
        # replace the whole page with an error, and the standings below are why
        # most visits happen, so a fault here must cost the week alone.
        _logger.warning("Could not build the week view", exc_info=True)
        week_view = None

    band_slot.markdown(
        _top_band(_league_card(league_info.get("name", "Unknown"),
                               scoring_type, standings, current_gw),
                  _my_team_card(standings) if has_team else "",
                  week_card_html(week_view) if week_view else ""),
        unsafe_allow_html=True)

    if week_view is not None:
        with control_cols[-1]:
            render_week_actions(week_view)

    # ---------------------------
    # THE LEAGUE, AND THE WEEK IN FULL
    # ---------------------------
    # A tab rather than an expander in the right-hand column: the detail is a
    # page's worth of rows, and an expander there was a wide control in a narrow
    # space that hid it behind two clicks.
    league_tab, week_tab = st.tabs(["🏆  League", "🗓️  My week in full"])

    with week_tab:
        render_week_detail(week_view)

    with league_tab:
        if scoring_type == "H2H":
            df = format_h2h_standings(standings)

            # Computed once for the table's streak column, the panels under it
            # and the records row below.
            h2h_matches = get_all_h2h_league_matches(league_id)
            h2h_weekly_scores = build_classic_h2h_weekly_scores(h2h_matches)
            h2h_streaks_df = (calculate_h2h_streaks(h2h_weekly_scores)
                              if not h2h_weekly_scores.empty else None)

            st.subheader("League Standings")
            if df.empty:
                st.info("No standings data available yet.")
            else:
                # A radio rather than a checkbox: the two are alternative views
                # of one table, not independent options, and only the chosen one
                # is computed -- replaying every fixture is the expensive half.
                view = st.radio(
                    "Standings view", ["Table", "Luck Adjusted"],
                    horizontal=True, label_visibility="collapsed",
                    key=f"standings_view_{league_id}",
                    help="Replays every team against every other team each "
                         "gameweek — an all-play record — so Luck +/- shows how "
                         "much the actual schedule helped or hurt.",
                )
                if view == "Luck Adjusted":
                    with st.spinner("Replaying every fixture..."):
                        luck_df = get_classic_h2h_luck_adjusted(league_id, standings)
                    render_luck_adjusted_table(luck_df)
                    st.caption("**Luck +/-** is the gap between your real record "
                               "and your all-play one: positive means the "
                               "schedule has helped.")
                else:
                    display_df = df.drop(columns=["entry_id"], errors="ignore")
                    if h2h_streaks_df is not None and not h2h_streaks_df.empty:
                        # On the entry id: two managers who never renamed their
                        # squad are both "Change Name", and a name join gave
                        # them each other's streak.
                        display_df = display_df.copy()
                        if ("Team_ID" in h2h_streaks_df.columns
                                and "entry_id" in df.columns):
                            streak_map = dict(zip(h2h_streaks_df["Team_ID"],
                                                  h2h_streaks_df["Current_Streak"]))
                            display_df["Streak"] = df["entry_id"].map(streak_map).fillna("-")
                        else:
                            streak_map = dict(zip(h2h_streaks_df["Team"],
                                                  h2h_streaks_df["Current_Streak"]))
                            display_df["Streak"] = display_df["Team"].map(streak_map).fillna("-")
                        cols = list(display_df.columns)
                        l_idx = cols.index("L")
                        cols.insert(l_idx + 1, cols.pop(cols.index("Streak")))
                        display_df = display_df[cols]
                    render_standings_table(display_df, is_h2h=True)

                # Under the table, side by side: both are narrow, and neither
                # is worth the full width the standings earn. Rendered
                # unconditionally -- measured at **0.002s** for 2,000 seeded
                # simulations of a 20-manager league, with its fixtures already
                # fetched for the streaks column, so the control it used to sit
                # behind cost a click and saved nothing.
                odds_col, streak_col = st.columns(2, gap="medium")
                with odds_col:
                    st.markdown("##### 🔮 Season Odds")
                    render_classic_h2h_season_odds(
                        league_id, current_gw,
                        standings.get("results") or [], h2h_matches or [])
                with streak_col:
                    st.markdown("##### 🔥 Streaks")
                    if h2h_streaks_df is not None and not h2h_streaks_df.empty:
                        streaks_display = h2h_streaks_df.drop(
                            columns=["Team_ID"], errors="ignore").rename(columns={
                                "Current_Streak": "Current",
                                "Longest_Win_Streak": "Best Win Run",
                                "Longest_Unbeaten_Streak": "Best Unbeaten Run",
                                "Longest_Loss_Streak": "Worst Loss Run",
                            })
                        render_styled_table(
                            streaks_display,
                            text_align={c: "center" for c in
                                        ("Current", "Best Win Run",
                                         "Best Unbeaten Run", "Worst Loss Run")},
                            positive_color_cols=["Best Win Run",
                                                 "Best Unbeaten Run"],
                            negative_color_cols=["Worst Loss Run"],
                        )
                    else:
                        st.caption("No played fixtures yet, so there are no "
                                   "streaks to show.")
        else:
            df = format_classic_standings(standings)
            h2h_matches, h2h_weekly_scores, h2h_streaks_df = [], pd.DataFrame(), None

            st.subheader("League Standings")
            if df.empty:
                st.info("No standings data available yet.")
            else:
                render_standings_table(
                    df.drop(columns=["entry_id"], errors="ignore"), is_h2h=False)

                st.markdown("##### 🔮 Season Odds")
                # Sigma is measured from a sample of manager histories, which
                # is a cached fetch shared with the progression charts below.
                # It is a league-level noise parameter, so a sample is a sound
                # basis for it -- unlike the means, which come from every
                # manager's own total.
                _sample_ids = df["entry_id"].head(10).tolist()
                _sample_names = dict(zip(df["entry_id"], df["Team"]))
                try:
                    _sample = fetch_team_histories(_sample_ids, _sample_names)
                except Exception:
                    _sample = None
                _odds_df = df.copy()
                _odds_df.attrs["truncated"] = bool(standings.get("has_next"))
                render_classic_points_season_odds(
                    league_id, current_gw, _odds_df, sigma_sample=_sample)

        # Show pagination info
        has_next = standings.get("has_next", False)
        if has_next:
            st.caption("Showing first page of standings. Large leagues have additional pages.")

        st.divider()

        # ---------------------------
        # POINTS PROGRESSION CHARTS
        # ---------------------------
        if df.empty:
            st.info("Charts will be available once the season starts.")
            return

        st.subheader("Season Progression")

        # Slider to select how many teams to show in charts
        num_teams = len(df)
        if num_teams > 5:
            teams_to_show = st.slider(
                "Number of teams to show in charts",
                min_value=3,
                max_value=min(num_teams, 20),
                value=min(num_teams, 10),
                help="Limit teams shown for cleaner charts. Teams are selected by current rank."
            )
        else:
            teams_to_show = num_teams

        # Get top N teams for charts
        top_teams_df = df.head(teams_to_show)
        team_ids = top_teams_df["entry_id"].tolist()
        team_names = dict(zip(top_teams_df["entry_id"], top_teams_df["Team"]))

        # Fetch history for these teams
        with st.spinner("Loading team histories..."):
            history_df = fetch_team_histories(team_ids, team_names)

        if history_df.empty:
            st.info("No gameweek data available yet. Charts will appear once games are played.")
            return

        # Create tabs for different charts
        tab1, tab2, tab3 = st.tabs(["Total Points", "Gameweek Points", "Rank Progression"])

        with tab1:
            pts_view = st.radio(
                "View",
                ["All Teams", "vs League Average"],
                index=1,
                horizontal=True,
                key="classic_total_pts_view",
            )
            if pts_view == "All Teams":
                fig = plot_total_points_over_time(history_df)
            else:
                fig = plot_total_points_vs_avg(history_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to display chart.")

        with tab2:
            gw_pts_view = st.radio(
                "View",
                ["All Teams", "vs League Average"],
                index=1,
                horizontal=True,
                key="classic_gw_pts_view",
            )
            if gw_pts_view == "All Teams":
                fig = plot_gw_points_over_time(history_df)
            else:
                fig = plot_gw_points_vs_avg(history_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to display chart.")

        with tab3:
            # Option to show overall rank or GW rank
            rank_type = st.radio(
                "Rank Type",
                ["Gameweek Rank", "Overall Rank"],
                horizontal=True
            )
            use_overall = rank_type == "Overall Rank"

            fig = plot_rank_progression(history_df, use_overall_rank=use_overall)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Lower rank = better position (rank 1 is best)")
            else:
                st.info("Not enough data to display chart.")

        st.divider()

        # ---------------------------
        # SEASON SUPERLATIVES
        # ---------------------------
        # One row of four, rather than a 2x2 block and then a second section for
        # the streaks table -- which is now beside the standings, where the other
        # league table is. The four cards are the same shape, so they line up
        # without being measured.
        if scoring_type == "H2H" and h2h_weekly_scores is not None \
                and not h2h_weekly_scores.empty:
            st.subheader("Season Records")
            ws = h2h_weekly_scores.copy()
            ws["Margin"] = ws["Points"] - ws["Opp_Points"]
            wins = ws[ws["Margin"] > 0]

            best = ws.loc[ws["Points"].idxmax()]
            worst = ws.loc[ws["Points"].idxmin()]
            records = [
                ("Highest GW score", "%d pts" % int(best["Points"]),
                 "%s (GW%d)" % (best["Team"], int(best["Gameweek"])),
                 "against %d" % int(best["Opp_Points"]), cards.ACCENT),
                ("Lowest GW score", "%d pts" % int(worst["Points"]),
                 "%s (GW%d)" % (worst["Team"], int(worst["Gameweek"])),
                 "against %d" % int(worst["Opp_Points"]), "#f87171"),
            ]
            if not wins.empty:
                big = wins.loc[wins["Margin"].idxmax()]
                close = wins.loc[wins["Margin"].idxmin()]
                records.append(
                    ("Biggest win", "%d-%d" % (int(big["Points"]), int(big["Opp_Points"])),
                     "%s (GW%d)" % (big["Team"], int(big["Gameweek"])),
                     "+%d margin" % int(big["Margin"]), cards.ACCENT_ALT))
                records.append(
                    ("Closest match", "%d-%d" % (int(close["Points"]), int(close["Opp_Points"])),
                     "%s (GW%d)" % (close["Team"], int(close["Gameweek"])),
                     "by %d pt%s" % (int(close["Margin"]),
                                     "" if int(close["Margin"]) == 1 else "s"),
                     "#facc15"))

            bodies = [
                cards.eyebrow(title)
                + f'<div style="color:{accent};font-size:1.5em;font-weight:800;'
                  f'line-height:1.15;">{value}</div>'
                + f'<div style="font-size:0.84em;font-weight:600;margin-top:3px;">'
                  f'{who}</div>'
                + cards.push(cards.note(detail))
                for title, value, who, detail, accent in records
            ]
            for column, html in zip(st.columns(len(bodies), gap="small"),
                                    cards.card_row(bodies)):
                with column:
                    st.markdown(html, unsafe_allow_html=True)

