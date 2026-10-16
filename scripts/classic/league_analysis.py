"""
Classic FPL - League Analysis Page

Provides advanced league-wide analytics including:
- Chip usage analysis across the league
- Rank movement and points behind leader
- Team value comparison
- Transfer activity analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Dict, List

import config
from scripts.common.bench_analysis import compute_classic_league_bench_data, render_league_bench_analysis
from scripts.common.error_helpers import show_api_error
from scripts.common.utils import (
    get_league_standings,
    get_classic_team_history,
    get_current_gameweek,
    get_classic_bootstrap_static,
    get_classic_team_position_data,
)
from scripts.common.styled_tables import render_styled_table
from scripts.common.classic_squad import resolve_classic_squad
from scripts.common.data_validation import check_mini_league_ownership, format_issues
from scripts.common.error_helpers import get_logger
from scripts.common.mini_league import (
    DEFAULT_DIFFERENTIAL_MAX_OWN,
    compute_league_ownership,
    differentials,
    projected_team_totals,
    template_gaps,
    template_score,
)
from scripts.common.player_pool import blended_pool

_logger = get_logger("fpl_app.classic_league_analysis")

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title_font=dict(size=20, color="#ffffff"),
    title_x=0.5,
    title_xanchor="center",
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13)),
)


def _stat_card(label: str, value: str, detail: str = "", accent: str = "#00ff87") -> str:
    detail_html = f'<div style="color:#9ca3af;font-size:12px;margin-top:2px;">{detail}</div>' if detail else ""
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'{detail_html}'
        f'</div>'
    )


# ---------------------------
# DATA FETCHING
# ---------------------------

def get_league_display_options() -> list:
    """Build list of league options for the dropdown."""
    leagues = config.FPL_CLASSIC_LEAGUE_IDS or []
    options = []

    for league in leagues:
        league_id = league["id"]
        if league.get("name"):
            name = league["name"]
        else:
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


@st.cache_data(ttl=600)
def fetch_team_histories_for_league(team_ids: List[int]) -> Dict[int, dict]:
    """Fetch history data for multiple teams."""
    histories = {}
    for team_id in team_ids:
        history = get_classic_team_history(team_id)
        if history:
            histories[team_id] = history
    return histories


# ---------------------------
# CHIP ANALYSIS
# ---------------------------

def analyze_chip_usage(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Analyze chip usage across all teams in the league."""
    chip_data = []

    chip_display_names = {
        "wildcard": "Wildcard",
        "freehit": "Free Hit",
        "bboost": "Bench Boost",
        "3xc": "Triple Captain"
    }

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        chips_used = history.get("chips", [])

        # Track which chips this team has used
        team_chips = {
            "Team": team_name,
            "Wildcard": "-",
            "Free Hit": "-",
            "Bench Boost": "-",
            "Triple Captain": "-",
            "Total Used": 0
        }

        for chip in chips_used:
            chip_name = chip.get("name", "")
            chip_gw = chip.get("event", "?")
            display_name = chip_display_names.get(chip_name, chip_name)

            if display_name in team_chips:
                team_chips[display_name] = f"GW{chip_gw}"
                team_chips["Total Used"] += 1

        chip_data.append(team_chips)

    df = pd.DataFrame(chip_data)
    return df.sort_values("Total Used", ascending=False)


def plot_chip_timing(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> Optional[go.Figure]:
    """Create a timeline of when chips were used."""
    chip_events = []

    chip_display_names = {
        "wildcard": "Wildcard",
        "freehit": "Free Hit",
        "bboost": "Bench Boost",
        "3xc": "Triple Captain"
    }

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        chips_used = history.get("chips", [])

        for chip in chips_used:
            chip_name = chip.get("name", "")
            chip_gw = chip.get("event")
            display_name = chip_display_names.get(chip_name, chip_name)

            if chip_gw:
                chip_events.append({
                    "Team": team_name,
                    "Chip": display_name,
                    "Gameweek": chip_gw
                })

    if not chip_events:
        return None

    df = pd.DataFrame(chip_events)

    # Color mapping for chips
    color_map = {
        "Wildcard": "#e74c3c",
        "Free Hit": "#3498db",
        "Bench Boost": "#2ecc71",
        "Triple Captain": "#9b59b6"
    }

    fig = px.scatter(
        df,
        x="Gameweek",
        y="Team",
        color="Chip",
        title="Chip Usage Timeline",
        color_discrete_map=color_map,
        symbol="Chip"
    )

    fig.update_traces(marker=dict(size=15))
    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        xaxis_title="Gameweek",
        legend_title="Chip",
        height=max(400, len(df["Team"].unique()) * 30),
    )
    fig.update_yaxes(title="")

    return fig


# ---------------------------
# RANK ANALYSIS
# ---------------------------

def build_rank_progression(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Build rank progression data over gameweeks."""
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        for gw_data in current:
            gw = gw_data.get("event")
            overall_rank = gw_data.get("overall_rank")
            total_pts = gw_data.get("total_points", 0)

            if gw and overall_rank:
                all_data.append({
                    "Team": team_name,
                    "Gameweek": gw,
                    "Overall_Rank": overall_rank,
                    "Total_Points": total_pts
                })

    return pd.DataFrame(all_data)


def calculate_rank_changes(rank_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rank changes from start to current."""
    if rank_df.empty:
        return pd.DataFrame()

    # Get first and last gameweek for each team
    first_gw = rank_df.groupby("Team").first().reset_index()
    last_gw = rank_df.groupby("Team").last().reset_index()

    changes = first_gw[["Team"]].copy()
    changes["Start_Rank"] = first_gw["Overall_Rank"]
    changes["Current_Rank"] = last_gw["Overall_Rank"]
    changes["Rank_Change"] = changes["Start_Rank"] - changes["Current_Rank"]
    changes["Start_GW"] = first_gw["Gameweek"]
    changes["Current_GW"] = last_gw["Gameweek"]

    return changes.sort_values("Rank_Change", ascending=False)


def plot_rank_progression(rank_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot overall rank progression over time."""
    if rank_df.empty:
        return None

    fig = px.line(
        rank_df,
        x="Gameweek",
        y="Overall_Rank",
        color="Team",
        title="Overall Rank Progression",
        labels={"Overall_Rank": "Overall Rank", "Gameweek": "Gameweek"}
    )

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        hovermode="x unified",
    )
    fig.update_yaxes(autorange="reversed", title="Overall Rank (lower = better)")

    return fig


# ---------------------------
# POINTS ANALYSIS
# ---------------------------

def calculate_points_behind_leader(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Calculate points behind the leader over time."""
    # Build points progression
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        for gw_data in current:
            gw = gw_data.get("event")
            total_pts = gw_data.get("total_points", 0)

            if gw:
                all_data.append({
                    "Team": team_name,
                    "Gameweek": gw,
                    "Total_Points": total_pts
                })

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Calculate leader's points each gameweek
    leader_pts = df.groupby("Gameweek")["Total_Points"].max().to_dict()

    # Calculate points behind
    df["Leader_Points"] = df["Gameweek"].map(leader_pts)
    df["Points_Behind"] = df["Leader_Points"] - df["Total_Points"]

    return df


def plot_points_behind_leader(points_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot points behind leader over time."""
    if points_df.empty:
        return None

    fig = px.line(
        points_df,
        x="Gameweek",
        y="Points_Behind",
        color="Team",
        title="Points Behind Leader Over Time",
        labels={"Points_Behind": "Points Behind", "Gameweek": "Gameweek"}
    )

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        hovermode="x unified",
    )
    fig.update_yaxes(title="Points Behind Leader")

    return fig


# ---------------------------
# TEAM VALUE ANALYSIS
# ---------------------------

def analyze_team_values(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Analyze team values across the league."""
    value_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        if current:
            # Get latest gameweek data
            latest = current[-1]
            value = latest.get("value", 0) / 10  # Convert to millions
            bank = latest.get("bank", 0) / 10

            # Get starting value (GW1)
            start_value = current[0].get("value", 1000) / 10 if current else 100.0

            value_data.append({
                "Team": team_name,
                "Squad Value": value,
                "Bank": bank,
                "Total Value": value + bank,
                "Value Gain": (value + bank) - start_value
            })

    df = pd.DataFrame(value_data)
    return df.sort_values("Total Value", ascending=False)


def plot_value_comparison(value_df: pd.DataFrame) -> Optional[go.Figure]:
    """Create bar chart comparing team values."""
    if value_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Squad Value",
        x=value_df["Team"],
        y=value_df["Squad Value"],
        marker_color="#3498db"
    ))

    fig.add_trace(go.Bar(
        name="Bank",
        x=value_df["Team"],
        y=value_df["Bank"],
        marker_color="#2ecc71"
    ))

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        barmode="stack",
        xaxis_tickangle=-45,
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Value (millions)")

    return fig


# ---------------------------
# GAMEWEEK PERFORMANCE
# ---------------------------

def analyze_gameweek_performance(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Analyze gameweek-by-gameweek performance."""
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        gw_points = [gw.get("points", 0) for gw in current if gw.get("event")]

        if gw_points:
            all_data.append({
                "Team": team_name,
                "Games": len(gw_points),
                "Total_Points": sum(gw_points),
                "Avg_Points": round(sum(gw_points) / len(gw_points), 1),
                "Best_GW": max(gw_points),
                "Worst_GW": min(gw_points),
                "Std_Dev": round(pd.Series(gw_points).std(), 1) if len(gw_points) > 1 else 0
            })

    df = pd.DataFrame(all_data)
    return df.sort_values("Total_Points", ascending=False)


def get_weekly_scores(team_histories: Dict[int, dict], team_names: Dict[int, str]) -> pd.DataFrame:
    """Get weekly scores for all teams."""
    all_data = []

    for team_id, history in team_histories.items():
        team_name = team_names.get(team_id, f"Team {team_id}")
        current = history.get("current", [])

        for gw_data in current:
            gw = gw_data.get("event")
            pts = gw_data.get("points", 0)

            if gw:
                all_data.append({
                    "Team": team_name,
                    "Gameweek": gw,
                    "Points": pts
                })

    return pd.DataFrame(all_data)


def plot_scoring_distribution(weekly_scores: pd.DataFrame) -> Optional[go.Figure]:
    """Create box plot of scoring distribution."""
    if weekly_scores.empty:
        return None

    # Sort by median
    team_order = weekly_scores.groupby("Team")["Points"].median().sort_values(ascending=False).index.tolist()

    fig = px.box(
        weekly_scores,
        x="Team",
        y="Points",
        color="Team",
        title="Gameweek Scoring Distribution",
        category_orders={"Team": team_order}
    )

    fig.update_layout(
        **_DARK_CHART_LAYOUT,
        showlegend=False,
        xaxis_tickangle=-45,
    )
    fig.update_yaxes(title="Gameweek Points")

    return fig


# ---------------------------
# MAIN PAGE
# ---------------------------

# ---------------------------
# RIVAL TRACKER
# ---------------------------

# Rival squads are last-deadline data -- FPL publishes no picks for a gameweek
# whose deadline has not passed -- so they only change when a deadline does.
# Hence a long TTL, which is safe only because `allow_auth=False` below makes it
# impossible for a `my-team` payload to enter this cache: CLAUDE.md's rule is
# that my-team is never cached beyond 60s, since it changes the instant the
# manager makes a transfer.
_RIVAL_SQUAD_TTL = 900
_OWN_SQUAD_TTL = 60


def _squad_entry(res) -> dict:
    """The fields the tab needs out of a SquadResolution."""
    return {
        "picks": res.picks,
        "is_stale": bool(res.is_stale),
        "source_gw": res.source_gw,
        "source": res.source,
        "provenance": res.provenance,
    }


@st.cache_data(ttl=_RIVAL_SQUAD_TTL, show_spinner=False)
def _fetch_rival_squads(team_ids: tuple, current_gw: int) -> dict:
    """Resolve the other managers' squads.

    Cached, and keyed on plain ints only -- the bootstrap is fetched *inside* so
    a large dict never has to be hashed into the cache key.

    ``allow_auth=False`` is passed explicitly rather than relying on
    ``resolve_classic_squad``'s "is this the user's own team" default. The
    default would already be False for every rival, but stating it is what makes
    the long TTL above provably safe, and it keeps a credential from ever being
    spent on someone else's behalf.

    A team that cannot be resolved is *omitted* rather than given an empty
    squad, so the count reaching ``check_mini_league_ownership()`` is the real
    one and a shortfall is reported rather than silently changing every
    percentage's denominator.
    """
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        return {}

    resolved = {}
    for team_id in team_ids:
        try:
            res = resolve_classic_squad(int(team_id), bootstrap, current_gw,
                                        allow_auth=False)
        except Exception:
            _logger.warning("Could not resolve squad for team %s", team_id, exc_info=True)
            continue
        if not res.ok:
            _logger.info("No squad for team %s: %s", team_id, res.provenance)
            continue
        resolved[int(team_id)] = _squad_entry(res)
    return resolved


@st.cache_data(ttl=_OWN_SQUAD_TTL, show_spinner=False)
def _fetch_own_squad(team_id: int, current_gw: int) -> dict:
    """Resolve the user's own squad, on a short TTL.

    Separate from the rival loop because this one may come back live from
    ``my-team``, which must never be cached for long: it changes the moment the
    manager makes a transfer. ``allow_auth`` is left at its default, which
    resolves to True for the user's own configured team -- so with credentials
    set, your own row is live while the rivals' are last-deadline, and the tab
    says so.
    """
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap or not team_id:
        return {}
    try:
        res = resolve_classic_squad(int(team_id), bootstrap, current_gw)
    except Exception:
        _logger.warning("Could not resolve own squad %s", team_id, exc_info=True)
        return {}
    return {int(team_id): _squad_entry(res)} if res.ok else {}


def _resolve_league_squads(team_ids, current_gw, my_team_id) -> dict:
    """Every analysed squad, own team included even when outside the subset.

    The user's team can be configured but not a member of the selected league,
    or truncated out by the team slider -- in which case "Yours" would be
    uniformly empty, which reads as "you own nothing" rather than "we do not
    know who you are".
    """
    mine = int(my_team_id) if my_team_id else None
    rivals = tuple(int(t) for t in team_ids if mine is None or int(t) != mine)
    squads = dict(_fetch_rival_squads(rivals, current_gw))
    if mine is not None:
        squads.update(_fetch_own_squad(mine, current_gw))
    return squads


def _ownership_display(df: pd.DataFrame, n_squads: int) -> pd.DataFrame:
    """The ownership frame as the columns a human reads."""
    out = pd.DataFrame({
        "Player": df["Display_Name"],
        "Pos": df["Position"],
        "Club": df["Team"],
        "Owned By": df["Owners"].astype(int).astype(str) + f"/{n_squads}",
        "Own %": df["Own_Pct"],
        "EO %": df["EO_Pct"],
        "Capt": df["Captains"].astype(int),
        "Yours": df["Mine"].map({True: "✓", False: ""}),
        "Proj": df["Proj"],
        "Next 3": df["Proj_Next3"],
    })
    return out.reset_index(drop=True)


def _render_rival_tracker(ownership: pd.DataFrame,
                          totals: pd.DataFrame,
                          n_squads: int,
                          my_team_id,
                          squad_meta: dict,
                          proj_range: tuple = None):
    """Render the Rival Tracker tab.

    A separate function from the page body so it can be driven directly by a
    test with a populated fixture -- the page's own smoke test early-returns
    long before reaching any tab body, which is the convention the Draft Power
    Rankings tests follow.
    """
    st.subheader("Rival Tracker")
    st.caption(
        f"Ownership across the {n_squads} squad(s) analysed — not global "
        "ownership. A haul from a player most of your league also owns moves "
        "you nowhere."
    )

    if ownership is None or ownership.empty:
        st.info("Could not resolve any squads for this league.")
        return

    # Rival squads are last-deadline by construction: FPL publishes no picks for
    # a gameweek whose deadline has not passed, and my-team is the user's own
    # credential. Saying so is the point -- every value in a stale squad is
    # individually plausible, it is simply the wrong fifteen.
    stale = [tid for tid, meta in (squad_meta or {}).items() if meta.get("is_stale")]
    if stale:
        source_gws = sorted({meta.get("source_gw") for meta in squad_meta.values()
                             if meta.get("is_stale") and meta.get("source_gw")})
        gw_note = f" (as of GW{source_gws[0]})" if len(source_gws) == 1 else ""
        st.warning(
            f"⚠️ {len(stale)} of {n_squads} squads are last-deadline{gw_note}. "
            "FPL does not publish a rival's transfers or captain until the "
            "deadline passes, so their moves this week are not reflected."
        )

    have_mine = bool(ownership["Mine"].any())

    # --- Summary cards ---
    tmpl = template_score(ownership)
    diffs = differentials(ownership)
    gaps = template_gaps(ownership)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_stat_card(
            "Your Squad EO",
            f"{tmpl:.0f}%" if tmpl is not None else "—",
            "mean effective ownership" if tmpl is not None else "your team not in this league",
            accent="#60a5fa",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card(
            "Differentials",
            f"{len(diffs)}" if have_mine else "—",
            f"owned by ≤{DEFAULT_DIFFERENTIAL_MAX_OWN:.0f}% of the league",
            accent="#00ff87",
        ), unsafe_allow_html=True)
    with c3:
        top_gap = gaps.iloc[0] if not gaps.empty else None
        st.markdown(_stat_card(
            "Biggest Exposure",
            top_gap["Display_Name"] if top_gap is not None else "—",
            f"{top_gap['EO_Pct']:.0f}% EO, you don't own him" if top_gap is not None
            else "no high-ownership gaps",
            accent="#ff6b6b",
        ), unsafe_allow_html=True)
    with c4:
        best_diff = diffs.iloc[0] if not diffs.empty else None
        st.markdown(_stat_card(
            "Best Differential",
            best_diff["Display_Name"] if best_diff is not None else "—",
            f"{best_diff['Proj']:.1f} proj, {best_diff['Own_Pct']:.0f}% owned"
            if best_diff is not None and pd.notna(best_diff["Proj"]) else "—",
            accent="#00ff87",
        ), unsafe_allow_html=True)

    st.markdown("")

    # Colour every projection against the whole player pool, not against the
    # handful of rows shown: the weakest of five hand-picked differentials is
    # not a bad projection, and scaled against only its peers it renders red.
    overrides = {"Proj": proj_range} if proj_range else {}
    fmts = {"Own %": "{:.0f}%", "EO %": "{:.0f}%", "Proj": "{:.1f}", "Next 3": "{:.1f}"}
    aligns = {"Pos": "center", "Club": "center", "Owned By": "center",
              "Own %": "center", "EO %": "center", "Capt": "center",
              "Yours": "center", "Proj": "center", "Next 3": "center"}

    if have_mine:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Your differentials")
            st.caption("Yours, and most of the league's is not. This is where you gain.")
            render_styled_table(
                _ownership_display(diffs.head(15), n_squads).drop(columns=["Yours"]),
                col_formats=fmts, text_align=aligns,
                positive_color_cols=["Proj"], color_range_overrides=overrides,
            )
        with right:
            st.markdown("##### Template you're missing")
            st.caption("The league owns them heavily, you don't. This is where you lose.")
            render_styled_table(
                _ownership_display(gaps.head(15), n_squads).drop(columns=["Yours"]),
                col_formats=fmts, text_align=aligns,
                positive_color_cols=["Proj"], color_range_overrides=overrides,
            )
        st.markdown("")
    else:
        st.info(
            "Set your Classic team on the 🆔 League Setup page (or raise the "
            "team limit above so it is included) to see your differentials and "
            "the template you are missing."
        )

    # --- Full league ownership ---
    st.markdown("##### League ownership")
    st.caption(
        "Effective ownership weights by multiplier, so a captained player "
        "counts double — which is why it can exceed 100%. Your own players are "
        "accented."
    )
    display = _ownership_display(ownership, n_squads)
    render_styled_table(
        display,
        col_formats=fmts, text_align=aligns,
        positive_color_cols=["EO %", "Proj"],
        color_range_overrides=overrides,
        highlight_row=lambda row: row["Yours"] == "✓",
        max_height=600,
    )

    # --- The league's week ahead ---
    if totals is not None and not totals.empty:
        st.divider()
        st.markdown("##### The league's week ahead")
        st.caption(
            "Each squad's projected total, applying its own lineup and captain "
            "(bench counts 0, captain double). Rival lineups are the ones above "
            "— last-deadline where flagged."
        )

        tdisp = pd.DataFrame({
            "#": range(1, len(totals) + 1),
            "Manager": totals["Manager"],
            "Projected": totals["Proj_Total"],
            "vs You": totals["Gap_To_You"],
        })
        render_styled_table(
            tdisp,
            col_formats={"Projected": "{:.1f}", "vs You": "{:+.1f}"},
            text_align={"#": "center", "Projected": "center", "vs You": "center"},
            positive_color_cols=["Projected"],
            highlight_row=lambda row: bool(
                totals.loc[totals["Manager"] == row["Manager"], "Mine"].any()),
        )

        fig = go.Figure(go.Bar(
            x=totals["Manager"],
            y=totals["Proj_Total"],
            marker_color=["#00ff87" if m else "#60a5fa" for m in totals["Mine"]],
            hovertemplate="%{x}: %{y:.1f} projected<extra></extra>",
        ))
        fig.update_layout(
            **_DARK_CHART_LAYOUT,
            title="Projected gameweek total",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def show_classic_league_analysis_page():
    """Display the Classic League Analysis page."""

    st.title("Classic League Analysis")

    # Get configured leagues
    league_options = get_league_display_options()

    if not league_options:
        st.warning("No Classic leagues configured.")
        st.info(
            "Add leagues to `FPL_CLASSIC_LEAGUE_IDS` in your `.env` file:\n\n"
            "```\nFPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League\n```"
        )
        return

    # League selector
    if "classic_analysis_league_index" not in st.session_state:
        st.session_state.classic_analysis_league_index = 0

    display_options = [opt["display"] for opt in league_options]
    selected_display = st.selectbox(
        "Select League",
        options=display_options,
        index=st.session_state.classic_analysis_league_index,
        key="classic_analysis_league_selector"
    )

    new_index = display_options.index(selected_display)
    if new_index != st.session_state.classic_analysis_league_index:
        st.session_state.classic_analysis_league_index = new_index

    selected_league = league_options[st.session_state.classic_analysis_league_index]
    league_id = selected_league["id"]

    # Fetch standings to get team list
    with st.spinner("Loading league data..."):
        standings_data = get_league_standings(league_id)

    if not standings_data:
        show_api_error(f"loading data for league {league_id}", hint_key="classic_league_stale")
        return

    league_info = standings_data.get("league", {})
    standings = standings_data.get("standings", {})
    results = standings.get("results", [])

    if not results:
        st.info("No standings data available yet.")
        return

    # Build team info
    team_ids = [r.get("entry") for r in results if r.get("entry")]
    team_names = {r.get("entry"): r.get("entry_name", f"Team {r.get('entry')}") for r in results}

    # Slider to limit teams for performance
    num_teams = len(team_ids)
    if num_teams > 10:
        teams_to_analyze = st.slider(
            "Number of teams to analyze",
            min_value=5,
            max_value=min(num_teams, 30),
            value=min(num_teams, 15),
            help="Analyzing more teams takes longer but provides broader insights"
        )
        team_ids = team_ids[:teams_to_analyze]
        team_names = {k: v for k, v in team_names.items() if k in team_ids}

    # Fetch team histories
    with st.spinner(f"Loading data for {len(team_ids)} teams..."):
        team_histories = fetch_team_histories_for_league(team_ids)

    if not team_histories:
        st.info("Could not load team history data.")
        return

    # League header
    league_name = league_info.get("name", "Classic League")
    st.markdown(f"### {league_name}")

    current_gw = get_current_gameweek()
    if current_gw:
        st.caption(f"Analysis through Gameweek {current_gw}")

    st.divider()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Chip Usage",
        "Rank Movement",
        "Points Analysis",
        "Team Values",
        "Performance Stats",
        "Points by Position",
        "Bench Analysis",
        "🎯 Rival Tracker"
    ])

    # ---------------------------
    # TAB 1: CHIP USAGE
    # ---------------------------
    with tab1:
        st.subheader("Chip Usage Analysis")

        chip_df = analyze_chip_usage(team_histories, team_names)

        if not chip_df.empty:
            render_styled_table(
                chip_df,
                text_align={"Wildcard": "center", "Free Hit": "center",
                             "Bench Boost": "center", "Triple Captain": "center",
                             "Total Used": "center"},
            )

            st.divider()

            fig = plot_chip_timing(team_histories, team_names)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No chips have been played yet.")
        else:
            st.info("No chip data available.")

    # ---------------------------
    # TAB 2: RANK MOVEMENT
    # ---------------------------
    with tab2:
        st.subheader("Rank Movement Analysis")

        rank_df = build_rank_progression(team_histories, team_names)

        if not rank_df.empty:
            # Rank changes summary
            changes_df = calculate_rank_changes(rank_df)

            if not changes_df.empty:
                st.markdown("**Biggest Movers (Overall Rank)**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("*Most Improved*")
                    top_movers = changes_df.head(3)
                    for _, row in top_movers.iterrows():
                        change = row["Rank_Change"]
                        arrow = "+" if change > 0 else ""
                        st.markdown(
                            _stat_card(row["Team"], f"#{row['Current_Rank']:,}",
                                       detail=f"{arrow}{change:,} places"),
                            unsafe_allow_html=True,
                        )

                with col2:
                    st.markdown("*Biggest Drops*")
                    bottom_movers = changes_df.tail(3).iloc[::-1]
                    for _, row in bottom_movers.iterrows():
                        change = row["Rank_Change"]
                        arrow = "+" if change > 0 else ""
                        st.markdown(
                            _stat_card(row["Team"], f"#{row['Current_Rank']:,}",
                                       detail=f"{arrow}{change:,} places", accent="#f87171"),
                            unsafe_allow_html=True,
                        )

            st.divider()

            fig = plot_rank_progression(rank_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rank data available.")

    # ---------------------------
    # TAB 3: POINTS ANALYSIS
    # ---------------------------
    with tab3:
        st.subheader("Points Behind Leader")

        points_df = calculate_points_behind_leader(team_histories, team_names)

        if not points_df.empty:
            # Current standings summary
            latest_gw = points_df["Gameweek"].max()
            latest_df = points_df[points_df["Gameweek"] == latest_gw].sort_values("Points_Behind")

            st.markdown(f"**Current Standings (GW{latest_gw})**")

            display_latest = latest_df[["Team", "Total_Points", "Points_Behind"]].copy()
            display_latest.columns = ["Team", "Total Points", "Behind Leader"]

            render_styled_table(
                display_latest,
                col_formats={"Total Points": "{:,.0f}", "Behind Leader": "{:,.0f}"},
                negative_color_cols=["Behind Leader"],
            )

            st.divider()

            fig = plot_points_behind_leader(points_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No points data available.")

    # ---------------------------
    # TAB 4: TEAM VALUES
    # ---------------------------
    with tab4:
        st.subheader("Team Value Comparison")

        value_df = analyze_team_values(team_histories, team_names)

        if not value_df.empty:
            render_styled_table(
                value_df,
                col_formats={"Squad Value": "£{:.1f}m", "Bank": "£{:.1f}m",
                              "Total Value": "£{:.1f}m", "Value Gain": "{:+.1f}"},
                positive_color_cols=["Value Gain"],
            )

            st.divider()

            fig = plot_value_comparison(value_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No team value data available.")

    # ---------------------------
    # TAB 5: PERFORMANCE STATS
    # ---------------------------
    with tab5:
        st.subheader("Gameweek Performance Statistics")

        perf_df = analyze_gameweek_performance(team_histories, team_names)

        if not perf_df.empty:
            perf_display = perf_df.rename(columns={
                "Games": "GP", "Total_Points": "Total", "Avg_Points": "Avg/GW",
                "Best_GW": "Best", "Worst_GW": "Worst", "Std_Dev": "Std Dev",
            })
            render_styled_table(
                perf_display,
                col_formats={"Avg/GW": "{:.1f}", "Std Dev": "{:.1f}"},
                positive_color_cols=["Total", "Avg/GW"],
            )

            st.divider()

            weekly_scores = get_weekly_scores(team_histories, team_names)
            fig = plot_scoring_distribution(weekly_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available.")

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

        # Determine latest played gameweek from team histories
        latest_gw = None
        for _tid, _hist in team_histories.items():
            gw_list = _hist.get("current", [])
            if gw_list:
                last_event = gw_list[-1].get("event")
                if last_event and (latest_gw is None or last_event > latest_gw):
                    latest_gw = last_event

        if latest_gw is None:
            st.info("Could not determine latest played gameweek.")
        else:
            with st.spinner(f"Loading position data for {len(team_ids)} teams (GW1-{latest_gw})..."):
                pos_rows = []
                for tid in team_ids:
                    tname = team_names.get(tid, f"Team {tid}")
                    result = get_classic_team_position_data(tid, latest_gw)
                    positions = result["positions"]
                    total = sum(positions.values())
                    pos_rows.append({
                        "Team": tname,
                        "GK": positions["GK"],
                        "DEF": positions["DEF"],
                        "MID": positions["MID"],
                        "FWD": positions["FWD"],
                        "Total": total,
                    })

            pos_df = pd.DataFrame(pos_rows)

            if pos_df.empty:
                st.info("No position data available yet.")
            else:
                pos_df = pos_df.sort_values("Total", ascending=False).reset_index(drop=True)

                # Toggle between raw points and percentage
                view_mode = st.radio(
                    "View",
                    ["Raw Points", "Percentage"],
                    horizontal=True,
                    key="classic_league_pos_view"
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
                avg_df_row = pd.DataFrame([avg_row])
                table_df = pd.concat([display_df, avg_df_row], ignore_index=True)

                # Build dark-themed HTML table with green/red text coloring
                suffix = "%" if view_mode == "Percentage" else " pts"

                def _pos_text_color(val, col):
                    """Green for above avg, red for below avg, grey for average."""
                    avg_val = avg_row[col]
                    col_vals = table_df[table_df["Team"] != "League Average"][col]
                    col_min, col_max = col_vals.min(), col_vals.max()
                    if val > avg_val and col_max > avg_val:
                        t = min((val - avg_val) / (col_max - avg_val), 1.0)
                        r = int(156 + t * (74 - 156))
                        g = int(163 + t * (222 - 163))
                        b = int(175 + t * (128 - 175))
                        return f"rgb({r},{g},{b})"
                    elif val < avg_val and col_min < avg_val:
                        t = min((avg_val - val) / (avg_val - col_min), 1.0)
                        r = int(156 + t * (248 - 156))
                        g = int(163 + t * (113 - 163))
                        b = int(175 + t * (113 - 175))
                        return f"rgb({r},{g},{b})"
                    return "#9ca3af"

                th_style = ("background:linear-gradient(135deg,#37003c,#5a0060);color:#00ff87;"
                            "font-weight:600;font-size:13px;padding:10px 12px;border-bottom:2px solid #00ff87;"
                            "text-align:center;")
                team_th = th_style.replace("text-align:center", "text-align:left")

                parts = ['<div style="border:1px solid #333;border-radius:10px;overflow:hidden;margin-bottom:1rem;">',
                          '<table style="width:100%;border-collapse:collapse;font-size:14px;background:#1a1a2e;">',
                          "<thead><tr>"]
                for c in table_df.columns:
                    s = team_th if c == "Team" else th_style
                    parts.append(f'<th style="{s}">{c}</th>')
                parts.append("</tr></thead><tbody>")

                for row_idx, (_, row) in enumerate(table_df.iterrows()):
                    is_avg = row["Team"] == "League Average"
                    row_bg = "background:rgba(255,255,255,0.03);" if row_idx % 2 == 1 else "background:#1a1a2e;"
                    if is_avg:
                        row_bg = "background:#16213e;border-top:2px solid #00ff87;"
                    parts.append(f'<tr style="{row_bg}">')
                    for c in table_df.columns:
                        val = row[c]
                        if c == "Team":
                            fw = "font-weight:800;" if is_avg else "font-weight:600;"
                            color = "#00ff87" if is_avg else "#e0e0e0"
                            parts.append(f'<td style="padding:8px 12px;color:{color};{fw}border-bottom:1px solid #333;">{val}</td>')
                        elif c in pos_cols or c == "Total":
                            txt_color = "#00ff87" if is_avg else _pos_text_color(float(val), c) if c in pos_cols else "#e0e0e0"
                            fw = "font-weight:700;" if is_avg else "font-weight:600;"
                            formatted = f"{float(val):.1f}{suffix}"
                            parts.append(f'<td style="padding:8px 12px;color:{txt_color};{fw}text-align:center;border-bottom:1px solid #333;">{formatted}</td>')
                        else:
                            parts.append(f'<td style="padding:8px 12px;color:#e0e0e0;text-align:center;border-bottom:1px solid #333;">{val}</td>')
                    parts.append("</tr>")
                parts.append("</tbody></table></div>")
                st.markdown("".join(parts), unsafe_allow_html=True)

                st.divider()

                # Charts side-by-side: bar chart + pie chart
                bar_df = pos_df.melt(
                    id_vars=["Team"],
                    value_vars=pos_cols,
                    var_name="Position",
                    value_name="Points"
                )

                league_totals = {col: pos_df[col].sum() for col in pos_cols}
                pie_df = pd.DataFrame({
                    "Position": list(league_totals.keys()),
                    "Points": list(league_totals.values())
                })

                col_bar, col_pie = st.columns([3, 2])

                with col_bar:
                    fig_bar = px.bar(
                        bar_df, x="Team", y="Points", color="Position",
                        title="Points by Position (Team Breakdown)",
                        barmode="stack", color_discrete_map=POSITION_COLORS,
                        category_orders={"Position": pos_cols}
                    )
                    fig_bar.update_layout(
                        **_DARK_CHART_LAYOUT,
                        xaxis_tickangle=-45,
                        margin=dict(b=80),
                        height=450,
                    )
                    fig_bar.update_layout(
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13),
                                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig_bar.update_xaxes(title="")
                    fig_bar.update_yaxes(title="Total Points")
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_pie:
                    fig_pie = px.pie(
                        pie_df, values="Points", names="Position",
                        title="League-Wide Distribution",
                        color="Position", color_discrete_map=POSITION_COLORS,
                    )
                    fig_pie.update_traces(textinfo="percent+label")
                    fig_pie.update_layout(
                        paper_bgcolor="#1a1a2e",
                        font=dict(color="#ffffff", size=14),
                        title_font=dict(size=18, color="#ffffff"),
                        title_x=0.5, title_xanchor="center",
                        showlegend=False,
                        height=450,
                        margin=dict(t=60, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------------------
    # TAB 7: BENCH ANALYSIS
    # ---------------------------
    with tab7:
        st.subheader("Bench Analysis")
        st.caption("How effectively does each manager choose their starting lineup?")

        import json
        with st.spinner("Analyzing bench decisions across the league..."):
            league_bench = compute_classic_league_bench_data(
                tuple(team_ids),
                json.dumps({str(k): v for k, v in team_names.items()}),
                current_gw,
            )

        if league_bench:
            render_league_bench_analysis(league_bench, is_classic=True)
        else:
            st.info("Not enough data for bench analysis.")

    # ---------------------------
    # TAB 8: RIVAL TRACKER
    # ---------------------------
    with tab8:
        # Behind a button because `st.tabs` renders every body on every rerun,
        # and resolving a 20-team league is ~11s of network. Without the gate,
        # someone opening this page to look at Chip Usage pays for a tab they
        # never opened -- the same cost the Projections Hub removed by making
        # its sections lazy. The Draft Helper's "Scan" button is the precedent.
        loaded_key = f"rival_tracker_loaded_{league_id}"
        show = bool(st.session_state.get(loaded_key))

        if st.button("Load rival squads", key=f"load_rivals_{league_id}",
                     type="primary" if not show else "secondary"):
            st.session_state[loaded_key] = True
            show = True

        if not show:
            st.subheader("Rival Tracker")
            st.caption(
                "Who in your league owns what — differentials, effective "
                "ownership, and each rival's projected total."
            )
            st.info(
                f"Resolving all {len(team_ids)} squads takes a few seconds, so "
                "it is loaded on request rather than on every page view."
            )
        else:
            my_team_id = config.FPL_CLASSIC_TEAM_ID
            with st.spinner("Resolving squads across the league..."):
                squad_meta = _resolve_league_squads(team_ids, current_gw, my_team_id)
                squads = {tid: meta["picks"] for tid, meta in squad_meta.items()}
                pool = blended_pool(current_gw, config.ROTOWIRE_URL)
                ownership = compute_league_ownership(squads, pool, my_team_id)
                totals = projected_team_totals(squads, pool, team_names, my_team_id)

            issues = check_mini_league_ownership(ownership, squads, len(team_ids))
            if issues:
                # Logged rather than raised: a degraded answer on this tab must
                # not take down a page whose other seven tabs are fine.
                _logger.warning("Rival Tracker plausibility: %s", format_issues(issues))

            proj_range = None
            if pool is not None and not pool.empty and "Proj" in pool.columns:
                proj_max = pd.to_numeric(pool["Proj"], errors="coerce").max()
                if pd.notna(proj_max) and proj_max > 0:
                    proj_range = (0.0, float(proj_max))

            _render_rival_tracker(ownership, totals, len(squads), my_team_id,
                                  squad_meta, proj_range)
