# main.py
import logging
import os

import requests
import streamlit as st

import config
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_classic_league_standings,
    get_draft_league_details,
    get_fpl_player_mapping,
    get_h2h_league_standings,
    get_league_entries,
    get_league_player_ownership,
    get_rotowire_player_projections,
    is_gameweek_live,
)

_logger = logging.getLogger(__name__)

# --- Draft pages ---
from scripts.draft.home import show_home_page
from scripts.draft.fixture_projections import show_fixtures_page
from scripts.draft.team_analysis import show_team_stats_page
from scripts.draft.waiver_wire import show_waiver_wire_page
from scripts.draft.draft_helper import show_draft_helper_page
from scripts.draft.league_analysis import show_draft_league_analysis_page

# --- FPL cross-format pages ---
from scripts.fpl.fixtures import show_club_fixtures_section
from scripts.fpl.player_statistics import show_player_stats_page
from scripts.fpl.player_projections import show_player_projections_page
from scripts.fpl.projected_lineups import show_projected_lineups
from scripts.fpl.injuries import show_injuries_page
from scripts.fpl.settings import show_settings_page

# --- Classic pages ---
from scripts.classic.home import show_classic_home_page
from scripts.classic.team_analysis import show_classic_team_analysis_page
from scripts.classic.fixture_projections import show_classic_fixture_projections_page
from scripts.classic.transfers import show_classic_transfers_page
from scripts.classic.free_hit import show_free_hit_page
from scripts.classic.wildcard import show_wildcard_page
from scripts.classic.league_analysis import show_classic_league_analysis_page

# ------------------------------------------------------------
# Page config (must be first Streamlit command in the script)
# ------------------------------------------------------------
st.set_page_config(
    page_title="FPL Manager — Draft & Classic",
    page_icon="⚽",
    layout="wide",
)

# ------------------------------------------------------------
# FPL-themed CSS
# ------------------------------------------------------------
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Sidebar: FPL deep purple */
        [data-testid="stSidebar"] {
            background-color: #37003c;
            min-width: 260px;
            padding-top: 1.5rem;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff;
            font-size: 16px;
        }
        /* Radio buttons in sidebar: remove default bullet styling */
        [data-testid="stSidebar"] .stRadio label {
            cursor: pointer;
            padding: 2px 0;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            color: #04f5ff !important;
        }
        /* Active/selected radio option (FPL green) */
        [data-testid="stSidebar"] .stRadio [data-checked="true"] label {
            color: #00ff87 !important;
            font-weight: 600;
        }
        /* Section dividers */
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2);
        }
        /* Headings in main area */
        .main h1, .main h2, .main h3 {
            color: #37003c;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# Dashboard home page
# ------------------------------------------------------------
def _dashboard_css():
    """All CSS for the dashboard landing page, injected once globally."""
    return """
    <style>
    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #37003c 0%, #5a0060 50%, #37003c 100%);
        border-radius: 12px; padding: 1.2rem 1.8rem; margin-bottom: 1rem;
        color: white; position: relative; overflow: hidden;
    }
    .hero-banner::before {
        content: ''; position: absolute; top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(0,255,135,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner h1 { color: #ffffff !important; font-size: 1.8rem; margin: 0 0 0.2rem 0; }
    .hero-banner .hero-sub { color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 0; }
    .hero-badges { display: flex; gap: 10px; margin-top: 0.7rem; flex-wrap: wrap; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 600;
    }
    .badge-gw { background: rgba(0,255,135,0.2); color: #00ff87; border: 1px solid rgba(0,255,135,0.4); }
    .badge-live {
        background: rgba(0,255,135,0.25); color: #00ff87; border: 1px solid rgba(0,255,135,0.5);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    .badge-upcoming { background: rgba(4,245,255,0.15); color: #04f5ff; border: 1px solid rgba(4,245,255,0.4); }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 4px rgba(0,255,135,0.3); }
        50% { box-shadow: 0 0 12px rgba(0,255,135,0.6); }
    }
    /* Section headers */
    .section-header { display: flex; align-items: center; gap: 6px; margin: 0.3rem 0 0.6rem 0; font-size: 1.1rem; font-weight: 700; color: #37003c; }
    .section-icon { font-size: 1.2rem; }
    /* Fixture cards */
    .fixture-card {
        display: flex; align-items: stretch; justify-content: space-between;
        border-radius: 8px; padding: 8px 12px; margin-bottom: 5px;
        border-left: 4px solid transparent;
    }
    .fixture-ft { background: #f0f0f0; border-left-color: #bbb; }
    .fixture-live { background: linear-gradient(135deg, rgba(0,255,135,0.08), rgba(0,255,135,0.03)); border-left-color: #00cc6a; }
    .fixture-upcoming-card { background: #fff; border: 1px solid #e8e8e8; border-left: 4px solid #37003c; }
    .fixture-side { display: flex; flex-direction: column; justify-content: flex-start; flex: 1; }
    .fixture-side-home { align-items: flex-end; text-align: right; }
    .fixture-side-away { align-items: flex-start; text-align: left; }
    .fixture-team-row { display: flex; align-items: center; gap: 6px; min-height: 24px; }
    .fixture-side-home .fixture-team-row { justify-content: flex-end; }
    .fixture-side-away .fixture-team-row { justify-content: flex-start; }
    .fixture-team { font-weight: 700; font-size: 0.88rem; }
    .fixture-badge-img { width: 18px; height: 18px; object-fit: contain; }
    .fixture-win { color: #1a8a41; }
    .fixture-loss { color: #d32f2f; }
    .fixture-draw { color: #999; }
    .fixture-upcoming { color: #333; }
    .fixture-scorers { font-size: 0.72rem; color: #888; min-height: 16px; }
    .fixture-center { display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-width: 65px; padding: 0 6px; }
    .fixture-score { font-weight: 800; font-size: 0.9rem; padding: 3px 10px; border-radius: 6px; text-align: center; }
    .score-live { background: linear-gradient(135deg, #00ff87, #02efaa); color: #37003c; }
    .score-vs { background: #37003c; color: #ffffff; }
    .score-ft { background: #ddd; color: #333; }
    .fixture-status { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; margin-top: 2px; letter-spacing: 0.5px; }
    .status-ft { color: #999; }
    .status-live { color: #00a858; }
    .status-upcoming { color: #37003c; }
    /* League cards */
    .league-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(55,0,60,0.04); }
    .league-card-header { font-weight: 700; font-size: 0.95rem; color: #37003c; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 2px solid rgba(55,0,60,0.08); }
    .standings-row { display: flex; align-items: center; padding: 7px 12px; border-radius: 8px; margin-bottom: 5px; border: 1px solid #e8e8e8; }
    .standings-row-1 { background: linear-gradient(135deg, rgba(255,215,0,0.12), rgba(255,215,0,0.05)); border-color: rgba(255,215,0,0.3); }
    .standings-row-2 { background: linear-gradient(135deg, rgba(192,192,192,0.12), rgba(192,192,192,0.05)); border-color: rgba(192,192,192,0.3); }
    .standings-row-3 { background: linear-gradient(135deg, rgba(205,127,50,0.12), rgba(205,127,50,0.05)); border-color: rgba(205,127,50,0.3); }
    .my-team-row { background: linear-gradient(135deg, rgba(0,255,135,0.12), rgba(0,255,135,0.05)) !important; border: 2px solid rgba(0,255,135,0.5) !important; }
    .my-team-row .standings-name::after { content: ' ★'; color: #00cc6a; }
    .league-separator { text-align: center; color: #bbb; font-size: 0.8rem; padding: 1px 0; letter-spacing: 3px; }
    .standings-rank { font-size: 1.05rem; min-width: 28px; text-align: center; }
    .standings-name { flex: 1; font-weight: 600; color: #37003c; font-size: 0.85rem; }
    .standings-record { color: #666; font-size: 0.75rem; margin-right: 8px; }
    .standings-pts { font-weight: 800; color: #37003c; font-size: 0.88rem; background: rgba(0,255,135,0.15); padding: 2px 8px; border-radius: 10px; }
    /* In-form players */
    .performer-row { display: flex; align-items: center; padding: 5px 10px; border-radius: 8px; margin-bottom: 4px; background: #fafafa; border: 1px solid #e8e8e8; }
    .performer-rank { font-weight: 800; color: #37003c; min-width: 24px; font-size: 0.9rem; }
    .performer-name { flex: 1; font-weight: 600; color: #37003c; font-size: 0.88rem; }
    .pos-badge { display: inline-block; padding: 1px 6px; border-radius: 5px; font-size: 0.7rem; font-weight: 700; margin-right: 8px; min-width: 32px; text-align: center; }
    .pos-gk  { background: #f0c040; color: #333; }
    .pos-def { background: #4caf50; color: white; }
    .pos-mid { background: #2196f3; color: white; }
    .pos-fwd { background: #e91e63; color: white; }
    .form-badge { font-weight: 800; padding: 2px 10px; border-radius: 10px; font-size: 0.82rem; }
    .form-hot  { background: linear-gradient(135deg, #ff6b35, #ff4500); color: white; }
    .form-warm { background: linear-gradient(135deg, #ffa726, #ff9800); color: white; }
    .form-ok   { background: #e0e0e0; color: #333; }
    .team-badge { color: #888; font-size: 0.78rem; margin-right: 8px; min-width: 32px; }
    .pts-label { color: #888; font-size: 0.78rem; margin-left: 8px; }
    /* Injury watchlist */
    .injury-row { display: flex; align-items: center; padding: 5px 10px; border-radius: 8px; margin-bottom: 4px; border: 1px solid #e8e8e8; }
    .injury-out { background: rgba(244,67,54,0.06); border-color: rgba(244,67,54,0.2); }
    .injury-doubtful { background: rgba(255,152,0,0.06); border-color: rgba(255,152,0,0.2); }
    .injury-name { flex: 1; font-weight: 600; color: #37003c; font-size: 0.88rem; }
    .status-badge { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 0.72rem; font-weight: 700; }
    .status-out { background: #f44336; color: white; }
    .status-doubtful { background: #ff9800; color: white; }
    .injury-news { color: #888; font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .injury-info { display: flex; flex-direction: column; justify-content: center; flex: 1; }
    .injury-meta { display: flex; align-items: center; gap: 6px; }
    </style>
    """


def render_app_home():
    gw = config.CURRENT_GAMEWEEK

    # Inject all dashboard CSS once at the top
    st.markdown(_dashboard_css(), unsafe_allow_html=True)

    # ── Hero Banner ────────────────────────────────────────────────
    _render_hero_banner(gw)

    # ── Section 2: Fixtures + League snapshots ────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        _render_fixtures(gw)
    with col_right:
        _render_league_snapshots()

    # ── Section 3: In-form players + Injury watchlist ────────────────
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        _render_top_performers()
    with col_right2:
        _render_injury_watchlist()


def _render_hero_banner(gw):
    """Render a styled hero banner with GW number and live/upcoming badge."""
    try:
        live = is_gameweek_live(gw)
    except Exception:
        _logger.warning("Could not determine gameweek live status")
        live = False

    if live:
        status_badge = '<span class="hero-badge badge-live">🟢 LIVE</span>'
    else:
        status_badge = '<span class="hero-badge badge-upcoming">📅 Upcoming</span>'

    st.markdown(
        f'<div class="hero-banner">'
        f'<h1>⚽ FPL Manager Dashboard</h1>'
        f'<p class="hero-sub">Your command center for Draft &amp; Classic FPL</p>'
        f'<div class="hero-badges">'
        f'<span class="hero-badge badge-gw">🏟️ Gameweek {gw}</span>'
        f'{status_badge}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_fixtures(gw):
    """Show this GW's Premier League fixtures with status and result indicators."""
    st.markdown(
        '<div class="section-header"><span class="section-icon">📅</span> This Week\'s Fixtures</div>',
        unsafe_allow_html=True,
    )
    try:
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap:
            st.caption("Could not load fixture data.")
            return

        team_map = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
        team_code_map = {t["id"]: t["code"] for t in bootstrap["teams"]}
        player_map = {p["id"]: p["web_name"] for p in bootstrap.get("elements", [])}

        resp = requests.get(
            f"https://fantasy.premierleague.com/api/fixtures/?event={gw}",
            timeout=15,
        )
        resp.raise_for_status()
        fixtures = resp.json()

        if not fixtures:
            st.caption("No fixtures found for this gameweek.")
            return

        badge_url = "https://resources.premierleague.com/premierleague/badges/25/t{code}.png"

        cards_html = ""
        for fix in fixtures:
            home = team_map.get(fix.get("team_h"), "?")
            away = team_map.get(fix.get("team_a"), "?")
            h_code = team_code_map.get(fix.get("team_h"), 0)
            a_code = team_code_map.get(fix.get("team_a"), 0)
            h_badge = badge_url.format(code=h_code)
            a_badge = badge_url.format(code=a_code)
            h_score = fix.get("team_h_score")
            a_score = fix.get("team_a_score")

            # Extract goal scorers from stats
            h_scorers, a_scorers = [], []
            for stat in fix.get("stats", []):
                if stat.get("identifier") == "goals_scored":
                    for g in stat.get("h", []):
                        name = player_map.get(g["element"], "?")
                        h_scorers.append(name if g["value"] == 1 else f"{name} ({g['value']})")
                    for g in stat.get("a", []):
                        name = player_map.get(g["element"], "?")
                        a_scorers.append(name if g["value"] == 1 else f"{name} ({g['value']})")

            h_scorers_text = ", ".join(h_scorers) if h_scorers else "&nbsp;"
            a_scorers_text = ", ".join(a_scorers) if a_scorers else "&nbsp;"

            if fix.get("finished"):
                score_text = f"{h_score} - {a_score}"
                score_cls = "score-ft"
                card_cls = "fixture-ft"
                status_html = '<span class="fixture-status status-ft">FT</span>'
                if h_score > a_score:
                    home_cls, away_cls = "fixture-win", "fixture-loss"
                elif a_score > h_score:
                    home_cls, away_cls = "fixture-loss", "fixture-win"
                else:
                    home_cls, away_cls = "fixture-draw", "fixture-draw"
            elif fix.get("started"):
                score_text = f"{h_score or 0} - {a_score or 0}"
                score_cls = "score-live"
                card_cls = "fixture-live"
                status_html = '<span class="fixture-status status-live">LIVE</span>'
                home_cls = away_cls = "fixture-upcoming"
            else:
                score_text = "vs"
                score_cls = "score-vs"
                card_cls = "fixture-upcoming-card"
                status_html = '<span class="fixture-status status-upcoming">Upcoming</span>'
                home_cls = away_cls = "fixture-upcoming"

            cards_html += (
                f'<div class="fixture-card {card_cls}">'
                f'<div class="fixture-side fixture-side-home">'
                f'<div class="fixture-team-row">'
                f'<span class="fixture-team {home_cls}">{home}</span>'
                f'<img class="fixture-badge-img" src="{h_badge}" alt="{home}">'
                f'</div>'
                f'<div class="fixture-scorers">{h_scorers_text}</div></div>'
                f'<div class="fixture-center">'
                f'<span class="fixture-score {score_cls}">{score_text}</span>'
                f'{status_html}</div>'
                f'<div class="fixture-side fixture-side-away">'
                f'<div class="fixture-team-row">'
                f'<img class="fixture-badge-img" src="{a_badge}" alt="{away}">'
                f'<span class="fixture-team {away_cls}">{away}</span>'
                f'</div>'
                f'<div class="fixture-scorers">{a_scorers_text}</div></div>'
                f'</div>'
            )
        st.markdown(cards_html, unsafe_allow_html=True)
    except Exception:
        _logger.warning("Could not load fixtures", exc_info=True)
        st.caption("Could not load fixtures.")


def _render_league_snapshots():
    """Show abbreviated standings for all configured leagues."""
    st.markdown(
        '<div class="section-header"><span class="section-icon">🏆</span> My Leagues</div>',
        unsafe_allow_html=True,
    )

    has_leagues = False
    leagues_html = ""

    # Draft league
    draft_league_id = getattr(config, "FPL_DRAFT_LEAGUE_ID", None)
    if draft_league_id:
        try:
            html = _build_draft_snapshot(draft_league_id)
            if html:
                has_leagues = True
                leagues_html += html
        except Exception:
            _logger.warning("Could not load draft league snapshot", exc_info=True)

    # Classic / H2H leagues
    classic_leagues = getattr(config, "FPL_CLASSIC_LEAGUE_IDS", [])
    if isinstance(classic_leagues, list):
        for league_info in classic_leagues:
            try:
                league_id = league_info.get("id") if isinstance(league_info, dict) else int(league_info)
                league_name = league_info.get("name") if isinstance(league_info, dict) else None
                if league_id:
                    html = _build_classic_snapshot(league_id, league_name)
                    if html:
                        has_leagues = True
                        leagues_html += html
            except Exception:
                _logger.warning("Could not load classic league snapshot", exc_info=True)

    if has_leagues:
        st.markdown(leagues_html, unsafe_allow_html=True)
    else:
        st.caption("No leagues configured. Set league IDs in .env")


def _build_league_card_html(league_name, all_rows, icon):
    """Build HTML for a league snapshot card.

    Shows top 3 if user is in top 3 (highlighted), otherwise top 2 + separator
    + user's actual position (highlighted).

    all_rows: list of (rank, team_name, pts_display, record_or_none, is_me)
    """
    rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}

    my_row = None
    for row in all_rows:
        if row[4]:
            my_row = row
            break

    my_rank = my_row[0] if my_row else None

    if my_rank and my_rank <= 3:
        display_rows = all_rows[:3]
        show_separator = False
    elif my_row:
        display_rows = all_rows[:2]
        show_separator = True
    else:
        display_rows = all_rows[:3]
        show_separator = False

    def _row_html(rank, team_name, pts_display, record, is_me):
        rank_icon = rank_icons.get(rank, f"<b>{rank}</b>")
        row_class = f"standings-row-{rank}" if rank <= 3 and not is_me else ""
        highlight = "my-team-row" if is_me else ""
        record_part = f'<span class="standings-record">{record}</span>' if record else ""
        return (
            f'<div class="standings-row {row_class} {highlight}">'
            f'<span class="standings-rank">{rank_icon}</span>'
            f'<span class="standings-name">{team_name}</span>'
            f'{record_part}'
            f'<span class="standings-pts">{pts_display}</span>'
            f'</div>'
        )

    rows_html = "".join(_row_html(*row) for row in display_rows)
    if show_separator and my_row:
        rows_html += '<div class="league-separator">• • •</div>'
        rows_html += _row_html(*my_row)

    return (
        f'<div class="league-card">'
        f'<div class="league-card-header">{icon} {league_name}</div>'
        f'{rows_html}'
        f'</div>'
    )


def _build_draft_snapshot(league_id):
    """Build HTML for a draft league snapshot card."""
    details = get_draft_league_details(league_id)
    if not details:
        return None

    league_name = details.get("league", {}).get("name", "Draft League")
    entries_map = {e["id"]: e for e in details.get("league_entries", [])}
    standings = details.get("standings", [])
    if not standings:
        return None

    my_team_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    my_league_entry_id = None
    if my_team_id:
        for entry in details.get("league_entries", []):
            if entry.get("entry_id") == my_team_id:
                my_league_entry_id = entry.get("id")
                break

    all_rows = []
    for s in standings:
        entry_id = s.get("league_entry")
        entry = entries_map.get(entry_id, {})
        team_name = entry.get("entry_name", f"Team {entry_id}")
        w, d, l_ = s.get("matches_won", 0), s.get("matches_drawn", 0), s.get("matches_lost", 0)
        pts = s.get("total", 0)
        rank = s.get("rank", 0)
        is_me = (entry_id == my_league_entry_id)
        all_rows.append((rank, team_name, str(pts), f"{w}W {d}D {l_}L", is_me))

    return _build_league_card_html(league_name, all_rows, "📋")


def _build_classic_snapshot(league_id, league_name_override=None):
    """Build HTML for a classic or H2H league snapshot card.

    Tries classic standings first, falls back to H2H standings.
    """
    data = get_classic_league_standings(league_id)
    if not data:
        data = get_h2h_league_standings(league_id)
    if not data:
        return None

    league_name = league_name_override or data.get("league", {}).get("name", "League")
    results = data.get("standings", {}).get("results", [])
    if not results:
        return None

    my_team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    all_rows = []
    for r in results:
        rank = r.get("rank", 0)
        team_name = r.get("entry_name", "?")
        pts = r.get("total", 0)
        is_me = (r.get("entry") == my_team_id)
        all_rows.append((rank, team_name, f"{pts:,}", None, is_me))

    league_scoring = data.get("league", {}).get("scoring", "c")
    icon = "⚔️" if league_scoring == "h" else "🏆"
    return _build_league_card_html(league_name, all_rows, icon)


def _render_top_performers():
    """Show top 10 players by form with colored badges."""
    st.markdown(
        '<div class="section-header"><span class="section-icon">🔥</span> In-Form Players</div>',
        unsafe_allow_html=True,
    )
    try:
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap or not bootstrap.get("elements"):
            st.caption("Could not load player data.")
            return

        elements = bootstrap["elements"]
        team_map = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        pos_map = {1: ("GK", "pos-gk"), 2: ("DEF", "pos-def"),
                   3: ("MID", "pos-mid"), 4: ("FWD", "pos-fwd")}

        sorted_players = sorted(
            elements, key=lambda p: float(p.get("form", 0) or 0), reverse=True,
        )[:10]

        rows_html = ""
        for i, p in enumerate(sorted_players, 1):
            pos_label, pos_class = pos_map.get(p.get("element_type"), ("?", ""))
            team = team_map.get(p.get("team"), "?")
            form = float(p.get("form", 0) or 0)
            pts = p.get("total_points", 0)
            name = p.get("web_name", "?")

            form_class = "form-hot" if form >= 7.0 else "form-warm" if form >= 5.0 else "form-ok"

            rows_html += (
                f'<div class="performer-row">'
                f'<span class="performer-rank">{i}</span>'
                f'<span class="pos-badge {pos_class}">{pos_label}</span>'
                f'<span class="performer-name">{name}</span>'
                f'<span class="team-badge">{team}</span>'
                f'<span class="form-badge {form_class}">{form:.1f}</span>'
                f'<span class="pts-label">{pts} pts</span>'
                f'</div>'
            )

        st.markdown(rows_html, unsafe_allow_html=True)
    except Exception:
        _logger.warning("Could not load top performers", exc_info=True)
        st.caption("Could not load top performers.")


def _render_injury_watchlist():
    """Show players flagged as Out or Doubtful with status badges."""
    st.markdown(
        '<div class="section-header"><span class="section-icon">🏥</span> Injury Watchlist</div>',
        unsafe_allow_html=True,
    )
    try:
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap or not bootstrap.get("elements"):
            st.caption("Could not load player data.")
            return

        elements = bootstrap["elements"]
        team_map = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        pos_map = {1: ("GK", "pos-gk"), 2: ("DEF", "pos-def"),
                   3: ("MID", "pos-mid"), 4: ("FWD", "pos-fwd")}

        flagged = [p for p in elements if p.get("status") in ("i", "d", "s", "u", "n")
                    and p.get("news")]
        status_order = {"i": 0, "s": 0, "n": 0, "u": 0, "d": 1}
        flagged.sort(key=lambda p: (status_order.get(p.get("status"), 2),
                                      -p.get("total_points", 0)))
        flagged = flagged[:10]

        if not flagged:
            st.caption("No flagged players — all clear! ✅")
            return

        rows_html = ""
        for p in flagged:
            name = p.get("web_name", "?")
            team = team_map.get(p.get("team"), "?")
            pos_label, pos_class = pos_map.get(p.get("element_type"), ("?", ""))
            status = p.get("status", "?")
            news = (p.get("news") or "")[:60]

            if status in ("i", "s", "n", "u"):
                status_text, badge_cls, row_cls = "OUT", "status-out", "injury-out"
            else:
                status_text, badge_cls, row_cls = "DOUBT", "status-doubtful", "injury-doubtful"

            rows_html += (
                f'<div class="injury-row {row_cls}">'
                f'<span class="pos-badge {pos_class}" style="margin-right:10px">{pos_label}</span>'
                f'<div class="injury-info">'
                f'<div class="injury-meta">'
                f'<span class="injury-name">{name}</span>'
                f'<span class="team-badge">{team}</span>'
                f'<span class="status-badge {badge_cls}">{status_text}</span>'
                f'</div>'
                f'<div class="injury-news">{news}</div>'
                f'</div></div>'
            )

        st.markdown(rows_html, unsafe_allow_html=True)
    except Exception:
        _logger.warning("Could not load injury watchlist", exc_info=True)
        st.caption("Could not load injury watchlist.")

# ------------------------------------------------------------
# Page routing tables (label → function)
# ------------------------------------------------------------
FPL_PAGES = {
    "🏠  Home": render_app_home,
    "📅  Gameweek Fixtures": show_club_fixtures_section,
    "📋  Projected Lineups": show_projected_lineups,
    "📊  Projections Hub": show_player_projections_page,
    "📈  Player Statistics": show_player_stats_page,
    "🏥  Player Injuries": show_injuries_page,
    "⚙️  Alert Settings": show_settings_page,
}

DRAFT_PAGES = {
    "🏠  Home": show_home_page,
    "📅  Fixture Projections": show_fixtures_page,
    "🔄  Waiver Wire": show_waiver_wire_page,
    "👥  Team Analysis": show_team_stats_page,
    "🏆  League Analysis": show_draft_league_analysis_page,
    "📝  Draft Helper": show_draft_helper_page,
}

CLASSIC_PAGES = {
    "🏠  Home": show_classic_home_page,
    "📅  Fixture Projections": show_classic_fixture_projections_page,
    "🔄  Transfer Suggestions": show_classic_transfers_page,
    "⚡  Free Hit Optimizer": show_free_hit_page,
    "🃏  Wildcard Optimizer": show_wildcard_page,
    "👥  Team Analysis": show_classic_team_analysis_page,
    "🏆  League Analysis": show_classic_league_analysis_page,
}

SECTIONS = {
    "⚽  FPL App Home": FPL_PAGES,
    "📋  Draft": DRAFT_PAGES,
    "🏆  Classic": CLASSIC_PAGES,
}

# ------------------------------------------------------------
# Startup Preload - warm caches for faster page navigation
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading app data...")
def preload_app_data():
    """
    Preload commonly used data at app startup.

    Uses @st.cache_resource so this runs once per session and persists
    across page navigations. Individual functions use @st.cache_data
    which will be warm after this initial load.
    """
    data = {}

    # Core player data (used by almost every page)
    data['fpl_players'] = get_fpl_player_mapping()
    data['bootstrap_static'] = get_classic_bootstrap_static()

    # Draft league data (if configured)
    draft_league_id = getattr(config, 'FPL_DRAFT_LEAGUE_ID', None)
    if draft_league_id:
        data['league_entries'] = get_league_entries(draft_league_id)
        data['league_ownership'] = get_league_player_ownership(draft_league_id)

    # Rotowire projections (expensive scrape, used by multiple pages)
    try:
        rotowire_url = config.ROTOWIRE_URL
        if rotowire_url:
            data['rotowire_projections'] = get_rotowire_player_projections(rotowire_url)
    except Exception:
        pass  # Rotowire URL discovery may fail, that's ok

    return data

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # Preload data at startup for faster page navigation
    preload_app_data()

    # Sidebar: logo + title at the top
    logo_path = "static/fpl_logo1.jpeg"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.title("FPL Manager")
    apply_custom_styles()

    st.sidebar.divider()

    # Section selector
    section = st.sidebar.radio(
        "Section",
        list(SECTIONS.keys()),
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    # Page selector for the active section
    pages = SECTIONS[section]
    page = st.sidebar.radio(
        "Page",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    # Route to the selected page
    pages[page]()


if __name__ == "__main__":
    main()
