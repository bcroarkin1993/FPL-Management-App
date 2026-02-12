# main.py
import os
import streamlit as st
import config

from scripts.common.utils import (
    get_fpl_player_mapping,
    get_league_entries,
    get_classic_bootstrap_static,
    get_rotowire_player_projections,
    get_league_player_ownership,
)

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
        }
        [data-testid="stSidebar"] * {
            color: #ffffff;
            font-size: 16px;
        }
        /* Nav link hover (FPL cyan) */
        [data-testid="stSidebar"] a:hover {
            color: #04f5ff !important;
        }
        /* Active nav link (FPL green) */
        [data-testid="stSidebar"] [aria-selected="true"] {
            color: #00ff87 !important;
            font-weight: 600;
        }
        /* Section headers in sidebar */
        [data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {
            color: #00ff87 !important;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 13px;
            letter-spacing: 1px;
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
# Small helpers for new sections
# ------------------------------------------------------------
def render_app_home():
    st.title("FPL Manager — App Home")
    st.markdown(
        """
        Welcome to your one-stop **FPL Manager** app covering both **Draft** and **Classic** formats.

        **What you'll find here:**
        - Cross-format tools (fixtures, projected lineups, player projections, player statistics)
        - Dedicated **Draft** and **Classic** hubs
        - Visuals, analytics, and utilities to help you dominate your leagues
        """
    )

    # (Optional) quick glance at IDs if present
    with st.expander("Configured IDs (from .env / config)"):
        st.write({
            "Draft League ID": getattr(config, "FPL_DRAFT_LEAGUE_ID", None),
            "Classic League ID": getattr(config, "FPL_CLASSIC_LEAGUE_ID", None),
            "My Team ID (Draft)": getattr(config, "FPL_TEAM_ID", None),
        })

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

    # Sidebar logo and title
    st.sidebar.title("FPL Manager")
    logo_path = "images/fpl_logo1.jpeg"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_column_width=True)
    apply_custom_styles()

    # Navigation using st.navigation() with grouped sections
    pages = {
        "FPL App Home": [
            st.Page(render_app_home, title="Home", icon="🏠"),
            st.Page(show_club_fixtures_section, title="Gameweek Fixtures", icon="📅"),
            st.Page(show_projected_lineups, title="Projected Lineups", icon="📋"),
            st.Page(show_player_projections_page, title="Projections Hub", icon="📊"),
            st.Page(show_player_stats_page, title="Player Statistics", icon="📈"),
            st.Page(show_injuries_page, title="Player Injuries", icon="🏥"),
            st.Page(show_settings_page, title="Alert Settings", icon="⚙️"),
        ],
        "Draft": [
            st.Page(show_home_page, title="Home", icon="🏠"),
            st.Page(show_fixtures_page, title="Fixture Projections", icon="📅"),
            st.Page(show_waiver_wire_page, title="Waiver Wire", icon="🔄"),
            st.Page(show_team_stats_page, title="Team Analysis", icon="👥"),
            st.Page(show_draft_league_analysis_page, title="League Analysis", icon="🏆"),
            st.Page(show_draft_helper_page, title="Draft Helper", icon="📝"),
        ],
        "Classic": [
            st.Page(show_classic_home_page, title="Home", icon="🏠"),
            st.Page(show_classic_fixture_projections_page, title="Fixture Projections", icon="📅"),
            st.Page(show_classic_transfers_page, title="Transfer Suggestions", icon="🔄"),
            st.Page(show_free_hit_page, title="Free Hit Optimizer", icon="⚡"),
            st.Page(show_wildcard_page, title="Wildcard Optimizer", icon="🃏"),
            st.Page(show_classic_team_analysis_page, title="Team Analysis", icon="👥"),
            st.Page(show_classic_league_analysis_page, title="League Analysis", icon="🏆"),
        ],
    }
    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
