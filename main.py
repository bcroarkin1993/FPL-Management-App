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
# Optional: light CSS polish (keeps your existing palette)
# ------------------------------------------------------------
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Sidebar: background + font color */
        [data-testid="stSidebar"] {
            background-color: #00FFFF; /* Aqua */
        }
        [data-testid="stSidebar"] * {
            color: #FF005F; /* Magenta */
            font-size: 16px;
        }
        /* Headings in main area */
        .main h1, .main h2, .main h3 {
            color: #3C005F; /* Dark Purple */
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

        **What you’ll find here:**
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

def render_classic_home():
    st.title("Classic League — Home")
    st.markdown(
        """
        This is your **Classic** hub. Here you'll soon find:
        - **League table** & mini-leagues
        - **Team analysis** (transfers, captaincy, bank, etc.)
        - **Fixture projections** & **waiver/free transfers** helpers
        - **Season-long charts**

        *Note:* Some Classic-specific pages may still be under construction if you haven’t implemented
        their data sources yet. The navigation is wired and ready for you to plug in.
        """
    )

def render_placeholder(page_label: str):
    st.header(page_label)
    st.info("This section is coming soon. Wire in your Classic endpoints or reuse your Draft logic if applicable.")

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

    # Sidebar
    st.sidebar.title("Navigation")
    logo_path = "images/fpl_logo1.jpeg"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_column_width=True)
    apply_custom_styles()

    # Top-level section
    section = st.sidebar.selectbox(
        "Choose Section",
        ["FPL App Home", "Draft", "Classic"],
        index=0
    )

    # Sub-navigation based on section
    if section == "FPL App Home":
        subpage = st.sidebar.radio(
            "FPL App Pages",
            [
                "Home",
                "Gameweek Fixtures",
                "Projected Lineups",
                "Projections Hub",
                "Player Statistics",
                "Player Injuries",
                "Alert Settings"
            ],
        )

        if subpage == "Home":
            render_app_home()
        elif subpage == "Gameweek Fixtures":
            show_club_fixtures_section()
        elif subpage == "Projected Lineups":
            show_projected_lineups()
        elif subpage == "Projections Hub":
            show_player_projections_page()
        elif subpage == "Player Statistics":
            show_player_stats_page()
        elif subpage == "Player Injuries":
            show_injuries_page()
        elif subpage == "Alert Settings":
            show_settings_page()

    elif section == "Draft":
        subpage = st.sidebar.radio(
            "Draft Pages",
            [
                "Home",
                "Fixture Projections",
                "Waiver Wire",
                "Team Analysis",
                "League Analysis",
                "Draft Helper"
            ],
        )

        if subpage == "Home":
            # Your existing draft home
            show_home_page()
        elif subpage == "Fixture Projections":
            show_fixtures_page()
        elif subpage == "Waiver Wire":
            show_waiver_wire_page()
        elif subpage == "Team Analysis":
            show_team_stats_page()
        elif subpage == "League Analysis":
            show_draft_league_analysis_page()
        elif subpage == "Draft Helper":
            show_draft_helper_page()

    else:  # Classic
        subpage = st.sidebar.radio(
            "Classic Pages",
            [
                "Home",
                "Fixture Projections",
                "Transfer Suggestions",
                "Free Hit Optimizer",
                "Wildcard Optimizer",
                "Team Analysis",
                "League Analysis",
            ],
        )

        if subpage == "Home":
            show_classic_home_page()
        elif subpage == "Fixture Projections":
            show_classic_fixture_projections_page()
        elif subpage == "Transfer Suggestions":
            show_classic_transfers_page()
        elif subpage == "Free Hit Optimizer":
            show_free_hit_page()
        elif subpage == "Wildcard Optimizer":
            show_wildcard_page()
        elif subpage == "Team Analysis":
            show_classic_team_analysis_page()
        elif subpage == "League Analysis":
            show_classic_league_analysis_page()


if __name__ == "__main__":
    main()