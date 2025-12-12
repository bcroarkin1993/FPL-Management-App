# main.py
import streamlit as st
import config

# --- Existing pages you already have ---
from scripts.draft.home import show_home_page  # Draft home
from scripts.fpl.fixtures import show_club_fixtures_section  # Global fixtures page
from scripts.draft.fixture_projections import show_fixture_projections_page  # Draft fixture projections
from scripts.draft.team_analysis import show_team_stats_page  # Draft team analysis
from scripts.draft.waiver_wire import show_waiver_wire_page  # Draft - waiver wire
from scripts.draft.draft_helper import show_draft_helper_page # Draft - Draft Helper
from scripts.fpl.player_statistics import show_player_stats_page  # Global player stats
from scripts.fpl.player_projections import show_player_projections_page  # Global player projections
from scripts.fpl.projected_lineups import show_projected_lineups  # Global projected lineups
from scripts.fpl.injuries import show_injuries_page  # Global projected lineups
from scripts.classic.free_hit import show_free_hit_page # Classic Free Hit page

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
# Main
# ------------------------------------------------------------
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.image("images/fpl_logo1.jpeg", use_column_width=True)
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
                "Player Projections",
                "Player Statistics",
                "Player Injuries"
            ],
        )

        if subpage == "Home":
            render_app_home()
        elif subpage == "Gameweek Fixtures":
            show_club_fixtures_section()
        elif subpage == "Projected Lineups":
            show_projected_lineups()
        elif subpage == "Player Projections":
            show_player_projections_page()
        elif subpage == "Player Statistics":
            show_player_stats_page()
        elif subpage == "Player Injuries":
            show_injuries_page()

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
            show_fixture_projections_page()
        elif subpage == "Waiver Wire":
            show_waiver_wire_page()
        elif subpage == "Team Analysis":
            show_team_stats_page()
        elif subpage == "League Analysis":
            # If you have a dedicated draft league analysis page, call it here.
            # Otherwise, show a placeholder:
            render_placeholder("Draft — League Analysis")
        elif subpage == "Draft Helper":
            show_draft_helper_page()

    else:  # Classic
        subpage = st.sidebar.radio(
            "Classic Pages",
            [
                "Home",
                "Fixture Projections",
                "Waiver Wire",
                "Team Analysis",
                "League Analysis",
                "Free Hit Optimizer"
            ],
        )

        if subpage == "Home":
            render_classic_home()
        elif subpage == "Fixture Projections":
            # Replace with your Classic fixture projections function when ready
            render_placeholder("Classic — Fixture Projections")
        elif subpage == "Waiver Wire":
            # Replace with your Classic waiver/free transfers helper when ready
            render_placeholder("Classic — Waiver/Transfers")
        elif subpage == "Team Analysis":
            # Replace with your Classic team analysis when ready
            render_placeholder("Classic — Team Analysis")
        elif subpage == "League Analysis":
            # Replace with your Classic league analysis when ready
            render_placeholder("Classic — League Analysis")
        elif subpage == "Free Hit Optimizer":
            show_free_hit_page()


if __name__ == "__main__":
    main()