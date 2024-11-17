# main.py
import streamlit as st
from scripts.home import show_home_page
from scripts.fixture_projections import show_fixtures_page
from scripts.team_analysis import show_team_stats_page
from scripts.waiver_wire import show_waiver_wire_page
from scripts.player_statistics import show_league_stats_page
from scripts.player_projections import show_player_projections_page
from scripts.projected_lineups import show_projected_lineups

def apply_custom_styles():
    st.markdown(
        f"""
        <style>
        /* Sidebar background color */
        [data-testid="stSidebar"] {{
            background-color: #00FFFF;  /* Aqua */
        }}

        /* Sidebar font color */
        [data-testid="stSidebar"] .css-1d391kg {{
            color: #FF005F;  /* Magenta */
        }}

        /* Adjust font size in the sidebar */
        [data-testid="stSidebar"] .css-1d391kg, h1, h2, h3, h4 {{
            font-size: 16px;
        }}

        /* Main background color */
        .main {{
            background-color: #FFFFFF;  /* White */
        }}

        /* Header text color */
        h1, h2, h3 {{
            color: #3C005F;  /* Dark Purple */
        }}

        /* Accent color (for buttons, links, etc.) */
        .css-1q8dd3e a {{
            color: #00FF66;  /* Bright Green */
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.sidebar.title("Navigation")

    # Sidebar image
    st.sidebar.image('images/fpl_logo1.jpeg', use_column_width=True)

    # Apply the custom CSS styles
    apply_custom_styles()

    # Create the Sidebar navigation screen
    app_mode = st.sidebar.radio("Choose Page", ["Home", "EPL Projected Lineups", "Player Projections",
                                                "FPL Fixture Projections", "Waiver Wire", "Team Analysis",
                                                "League Statistics"])

    if app_mode == "Home":
        show_home_page()
    elif app_mode == "FPL Fixture Projections":
        show_fixtures_page()
    elif app_mode == "EPL Projected Lineups":
        show_projected_lineups()
    elif app_mode == "Player Projections":
        show_player_projections_page()
    elif app_mode == "Waiver Wire":
        show_waiver_wire_page()
    elif app_mode == "League Statistics":
        show_league_stats_page()
    elif app_mode == "Team Analysis":
        show_team_stats_page()

if __name__ == "__main__":
    main()
