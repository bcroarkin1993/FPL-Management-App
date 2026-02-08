"""
FPL Projections Hub

A unified projections page that aggregates data from multiple sources:
- Rotowire Player Projections (point projections)
- Fantasy Football Pundit Points Predictor (alternative projections with start %)
- FFP Goal Scorer & Assist Odds (betting market probabilities)
- FFP Clean Sheet Odds (team clean sheet probabilities)
- The Odds API Match Odds (match-level betting data)

Each data source is displayed in its own tab with clear attribution.
"""

import config
import pandas as pd
import streamlit as st
from scripts.common.utils import (
    get_rotowire_player_projections,
    get_rotowire_rankings_url,
    get_ffp_projections_data,
    get_ffp_points_predictor,
    get_ffp_goalscorer_odds,
    get_ffp_clean_sheet_odds,
    get_odds_api_match_odds,
)


# =============================================================================
# Data Source Banners
# =============================================================================

def _render_source_banner(source: str, description: str, color: str, border_color: str, url: str = None):
    """Render a styled data source attribution banner."""
    link_html = f'<a href="{url}" target="_blank">{source}</a>' if url else source
    st.markdown(f"""
    <div style="background: {color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>Data Source:</strong> {link_html}<br>
        <small>{description}</small>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Rotowire Projections Tab
# =============================================================================

def rotowire_url_selector():
    """Allow user to override the auto-detected Rotowire URL."""
    auto_url = get_rotowire_rankings_url()

    with st.expander("Rotowire URL Settings", expanded=False):
        if auto_url:
            st.success(f"Auto-detected: {auto_url}")
        else:
            st.warning("Could not auto-detect this week's rankings article.")

        manual_url = st.text_input(
            "Override URL (optional)",
            value=auto_url or "",
            placeholder="https://www.rotowire.com/soccer/article/...",
            help="Paste the Rotowire rankings article URL here if auto-detect fails."
        )

        return manual_url.strip() or auto_url


def render_rotowire_projections():
    """Render the Rotowire player projections tab."""
    _render_source_banner(
        "Rotowire",
        "Weekly gameweek projections based on expert analysis, matchups, and form.",
        "#f0f9ff", "#0ea5e9",
        "https://www.rotowire.com/soccer/"
    )

    # Get projections
    if config.ROTOWIRE_URL:
        url = config.ROTOWIRE_URL
    else:
        url = rotowire_url_selector()

    if not url:
        st.info("No Rotowire URL available. Please configure one above.")
        return

    player_projections = get_rotowire_player_projections(url)

    if player_projections is None or player_projections.empty:
        st.warning("Could not load Rotowire projections. The URL may be invalid or the page structure changed.")
        return

    # Prepare data
    display_cols = ['Player', 'Team', 'Position', 'Pos Rank', 'Matchup', 'TSB %', 'Points', 'Price']
    available_cols = [c for c in display_cols if c in player_projections.columns]
    player_projections = player_projections[available_cols].copy()

    # Add value column if we have Points and Price
    if 'Points' in player_projections.columns and 'Price' in player_projections.columns:
        player_projections['Value'] = (player_projections['Points'] / player_projections['Price']).round(2)

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)

        num_players = col1.slider(
            "Number of players to display",
            min_value=10, max_value=300, value=100, step=10,
            key="rw_num_players"
        )

        player_filter = col2.text_input("Search by name", placeholder="e.g., Salah", key="rw_player_filter")

        col3, col4 = st.columns(2)

        if 'Position' in player_projections.columns:
            all_positions = player_projections['Position'].dropna().unique().tolist()
            position_filter = col3.multiselect("Position", options=all_positions, default=all_positions, key="rw_pos")
        else:
            position_filter = None

        if 'Price' in player_projections.columns:
            min_price = float(player_projections['Price'].min())
            max_price = float(player_projections['Price'].max())
            price_filter = col4.slider("Max price", min_value=min_price, max_value=max_price, value=max_price, step=0.5, key="rw_price")
        else:
            price_filter = None

    # Apply filters
    df = player_projections.head(num_players)

    if player_filter:
        df = df[df['Player'].str.contains(player_filter, case=False, na=False)]

    if position_filter and 'Position' in df.columns:
        df = df[df['Position'].isin(position_filter)]

    if price_filter and 'Price' in df.columns:
        df = df[df['Price'] <= price_filter]

    # Display
    st.subheader(f"GW{config.CURRENT_GAMEWEEK} Player Projections")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(f"Showing {len(df)} players. Data from Rotowire.")


# =============================================================================
# Fantasy Football Pundit Points Predictor Tab
# =============================================================================

def render_ffp_points_predictor():
    """Render the FFP points predictor tab."""
    _render_source_banner(
        "Fantasy Football Pundit",
        "Points predictions with start probability, multi-GW forecasts, and ownership data.",
        "#fef3c7", "#f59e0b",
        "https://www.fantasyfootballpundit.com/fpl-points-predictor/"
    )

    df = get_ffp_points_predictor()

    if df is None or df.empty:
        st.warning("Could not load FFP points predictor data. The data source may be temporarily unavailable.")
        return

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)

        num_players = col1.slider(
            "Number of players to display",
            min_value=10, max_value=300, value=100, step=10,
            key="ffp_num_players"
        )

        player_filter = col2.text_input("Search by name", placeholder="e.g., Salah", key="ffp_player_filter")

        col3, col4 = st.columns(2)

        if 'Position' in df.columns:
            all_positions = df['Position'].dropna().unique().tolist()
            position_filter = col3.multiselect("Position", options=all_positions, default=all_positions, key="ffp_pos")
        else:
            position_filter = None

        if 'Start %' in df.columns:
            min_start = col4.slider("Min start %", min_value=0, max_value=100, value=10, step=5, key="ffp_start")
        else:
            min_start = 0

    # Apply filters
    result = df.head(num_players)

    if player_filter and 'Player' in result.columns:
        result = result[result['Player'].str.contains(player_filter, case=False, na=False)]

    if position_filter and 'Position' in result.columns:
        result = result[result['Position'].isin(position_filter)]

    if min_start > 0 and 'Start %' in result.columns:
        result = result[result['Start %'] >= min_start]

    # Display
    st.subheader(f"GW{config.CURRENT_GAMEWEEK} Points Predictions")

    # Format for display
    display_df = result.copy()
    if 'Ownership %' in display_df.columns:
        display_df['Ownership %'] = display_df['Ownership %'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    if 'Start %' in display_df.columns:
        display_df['Start %'] = display_df['Start %'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(f"Showing {len(result)} players. Data from Fantasy Football Pundit.")

    # Comparison note
    with st.expander("About FFP vs Rotowire"):
        st.markdown("""
        **Key differences:**
        - **Start %**: FFP includes probability of player starting (accounting for rotation/injury risk)
        - **Pts (if starts)**: Points prediction assuming the player starts
        - **Multi-GW forecasts**: See expected points over next 2-6 gameweeks
        - **Ownership %**: Current FPL ownership for differential picks

        Both sources use different methodologies. Compare predictions to find consensus picks
        or identify divergences where one source sees value others miss.
        """)


# =============================================================================
# Goal Scorer & Assist Odds Tab
# =============================================================================

def render_goalscorer_odds():
    """Render the goal scorer and assist odds tab."""
    _render_source_banner(
        "Fantasy Football Pundit (Betting Odds)",
        "Anytime goalscorer and assist probabilities converted from bookmaker odds.",
        "#fefce8", "#eab308",
        "https://www.fantasyfootballpundit.com/premier-league-goalscorer-assist-odds/"
    )

    df = get_ffp_goalscorer_odds()

    if df is None or df.empty:
        st.warning("Could not load goalscorer odds data. The data source may be temporarily unavailable.")
        return

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)

        num_players = col1.slider(
            "Number of players to display",
            min_value=10, max_value=200, value=50, step=10,
            key="gs_num_players"
        )

        player_filter = col2.text_input("Search by name", placeholder="e.g., Haaland", key="gs_player_filter")

        col3, col4 = st.columns(2)

        if 'Position' in df.columns:
            all_positions = df['Position'].dropna().unique().tolist()
            position_filter = col3.multiselect("Position", options=all_positions, default=all_positions, key="gs_pos")
        else:
            position_filter = None

        if 'Team' in df.columns:
            all_teams = sorted(df['Team'].dropna().unique().tolist())
            team_filter = col4.multiselect("Team", options=all_teams, default=[], key="gs_team",
                                           help="Leave empty for all teams")
        else:
            team_filter = []

    # Apply filters
    result = df.head(num_players)

    if player_filter and 'Player' in result.columns:
        result = result[result['Player'].str.contains(player_filter, case=False, na=False)]

    if position_filter and 'Position' in result.columns:
        result = result[result['Position'].isin(position_filter)]

    if team_filter and 'Team' in result.columns:
        result = result[result['Team'].isin(team_filter)]

    # Display
    st.subheader(f"GW{config.CURRENT_GAMEWEEK} Goalscorer & Assist Probabilities")

    # Format percentages
    display_df = result.copy()
    for col in ['Goal %', 'Assist %', 'Return %', 'Start %']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(f"Showing {len(result)} players. Data from Fantasy Football Pundit (betting odds converted to %).")

    # Usage tips
    with st.expander("How to use this data"):
        st.markdown("""
        **Column explanations:**
        - **Goal %**: Probability of scoring at least one goal (anytime goalscorer)
        - **Assist %**: Probability of registering at least one assist
        - **Return %**: Probability of either a goal OR assist (useful for attacking returns)
        - **Start %**: Probability player will start the match

        **FPL applications:**
        - **Captaincy**: High Goal % players are strong captain options
        - **Differentials**: Players with high Return % but low ownership
        - **Transfers**: Compare Return % to identify best attacking assets
        """)


# =============================================================================
# Clean Sheet Odds Tab
# =============================================================================

def render_clean_sheet_odds():
    """Render the clean sheet odds tab."""
    _render_source_banner(
        "Fantasy Football Pundit (Clean Sheet Odds)",
        "Team clean sheet probabilities from betting markets.",
        "#f0fdf4", "#22c55e",
        "https://www.fantasyfootballpundit.com/premier-league-clean-sheet-odds/"
    )

    df = get_ffp_clean_sheet_odds()

    if df is None or df.empty:
        st.warning("Could not load clean sheet odds data. The data source may be temporarily unavailable.")
        return

    # Display
    st.subheader(f"GW{config.CURRENT_GAMEWEEK} Clean Sheet Probabilities")

    # Format for display
    display_df = df.copy()
    if 'CS Prob %' in display_df.columns:
        display_df['CS Prob %'] = display_df['CS Prob %'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(f"Showing {len(df)} teams. Data from Fantasy Football Pundit.")

    # Multi-GW note
    st.info("**Tip**: For multi-gameweek clean sheet analysis, visit the FFP website directly for GW1-6 forecasts.")

    # Usage tips
    with st.expander("How to use this data"):
        st.markdown("""
        **For defenders and goalkeepers:**
        - Higher CS % = more likely to earn clean sheet points (4 pts for DEF/GK)
        - Consider fixture difficulty when selecting defensive assets
        - Stack defenders from teams with high CS probability

        **Combining with other data:**
        - DEF with high CS % AND high assist odds = premium picks
        - Budget GKs from high CS % teams = value options
        """)


# =============================================================================
# Match Odds Tab (The Odds API)
# =============================================================================

def render_match_odds():
    """Render the match betting odds tab from The Odds API."""
    import os
    api_key = os.getenv("ODDS_API_KEY", "")

    _render_source_banner(
        "The Odds API",
        "Match betting odds aggregated from UK bookmakers (h2h markets).",
        "#ede9fe", "#8b5cf6",
        "https://the-odds-api.com"
    )

    if not api_key:
        st.warning("**ODDS_API_KEY not configured.** Add your API key to `.env` to enable match odds.")
        st.markdown("""
        Get a free API key at [the-odds-api.com](https://the-odds-api.com) (500 requests/month free tier).

        Add to your `.env` file:
        ```
        ODDS_API_KEY=your_api_key_here
        ```
        """)
        return

    df = get_odds_api_match_odds(api_key)

    if df is None or df.empty:
        st.warning("Could not load match odds data. The API may be temporarily unavailable or rate limited.")
        return

    # Display
    st.subheader(f"GW{config.CURRENT_GAMEWEEK} Match Odds")

    # Format percentages
    display_df = df.copy()
    for col in ['Home Win %', 'Draw %', 'Away Win %']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(f"Showing {len(df)} matches. Data from The Odds API (UK bookmakers average).")

    # Usage tips
    with st.expander("How to use this data"):
        st.markdown("""
        **Match odds indicate expected outcomes:**
        - **Home Win %**: Market probability of home team winning
        - **Draw %**: Market probability of a draw
        - **Away Win %**: Market probability of away team winning

        **FPL applications:**
        - Teams favored to win are more likely to score (good for attackers)
        - Underdogs may struggle to keep clean sheets
        - High draw % games may be low-scoring (consider defensive picks)

        **Note**: Probabilities may sum to >100% due to bookmaker margin.
        """)


# =============================================================================
# Main Page
# =============================================================================

def show_player_projections_page():
    """Main projections hub page with tabbed interface."""
    st.title("Projections Hub")
    st.caption("Player and team projections from multiple data sources to inform your FPL decisions.")

    # Create tabs for different data sources
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Rotowire",
        "🎯 FFP Predictor",
        "⚽ Goal/Assist Odds",
        "🧤 Clean Sheet Odds",
        "📈 Match Odds"
    ])

    with tab1:
        render_rotowire_projections()

    with tab2:
        render_ffp_points_predictor()

    with tab3:
        render_goalscorer_odds()

    with tab4:
        render_clean_sheet_odds()

    with tab5:
        render_match_odds()

    # Footer with data source summary
    st.markdown("---")
    with st.expander("About Data Sources"):
        st.markdown("""
        ### Data Sources Used

        | Source | Type | Status | Update Frequency |
        |--------|------|--------|------------------|
        | **Rotowire** | Player point projections | ✅ Active | Weekly (before each GW) |
        | **FFP Predictor** | Points with start % | ✅ Active | Updated throughout GW |
        | **FFP Goal/Assist Odds** | Betting probabilities | ✅ Active | Daily |
        | **FFP Clean Sheet Odds** | Team CS probabilities | ✅ Active | Daily |
        | **The Odds API** | Match betting odds | ✅ Active | Live updates |

        ### How to Use This Data

        - **Rotowire**: Expert rankings based on analysis and matchups
        - **FFP Predictor**: Alternative projections accounting for rotation risk
        - **Goal/Assist Odds**: Identify players most likely to return attacking points
        - **Clean Sheet Odds**: Find defenders/GKs with best CS potential
        - **Match Odds**: Understand expected match outcomes for context

        ### Comparing Sources

        When Rotowire and FFP agree on a player, that's a high-confidence pick.
        When they diverge, dig deeper to understand why (injury doubt, rotation risk, etc.).
        """)
