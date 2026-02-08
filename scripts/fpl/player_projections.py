"""
FPL Projections Hub

A unified projections page that aggregates data from multiple sources:
- Rotowire Player Projections (point projections)
- Goal Scorer Odds (betting market probabilities)
- Clean Sheet Odds (team clean sheet probabilities)

Each data source is displayed in its own tab with clear attribution.
"""

import config
import pandas as pd
import streamlit as st
from scripts.common.utils import get_rotowire_player_projections, get_rotowire_rankings_url


# =============================================================================
# Rotowire Projections (existing functionality)
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
    st.markdown("""
    <div style="background: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>Data Source:</strong> <a href="https://www.rotowire.com/soccer/" target="_blank">Rotowire</a><br>
        <small>Weekly gameweek projections based on expert analysis, matchups, and form.</small>
    </div>
    """, unsafe_allow_html=True)

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
            min_value=10, max_value=300, value=100, step=10
        )

        player_filter = col2.text_input("Search by name", placeholder="e.g., Salah")

        col3, col4 = st.columns(2)

        if 'Position' in player_projections.columns:
            all_positions = player_projections['Position'].dropna().unique().tolist()
            position_filter = col3.multiselect("Position", options=all_positions, default=all_positions)
        else:
            position_filter = None

        if 'Price' in player_projections.columns:
            min_price = float(player_projections['Price'].min())
            max_price = float(player_projections['Price'].max())
            price_filter = col4.slider("Max price", min_value=min_price, max_value=max_price, value=max_price, step=0.5)
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
# Goal Scorer Odds (placeholder for betting data)
# =============================================================================

def render_goal_scorer_odds():
    """Render the goal scorer odds tab (placeholder for betting data integration)."""
    st.markdown("""
    <div style="background: #fefce8; border-left: 4px solid #eab308; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>Data Source:</strong> Coming Soon<br>
        <small>Anytime goalscorer odds from betting markets, converted to probabilities.</small>
    </div>
    """, unsafe_allow_html=True)

    st.info("**Goal Scorer Odds** - This section will display anytime goalscorer probabilities from betting markets.")

    # Placeholder structure showing what the data will look like
    st.subheader("Expected Data Format")

    st.markdown("""
    This section will show:
    - **Player name** and team
    - **Goal probability %** (converted from betting odds)
    - **Odds value** (decimal odds from bookmakers)
    - **Fixture** for the gameweek

    Example of how data will be displayed:
    """)

    # Sample placeholder data
    sample_data = pd.DataFrame({
        "Player": ["M. Salah", "E. Haaland", "C. Palmer", "B. Saka", "A. Isak"],
        "Team": ["LIV", "MCI", "CHE", "ARS", "NEW"],
        "Position": ["M", "F", "M", "M", "F"],
        "Fixture": ["LIV vs BOU (H)", "MCI vs EVE (H)", "CHE vs WOL (A)", "ARS vs NFO (H)", "NEW vs BRE (H)"],
        "Goal Prob %": ["68%", "72%", "52%", "48%", "55%"],
        "Odds": [1.47, 1.39, 1.92, 2.08, 1.82],
    })

    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("""
    ### Integration Notes

    To enable this feature, you'll need to configure a data source:

    1. **The Odds API** - Paid API with free tier ([the-odds-api.com](https://the-odds-api.com))
    2. **Custom scraper** - Scrape from OddsChecker or similar aggregator
    3. **Manual CSV upload** - Upload odds data manually

    Once configured, add your API key or data source in the `.env` file:
    ```
    ODDS_API_KEY=your_api_key_here
    ```
    """)


# =============================================================================
# Clean Sheet Odds (placeholder for betting data)
# =============================================================================

def render_clean_sheet_odds():
    """Render the clean sheet odds tab (placeholder for betting data integration)."""
    st.markdown("""
    <div style="background: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>Data Source:</strong> Coming Soon<br>
        <small>Team clean sheet probabilities from betting markets.</small>
    </div>
    """, unsafe_allow_html=True)

    st.info("**Clean Sheet Odds** - This section will display team clean sheet probabilities from betting markets.")

    st.subheader("Expected Data Format")

    st.markdown("""
    This section will show:
    - **Team** name
    - **Clean Sheet probability %** (converted from betting odds)
    - **Fixture** and venue (H/A)
    - **Opponent attacking strength** for context

    Useful for selecting **goalkeepers and defenders**.
    """)

    # Sample placeholder data
    sample_data = pd.DataFrame({
        "Team": ["Liverpool", "Arsenal", "Man City", "Chelsea", "Newcastle"],
        "Short": ["LIV", "ARS", "MCI", "CHE", "NEW"],
        "Fixture": ["vs BOU (H)", "vs NFO (H)", "vs EVE (H)", "vs WOL (A)", "vs BRE (H)"],
        "CS Prob %": ["52%", "48%", "55%", "38%", "42%"],
        "CS Odds": [1.92, 2.08, 1.82, 2.63, 2.38],
        "Opp Attack Rating": ["Weak", "Weak", "Weak", "Medium", "Medium"],
    })

    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("""
    ### Multi-Gameweek View

    Once enabled, you'll be able to view clean sheet probabilities across multiple gameweeks
    to help with transfer planning:

    | Team | GW25 | GW26 | GW27 | GW28 | Avg CS% |
    |------|------|------|------|------|---------|
    | LIV  | 52%  | 45%  | 38%  | 55%  | 47.5%   |
    | ARS  | 48%  | 52%  | 48%  | 42%  | 47.5%   |
    | MCI  | 55%  | 48%  | 52%  | 48%  | 50.8%   |

    This helps identify teams with favorable defensive fixtures over a horizon.
    """)


# =============================================================================
# Main Page
# =============================================================================

def show_player_projections_page():
    """Main projections hub page with tabbed interface."""
    st.title("Projections Hub")
    st.caption("Player and team projections from multiple data sources to inform your FPL decisions.")

    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs([
        "📊 Rotowire Projections",
        "⚽ Goal Scorer Odds",
        "🧤 Clean Sheet Odds"
    ])

    with tab1:
        render_rotowire_projections()

    with tab2:
        render_goal_scorer_odds()

    with tab3:
        render_clean_sheet_odds()

    # Footer with data source summary
    st.markdown("---")
    with st.expander("About Data Sources"):
        st.markdown("""
        ### Data Sources Used

        | Source | Type | Status | Update Frequency |
        |--------|------|--------|------------------|
        | **Rotowire** | Player point projections | ✅ Active | Weekly (before each GW) |
        | **Goal Scorer Odds** | Betting market probabilities | 🔜 Coming Soon | Daily |
        | **Clean Sheet Odds** | Team CS probabilities | 🔜 Coming Soon | Daily |

        ### How to Use This Data

        - **Rotowire Projections**: Best for overall player rankings and expected points
        - **Goal Scorer Odds**: Identify players most likely to score (great for captaincy)
        - **Clean Sheet Odds**: Find defenders/GKs with best CS potential

        ### Contributing Data Sources

        If you have access to additional projection sources or APIs, they can be integrated here.
        Check `scripts/fpl/player_projections.py` for the integration pattern.
        """)
