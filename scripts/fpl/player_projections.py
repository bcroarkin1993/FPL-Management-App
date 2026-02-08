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
import numpy as np
import pandas as pd
import streamlit as st
from scripts.common.utils import (
    get_rotowire_player_projections,
    get_rotowire_rankings_url,
    get_ffp_projections_data,
    get_ffp_goalscorer_odds,
    get_ffp_clean_sheet_odds,
    get_odds_api_match_odds,
)


# =============================================================================
# Color Gradient Styling
# =============================================================================

def get_gradient_color(value, col_min, col_max, high_is_good=True):
    """Return RGB color on green-white-red scale."""
    if pd.isna(value) or col_max == col_min:
        return "#f5f5f5"  # neutral gray

    # Normalize to 0-1
    normalized = (value - col_min) / (col_max - col_min)

    # Invert if low is good
    if not high_is_good:
        normalized = 1 - normalized

    # Red (#f8696b) -> White (#ffffff) -> Green (#63be7b)
    if normalized >= 0.5:
        t = (normalized - 0.5) * 2
        r = int(255 - t * (255 - 99))
        g = int(255 - t * (255 - 190))
        b = int(255 - t * (255 - 123))
    else:
        t = normalized * 2
        r = int(248 - t * (248 - 255))
        g = int(105 + t * (255 - 105))
        b = int(107 + t * (255 - 107))

    return f"rgb({r},{g},{b})"


def style_dataframe_with_gradient(df: pd.DataFrame, gradient_cols: dict, format_cols: dict = None):
    """
    Apply gradient coloring to specified columns.

    Args:
        df: DataFrame to style
        gradient_cols: dict of {column_name: high_is_good} for gradient coloring
        format_cols: dict of {column_name: format_string} for number formatting

    Returns:
        Styled DataFrame
    """
    def apply_gradient(col):
        col_name = col.name
        if col_name not in gradient_cols:
            return [''] * len(col)

        high_is_good = gradient_cols[col_name]

        # Convert to numeric, handling both string and numeric inputs
        def to_numeric_value(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                cleaned = val.replace('%', '').replace('-', '').strip()
                if not cleaned:
                    return np.nan
                try:
                    return float(cleaned)
                except ValueError:
                    return np.nan
            return np.nan

        numeric_values = [to_numeric_value(v) for v in col]
        numeric_series = pd.Series(numeric_values)
        col_min = numeric_series.min()
        col_max = numeric_series.max()

        styles = []
        for num_val in numeric_values:
            color = get_gradient_color(num_val, col_min, col_max, high_is_good)
            styles.append(f'background-color: {color}')

        return styles

    styled = df.style.apply(apply_gradient, axis=0)

    # Apply number formatting if specified
    if format_cols:
        styled = styled.format(format_cols, na_rep='-')

    return styled


def _render_source_banner(source: str, description: str, bg_color: str, border_color: str, url: str = None):
    """Render a styled data source attribution banner."""
    link_html = f'<a href="{url}" target="_blank" style="color: {border_color}; font-weight: 600;">{source}</a>' if url else f'<strong>{source}</strong>'
    st.markdown(f"""
    <div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>Data Source:</strong> {link_html}<br>
        <small style="opacity: 0.85;">{description}</small>
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
    df = player_projections[available_cols].copy()

    # Add value column if we have Points and Price
    if 'Points' in df.columns and 'Price' in df.columns:
        df['Value'] = (df['Points'] / df['Price']).round(2)

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        player_filter = col1.text_input("Search by name", placeholder="e.g., Salah", key="rw_player_filter")

        if 'Position' in df.columns:
            all_positions = sorted(df['Position'].dropna().unique().tolist())
            position_filter = col2.multiselect("Position", options=all_positions, default=all_positions, key="rw_pos")
        else:
            position_filter = None

        if 'Price' in df.columns:
            min_price = float(df['Price'].min())
            max_price = float(df['Price'].max())
            price_filter = col3.slider("Max price", min_value=min_price, max_value=max_price, value=max_price, step=0.5, key="rw_price")
        else:
            price_filter = None

    # Apply filters
    result = df.copy()

    if player_filter:
        result = result[result['Player'].str.contains(player_filter, case=False, na=False)]

    if position_filter and 'Position' in result.columns:
        result = result[result['Position'].isin(position_filter)]

    if price_filter and 'Price' in result.columns:
        result = result[result['Price'] <= price_filter]

    # Display with gradient coloring
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Player Projections")

    # Format numeric columns before styling
    display_df = result.copy()

    def safe_round(x, decimals):
        """Round numeric values, pass through strings unchanged."""
        if pd.isna(x):
            return x
        if isinstance(x, (int, float)):
            return round(x, decimals)
        return x  # Already a string, leave as-is

    if 'Points' in display_df.columns:
        display_df['Points'] = display_df['Points'].apply(lambda x: safe_round(x, 1))
    if 'Value' in display_df.columns:
        display_df['Value'] = display_df['Value'].apply(lambda x: safe_round(x, 2))
    if 'TSB %' in display_df.columns:
        display_df['TSB %'] = display_df['TSB %'].apply(lambda x: safe_round(x, 1))
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].apply(lambda x: safe_round(x, 1))

    # Define which columns get gradient coloring and their direction
    gradient_cols = {
        'Points': True,      # High is good
        'Value': True,       # High is good
        'Pos Rank': False,   # Low is good (rank 1 is best)
        'TSB %': True,       # High ownership = popular pick
        'Price': False,      # Lower price = better value potential
    }

    # Only apply gradient to columns that exist
    active_gradient = {k: v for k, v in gradient_cols.items() if k in display_df.columns}

    styled_df = style_dataframe_with_gradient(display_df, active_gradient)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
    st.caption(f"Showing {len(result)} of {len(df)} players. Data from Rotowire.")


# =============================================================================
# Fantasy Football Pundit Data Tab
# =============================================================================

def render_ffp_data():
    """Render the FFP data tab."""
    _render_source_banner(
        "Fantasy Football Pundit",
        "Player data with start probability, ownership, and betting-derived probabilities.",
        "#fef3c7", "#f59e0b",
        "https://www.fantasyfootballpundit.com/fpl-points-predictor/"
    )

    raw_df = get_ffp_projections_data()

    if raw_df is None or raw_df.empty:
        st.warning("Could not load FFP data. The data source may be temporarily unavailable.")
        return

    # Check which columns actually have data (non-zero values)
    prediction_cols = ['Predicted', 'StartingPredicted', 'Next2GWs', 'Next3GWs', 'Next6GWs']
    has_predictions = any(
        col in raw_df.columns and (raw_df[col] != 0).any()
        for col in prediction_cols
    )

    # Build display columns based on available data
    base_cols = ['Name', 'Team', 'Position', 'Fixture', 'Price', 'Ownership', 'Start']

    if has_predictions:
        display_cols = base_cols + ['Predicted', 'StartingPredicted', 'Next2GWs', 'Next3GWs', 'Next6GWs']
        display_names = {
            'Name': 'Player', 'Ownership': 'Own %', 'Start': 'Start %',
            'Predicted': 'Pred Pts', 'StartingPredicted': 'Pts (if starts)',
            'Next2GWs': 'Next 2 GW', 'Next3GWs': 'Next 3 GW', 'Next6GWs': 'Next 6 GW'
        }
        gradient_cols = {
            'Pred Pts': True, 'Pts (if starts)': True,
            'Next 2 GW': True, 'Next 3 GW': True, 'Next 6 GW': True,
            'Start %': True, 'Own %': True
        }
    else:
        # Fallback to odds-based columns when predictions unavailable
        display_cols = base_cols + ['CS', 'AnytimeGoal', 'AnytimeAssist', 'AnytimeReturn']
        display_names = {
            'Name': 'Player', 'Ownership': 'Own %', 'Start': 'Start %',
            'CS': 'CS %', 'AnytimeGoal': 'Goal %', 'AnytimeAssist': 'Assist %', 'AnytimeReturn': 'Return %'
        }
        gradient_cols = {
            'Start %': True, 'Own %': True, 'CS %': True,
            'Goal %': True, 'Assist %': True, 'Return %': True
        }
        st.info("Point predictions are currently unavailable from FFP. Showing ownership and betting odds data instead.")

    available_cols = [c for c in display_cols if c in raw_df.columns]
    df = raw_df[available_cols].copy()
    df = df.rename(columns={k: v for k, v in display_names.items() if k in df.columns})

    # Filter to players with start chance
    if 'Start %' in df.columns:
        df = df[df['Start %'] > 0].copy()

    # Sort by a meaningful column
    sort_col = 'Pred Pts' if 'Pred Pts' in df.columns else ('Goal %' if 'Goal %' in df.columns else None)
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        player_filter = col1.text_input("Search by name", placeholder="e.g., Salah", key="ffp_player_filter")

        if 'Position' in df.columns:
            all_positions = sorted(df['Position'].dropna().unique().tolist())
            position_filter = col2.multiselect("Position", options=all_positions, default=all_positions, key="ffp_pos")
        else:
            position_filter = None

        if 'Start %' in df.columns:
            min_start = col3.slider("Min start %", min_value=0, max_value=100, value=0, step=10, key="ffp_start")
        else:
            min_start = 0

    # Apply filters
    result = df.copy()

    if player_filter and 'Player' in result.columns:
        result = result[result['Player'].str.contains(player_filter, case=False, na=False)]

    if position_filter and 'Position' in result.columns:
        result = result[result['Position'].isin(position_filter)]

    if min_start > 0 and 'Start %' in result.columns:
        result = result[result['Start %'] >= min_start]

    # Format columns for display
    display_df = result.copy()

    # Format Price to 1 decimal
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    # Format prediction columns to 1 decimal
    pred_cols = ['Pred Pts', 'Pts (if starts)', 'Next 2 GW', 'Next 3 GW', 'Next 6 GW']
    for col in pred_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x != 0 else "-")

    # Format percentage columns
    pct_cols = [c for c in display_df.columns if '%' in c]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) and x != 0 else "-")

    # Display with gradient
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Player Data")

    # Apply gradient only to existing columns
    active_gradient = {k: v for k, v in gradient_cols.items() if k in display_df.columns}
    styled_df = style_dataframe_with_gradient(display_df, active_gradient)

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
    st.caption(f"Showing {len(result)} of {len(df)} players. Data from Fantasy Football Pundit.")


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
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        player_filter = col1.text_input("Search by name", placeholder="e.g., Haaland", key="gs_player_filter")

        if 'Position' in df.columns:
            all_positions = sorted(df['Position'].dropna().unique().tolist())
            position_filter = col2.multiselect("Position", options=all_positions, default=all_positions, key="gs_pos")
        else:
            position_filter = None

        if 'Team' in df.columns:
            all_teams = sorted(df['Team'].dropna().unique().tolist())
            team_filter = col3.multiselect("Team", options=all_teams, default=[], key="gs_team",
                                           help="Leave empty for all teams")
        else:
            team_filter = []

    # Apply filters
    result = df.copy()

    if player_filter and 'Player' in result.columns:
        result = result[result['Player'].str.contains(player_filter, case=False, na=False)]

    if position_filter and 'Position' in result.columns:
        result = result[result['Position'].isin(position_filter)]

    if team_filter and 'Team' in result.columns:
        result = result[result['Team'].isin(team_filter)]

    # Format percentages for display
    display_df = result.copy()
    for col in ['Goal %', 'Assist %', 'Return %', 'Start %']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")

    # Display with gradient
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Goalscorer & Assist Probabilities")

    gradient_cols = {'Goal %': True, 'Assist %': True, 'Return %': True, 'Start %': True}
    active_gradient = {k: v for k, v in gradient_cols.items() if k in display_df.columns}
    styled_df = style_dataframe_with_gradient(display_df, active_gradient)

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
    st.caption(f"Showing {len(result)} of {len(df)} players. Betting odds converted to probabilities.")


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
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Clean Sheet Probabilities")

    # Format for display
    display_df = df.copy()
    if 'CS Prob %' in display_df.columns:
        display_df['CS Prob %'] = display_df['CS Prob %'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")

    gradient_cols = {'CS Prob %': True}
    styled_df = style_dataframe_with_gradient(display_df, gradient_cols)

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)
    st.caption(f"Showing {len(df)} teams. Betting odds converted to probabilities.")


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
        """)
        return

    df = get_odds_api_match_odds(api_key)

    if df is None or df.empty:
        st.warning("Could not load match odds data. The API may be temporarily unavailable or rate limited.")
        return

    # Display
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Match Odds")

    # Format percentages
    display_df = df.copy()
    for col in ['Home Win %', 'Draw %', 'Away Win %']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

    gradient_cols = {'Home Win %': True, 'Away Win %': True}
    styled_df = style_dataframe_with_gradient(display_df, gradient_cols)

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=450)
    st.caption(f"Showing {len(df)} matches. UK bookmakers average odds.")


# =============================================================================
# Main Page
# =============================================================================

def show_player_projections_page():
    """Main projections hub page with tabbed interface."""
    st.title("Projections Hub")
    st.caption("Player and team projections from multiple data sources to inform your FPL decisions.")

    # Create tabs for different data sources
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Rotowire",
        "FFP Data",
        "Goal/Assist Odds",
        "Clean Sheet Odds",
        "Match Odds"
    ])

    with tab1:
        render_rotowire_projections()

    with tab2:
        render_ffp_data()

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
        ### Data Sources

        | Source | Type | Update Frequency |
        |--------|------|------------------|
        | **Rotowire** | Expert point projections | Weekly (before each GW) |
        | **FFP Data** | Start %, ownership, predictions | Throughout GW |
        | **Goal/Assist Odds** | Betting probabilities | Daily |
        | **Clean Sheet Odds** | Team CS probabilities | Daily |
        | **The Odds API** | Match betting odds | Live updates |

        ### Color Scale

        Tables use a **green-white-red** gradient where:
        - **Green** = Good values (high points, high odds, low rank)
        - **White** = Average values
        - **Red** = Poor values

        ### Comparing Sources

        When Rotowire and FFP agree on a player, that's a high-confidence pick.
        When they diverge, investigate further (injury doubt, rotation risk, etc.).
        """)
