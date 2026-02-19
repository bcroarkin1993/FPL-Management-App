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
import requests
import streamlit as st
from scripts.common.utils import (
    get_rotowire_player_projections,
    get_rotowire_rankings_url,
    get_ffp_projections_data,
    get_ffp_goalscorer_odds,
    get_ffp_clean_sheet_odds,
    get_odds_api_match_odds,
    get_classic_bootstrap_static,
)
from scripts.common.styled_tables import render_styled_table


# =============================================================================
# Data Freshness Detection
# =============================================================================

@st.cache_data(ttl=300)
def _get_current_gw_teams() -> set:
    """Get set of team short names playing in the current gameweek."""
    try:
        # Get bootstrap data for team names
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap:
            return set()

        team_id_to_short = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}

        # Get current GW fixtures
        gw = config.CURRENT_GAMEWEEK
        url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        fixtures = resp.json()

        teams = set()
        for fx in fixtures:
            h, a = fx.get('team_h'), fx.get('team_a')
            if h:
                teams.add(team_id_to_short.get(h, ''))
            if a:
                teams.add(team_id_to_short.get(a, ''))

        return teams
    except Exception:
        return set()


def _is_ffp_data_current(ffp_df: pd.DataFrame) -> bool:
    """
    Check if FFP data is for the current gameweek by comparing fixtures.

    Returns True if the FFP fixture data matches current GW teams.
    """
    if ffp_df is None or ffp_df.empty:
        return False

    if 'Fixture' not in ffp_df.columns:
        return False

    current_teams = _get_current_gw_teams()
    if not current_teams:
        # Can't determine, assume current
        return True

    # Extract team abbreviations from FFP fixtures (format: "ARS (H)" or "MUN (A)")
    ffp_teams = set()
    for fixture in ffp_df['Fixture'].dropna().unique():
        # Extract team code (first 3 chars typically)
        parts = str(fixture).split()
        if parts:
            team_code = parts[0].upper()
            ffp_teams.add(team_code)

    # Check overlap - if FFP teams mostly match current GW teams, data is current
    if not ffp_teams:
        return False

    overlap = len(ffp_teams & current_teams)
    # If at least 50% of FFP teams are in current GW, consider it current
    return overlap >= len(ffp_teams) * 0.5


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

    # Extract GW from URL to detect stale data
    import re
    url_gw_match = re.search(r'gameweek-(\d+)', url.lower())
    data_gw = int(url_gw_match.group(1)) if url_gw_match else None
    current_gw = config.CURRENT_GAMEWEEK

    # Block stale data - don't show previous GW projections
    if data_gw and data_gw != current_gw:
        st.info(f"GW{current_gw} projections are not yet available from Rotowire. Check back closer to the gameweek deadline.")
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
        df['Value'] = df['Points'] / df['Price']

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

    # Display with gradient coloring and position badges
    display_gw = data_gw if data_gw else config.CURRENT_GAMEWEEK
    st.markdown(f"#### GW{display_gw} Player Projections")

    # Format numeric columns for display
    display_df = result.copy()
    if 'Points' in display_df.columns:
        display_df['Points'] = pd.to_numeric(display_df['Points'], errors='coerce').round(1)
    if 'Value' in display_df.columns:
        display_df['Value'] = pd.to_numeric(display_df['Value'], errors='coerce').round(2)
    if 'Price' in display_df.columns:
        display_df['Price'] = pd.to_numeric(display_df['Price'], errors='coerce').round(1)
    if 'TSB %' in display_df.columns:
        display_df['TSB %'] = pd.to_numeric(display_df['TSB %'], errors='coerce').round(1)

    # Define which columns get color scaling
    positive_cols = [c for c in ['Points', 'Value', 'TSB %'] if c in display_df.columns]
    negative_cols = [c for c in ['Pos Rank', 'Price'] if c in display_df.columns]

    render_styled_table(
        display_df,
        col_formats={'Points': '{:.1f}', 'Value': '{:.2f}', 'Price': '{:.1f}', 'TSB %': '{:.1f}'},
        positive_color_cols=positive_cols,
        negative_color_cols=negative_cols,
        max_height=600,
    )
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

    current_gw = config.CURRENT_GAMEWEEK

    # Check if data is for current gameweek
    if not _is_ffp_data_current(raw_df):
        st.info(f"GW{current_gw} data is not yet available from Fantasy Football Pundit. Check back closer to the gameweek deadline.")
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
        display_df['Price'] = pd.to_numeric(display_df['Price'], errors='coerce').round(1)

    # Format prediction columns to 1 decimal
    pred_cols = ['Pred Pts', 'Pts (if starts)', 'Next 2 GW', 'Next 3 GW', 'Next 6 GW']
    for col in pred_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(1)

    # Keep percentage columns numeric for color scaling
    pct_cols = [c for c in display_df.columns if '%' in c]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    # Display with styled table
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Player Data")

    # Determine color columns based on available data
    positive_cols = [c for c in gradient_cols.keys() if gradient_cols.get(c, True) and c in display_df.columns]

    # Build col_formats with % suffix for percentage columns
    col_fmts = {'Price': '{:.1f}'}
    for col in pct_cols:
        if col in display_df.columns:
            col_fmts[col] = '{:.0f}%'
    for col in pred_cols:
        if col in display_df.columns:
            col_fmts[col] = '{:.1f}'

    render_styled_table(
        display_df,
        col_formats=col_fmts,
        positive_color_cols=positive_cols,
        max_height=600,
    )
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

    # First check if FFP data is current using the base projections data
    raw_df = get_ffp_projections_data()
    if raw_df is not None and not _is_ffp_data_current(raw_df):
        current_gw = config.CURRENT_GAMEWEEK
        st.info(f"GW{current_gw} data is not yet available from Fantasy Football Pundit. Check back closer to the gameweek deadline.")
        return

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

    # Keep percentages numeric for color scaling
    display_df = result.copy()
    for col in ['Goal %', 'Assist %', 'Return %', 'Start %']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    # Display with styled table
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Goalscorer & Assist Probabilities")

    positive_cols = [c for c in ['Goal %', 'Assist %', 'Return %', 'Start %'] if c in display_df.columns]
    col_fmts = {col: '{:.0f}%' for col in positive_cols}

    render_styled_table(
        display_df,
        col_formats=col_fmts,
        positive_color_cols=positive_cols,
        max_height=600,
    )
    st.caption(f"Showing {len(result)} of {len(df)} players. Betting odds converted to probabilities.")


# =============================================================================
# Clean Sheet Odds Tab
# =============================================================================

def render_clean_sheet_odds():
    """Render the clean sheet odds tab with horizontal bar visualization."""
    _render_source_banner(
        "Fantasy Football Pundit (Clean Sheet Odds)",
        "Team clean sheet probabilities from betting markets.",
        "#f0fdf4", "#22c55e",
        "https://www.fantasyfootballpundit.com/premier-league-clean-sheet-odds/"
    )

    # First check if FFP data is current using the base projections data
    raw_df = get_ffp_projections_data()
    if raw_df is not None and not _is_ffp_data_current(raw_df):
        current_gw = config.CURRENT_GAMEWEEK
        st.info(f"GW{current_gw} data is not yet available from Fantasy Football Pundit. Check back closer to the gameweek deadline.")
        return

    df = get_ffp_clean_sheet_odds()

    if df is None or df.empty:
        st.warning("Could not load clean sheet odds data. The data source may be temporarily unavailable.")
        return

    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Clean Sheet Probabilities")

    # Create horizontal bar chart visualization using native Streamlit
    if 'CS Prob %' in df.columns and 'Team' in df.columns:
        # Sort by CS probability descending
        df_sorted = df.sort_values('CS Prob %', ascending=False).reset_index(drop=True)

        # Use columns for a cleaner layout
        for _, row in df_sorted.iterrows():
            team = row['Team']
            fixture = row.get('Fixture', '')
            prob = row['CS Prob %']

            if pd.isna(prob):
                prob = 0

            # Create 3-column layout: Team | Progress Bar | Fixture
            col1, col2, col3 = st.columns([1, 3, 1.5])

            with col1:
                st.markdown(f"**{team}**")

            with col2:
                # Use Streamlit's progress bar
                st.progress(min(prob / 100, 1.0), text=f"{prob:.0f}%")

            with col3:
                st.caption(fixture)

        st.caption(f"Showing {len(df)} teams. Betting odds converted to probabilities.")
    else:
        st.warning("Data format not as expected.")


# =============================================================================
# Match Odds Tab (The Odds API)
# =============================================================================

def _get_odds_color(pct: float) -> str:
    """Get background color based on win probability."""
    if pct >= 55:
        return "#dcfce7"  # Light green - strong favorite
    elif pct >= 40:
        return "#fef9c3"  # Light yellow - slight favorite
    elif pct >= 30:
        return "#fef3c7"  # Light orange - competitive
    else:
        return "#fee2e2"  # Light red - underdog


def _render_match_card_native(home_team: str, away_team: str, kickoff: str, home_pct: float, draw_pct: float, away_pct: float):
    """Render a single match with color-coded odds visualization."""
    from zoneinfo import ZoneInfo

    # Format kickoff time in EST with full format
    try:
        kickoff_dt = pd.to_datetime(kickoff)
        # Convert UTC to EST
        if kickoff_dt.tzinfo is None:
            kickoff_dt = kickoff_dt.replace(tzinfo=ZoneInfo('UTC'))
        kickoff_est = kickoff_dt.astimezone(ZoneInfo('America/New_York'))

        # Format: "Tuesday, February 10th, 2:30pm EST"
        day_name = kickoff_est.strftime("%A")
        month = kickoff_est.strftime("%B")
        day = kickoff_est.day
        # Add ordinal suffix
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        time_str = kickoff_est.strftime("%I:%M%p").lstrip('0').lower()
        kickoff_str = f"{day_name}, {month} {day}{suffix}, {time_str} EST"
    except:
        kickoff_str = str(kickoff) if kickoff else ""

    # Get colors for each outcome
    home_color = _get_odds_color(home_pct)
    draw_color = _get_odds_color(draw_pct)
    away_color = _get_odds_color(away_pct)

    # Determine favorite indicator (star = heavy favorite with 10%+ advantage)
    if home_pct > away_pct + 10:
        home_indicator = " ⭐"
        away_indicator = ""
    elif away_pct > home_pct + 10:
        home_indicator = ""
        away_indicator = " ⭐"
    else:
        home_indicator = ""
        away_indicator = ""

    # Build styled card using HTML with larger team names and Home/Away labels
    card_html = f'''
    <div style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 14px; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <div style="text-align: center; color: #6b7280; font-size: 13px; margin-bottom: 12px;">{kickoff_str}</div>
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
            <div style="flex: 2; text-align: left;">
                <div style="font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">Home</div>
                <div style="font-weight: 700; font-size: 17px;">{home_team}{home_indicator}</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 8px 10px; border-radius: 6px; background: {home_color}; font-weight: 700; font-size: 16px;">{home_pct:.0f}%</div>
            <div style="flex: 1; text-align: center; padding: 8px 10px; border-radius: 6px; background: {draw_color}; margin: 0 4px; font-size: 14px; color: #4b5563;">Draw<br><strong>{draw_pct:.0f}%</strong></div>
            <div style="flex: 1; text-align: center; padding: 8px 10px; border-radius: 6px; background: {away_color}; font-weight: 700; font-size: 16px;">{away_pct:.0f}%</div>
            <div style="flex: 2; text-align: right;">
                <div style="font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">Away</div>
                <div style="font-weight: 700; font-size: 17px;">{away_indicator}{away_team}</div>
            </div>
        </div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)


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

    # Parse kickoff times and group by gameweek
    if 'Kickoff' in df.columns:
        df['Kickoff_dt'] = pd.to_datetime(df['Kickoff'], errors='coerce')
        df = df.sort_values('Kickoff_dt')

        # Group matches by date range (approximate GW - matches within 4 days of each other)
        current_gw = config.CURRENT_GAMEWEEK
        gw_matches = []
        current_gw_num = current_gw

        if len(df) > 0:
            first_date = df['Kickoff_dt'].min()
            for idx, row in df.iterrows():
                match_date = row['Kickoff_dt']
                if pd.notna(match_date) and pd.notna(first_date):
                    days_diff = (match_date - first_date).days
                    # If more than 5 days from first match, it's next GW
                    if days_diff > 5:
                        current_gw_num = current_gw + 1
                        first_date = match_date
                gw_matches.append(current_gw_num)

            df['GW'] = gw_matches

            # Display by gameweek
            for gw in sorted(df['GW'].unique()):
                gw_df = df[df['GW'] == gw].copy()

                st.markdown(f"#### GW{gw} Match Odds")

                # Display matches using native Streamlit components
                for _, row in gw_df.iterrows():
                    home_team = row.get('Home Team', '')
                    away_team = row.get('Away Team', '')
                    kickoff = row.get('Kickoff', '')
                    home_pct = row.get('Home Win %', 0) or 0
                    draw_pct = row.get('Draw %', 0) or 0
                    away_pct = row.get('Away Win %', 0) or 0

                    _render_match_card_native(home_team, away_team, kickoff, home_pct, draw_pct, away_pct)

                st.caption(f"{len(gw_df)} matches. UK bookmakers average odds.")

                if gw != max(df['GW'].unique()):
                    st.markdown("---")
    else:
        # Fallback if no Kickoff column
        st.markdown(f"#### Match Odds")

        for _, row in df.iterrows():
            home_team = row.get('Home Team', '')
            away_team = row.get('Away Team', '')
            kickoff = row.get('Kickoff', '')
            home_pct = row.get('Home Win %', 0) or 0
            draw_pct = row.get('Draw %', 0) or 0
            away_pct = row.get('Away Win %', 0) or 0

            _render_match_card_native(home_team, away_team, kickoff, home_pct, draw_pct, away_pct)


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

        Tables use a **red-to-green text color** gradient where:
        - **Green text** = Good values (high points, high odds, low rank)
        - **Yellow text** = Average values
        - **Red text** = Poor values

        ### Match Odds Visual Guide

        Each match card displays:
        - **Home team** (left) and **Away team** (right) with clear labels
        - **Win probabilities** with color-coded backgrounds:
          - 🟢 Green (55%+) = Strong favorite
          - 🟡 Yellow (40-55%) = Slight favorite
          - 🟠 Orange (30-40%) = Competitive match
          - 🔴 Red (<30%) = Underdog
        - **⭐ Star indicator** = Heavy favorite (10%+ win probability advantage over opponent)
        - **Kickoff times** shown in Eastern Time (EST/EDT)
        """)
