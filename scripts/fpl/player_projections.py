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
from datetime import datetime
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


def style_dataframe_with_gradient(df: pd.DataFrame, gradient_cols: dict, position_col: str = None):
    """
    Apply gradient coloring to specified columns.

    Args:
        df: DataFrame to style
        gradient_cols: dict of {column_name: high_is_good} for gradient coloring
        position_col: if set, apply Pos Rank gradient within each position group

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

    def apply_gradient_by_position(col):
        """Apply gradient for Pos Rank within each position group."""
        col_name = col.name
        if col_name != 'Pos Rank' or position_col is None or position_col not in df.columns:
            return apply_gradient(col)

        styles = [''] * len(col)
        positions = df[position_col].values

        for pos in df[position_col].unique():
            if pd.isna(pos):
                continue
            mask = positions == pos
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            pos_values = []
            for idx in indices:
                val = col.iloc[idx]
                if pd.isna(val):
                    pos_values.append(np.nan)
                elif isinstance(val, (int, float)):
                    pos_values.append(float(val))
                else:
                    pos_values.append(np.nan)

            pos_series = pd.Series(pos_values)
            col_min = pos_series.min()
            col_max = pos_series.max()

            for i, idx in enumerate(indices):
                color = get_gradient_color(pos_values[i], col_min, col_max, False)  # Low rank is good
                styles[idx] = f'background-color: {color}'

        return styles

    # Use position-aware gradient for Pos Rank if position_col is provided
    if position_col and 'Pos Rank' in gradient_cols:
        # Apply regular gradient to non-Pos Rank columns
        other_cols = {k: v for k, v in gradient_cols.items() if k != 'Pos Rank'}
        styled = df.style.apply(apply_gradient, axis=0, subset=list(other_cols.keys()) if other_cols else None)
        # Apply position-aware gradient to Pos Rank
        if 'Pos Rank' in df.columns:
            styled = styled.apply(apply_gradient_by_position, axis=0, subset=['Pos Rank'])
        return styled
    else:
        return df.style.apply(apply_gradient, axis=0)


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
# Position Badges
# =============================================================================

# Position badge colors
POSITION_COLORS = {
    'GK': {'bg': '#f97316', 'text': '#ffffff'},   # Orange
    'GKP': {'bg': '#f97316', 'text': '#ffffff'},  # Orange (alternate name)
    'DEF': {'bg': '#3b82f6', 'text': '#ffffff'},  # Blue
    'MID': {'bg': '#22c55e', 'text': '#ffffff'},  # Green
    'FWD': {'bg': '#ef4444', 'text': '#ffffff'},  # Red
}


def get_position_badge_html(position: str) -> str:
    """Generate HTML for a colored position badge."""
    if pd.isna(position):
        return ""

    pos = str(position).strip().upper()
    colors = POSITION_COLORS.get(pos, {'bg': '#6b7280', 'text': '#ffffff'})

    return f'''<span style="
        background: {colors['bg']};
        color: {colors['text']};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    ">{pos}</span>'''


def render_player_table_with_badges(df: pd.DataFrame, gradient_cols: dict, position_col: str = 'Position'):
    """
    Render a player table with position badges as HTML.

    This creates a custom HTML table with position badges while maintaining
    gradient coloring for numeric columns.
    """
    if df.empty:
        st.info("No data to display.")
        return

    # Convert position to badges
    if position_col in df.columns:
        df = df.copy()
        df['_pos_badge'] = df[position_col].apply(get_position_badge_html)

    # Build HTML table
    html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 14px;">'

    # Header row
    html += '<thead><tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">'
    for col in df.columns:
        if col.startswith('_'):
            continue
        if col == position_col:
            html += f'<th style="padding: 12px 8px; text-align: left; font-weight: 600;">Pos</th>'
        else:
            html += f'<th style="padding: 12px 8px; text-align: left; font-weight: 600;">{col}</th>'
    html += '</tr></thead>'

    # Calculate gradient bounds for each column
    col_bounds = {}
    for col, high_is_good in gradient_cols.items():
        if col in df.columns:
            numeric_vals = pd.to_numeric(df[col].astype(str).str.replace('%', '').str.replace('-', ''), errors='coerce')
            col_bounds[col] = {
                'min': numeric_vals.min(),
                'max': numeric_vals.max(),
                'high_is_good': high_is_good
            }

    # Data rows
    html += '<tbody>'
    for idx, row in df.iterrows():
        html += '<tr style="border-bottom: 1px solid #e2e8f0;">'
        for col in df.columns:
            if col.startswith('_'):
                continue

            val = row[col]

            if col == position_col:
                # Use badge instead of plain text
                html += f'<td style="padding: 10px 8px;">{row["_pos_badge"]}</td>'
            elif col in col_bounds:
                # Apply gradient background
                bounds = col_bounds[col]
                try:
                    numeric_val = float(str(val).replace('%', '').replace('-', '')) if val else None
                    if numeric_val is not None and pd.notna(numeric_val):
                        color = get_gradient_color(numeric_val, bounds['min'], bounds['max'], bounds['high_is_good'])
                    else:
                        color = '#f5f5f5'
                except (ValueError, TypeError):
                    color = '#f5f5f5'
                html += f'<td style="padding: 10px 8px; background: {color};">{val}</td>'
            else:
                html += f'<td style="padding: 10px 8px;">{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'

    st.markdown(html, unsafe_allow_html=True)


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
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Player Projections")

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

    # Define which columns get gradient coloring
    gradient_cols = {
        'Points': True,      # High is good
        'Value': True,       # High is good
        'Pos Rank': False,   # Low is good (rank 1 is best) - will be by position
        'TSB %': True,       # High ownership = popular pick
        'Price': False,      # Lower price = better value potential
    }

    # Only apply gradient to columns that exist
    active_gradient = {k: v for k, v in gradient_cols.items() if k in display_df.columns}

    # Render with position badges
    render_player_table_with_badges(display_df, active_gradient, position_col='Position')
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
        display_df['Price'] = pd.to_numeric(display_df['Price'], errors='coerce').round(1)

    # Format prediction columns to 1 decimal
    pred_cols = ['Pred Pts', 'Pts (if starts)', 'Next 2 GW', 'Next 3 GW', 'Next 6 GW']
    for col in pred_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(1)

    # Format percentage columns
    pct_cols = [c for c in display_df.columns if '%' in c]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) and x != 0 else "-")

    # Display with gradient and position badges
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Player Data")

    # Apply gradient only to existing columns
    active_gradient = {k: v for k, v in gradient_cols.items() if k in display_df.columns}

    # Render with position badges
    render_player_table_with_badges(display_df, active_gradient, position_col='Position')
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

    # Display with gradient and position badges
    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Goalscorer & Assist Probabilities")

    gradient_cols = {'Goal %': True, 'Assist %': True, 'Return %': True, 'Start %': True}
    active_gradient = {k: v for k, v in gradient_cols.items() if k in display_df.columns}

    # Render with position badges
    render_player_table_with_badges(display_df, active_gradient, position_col='Position')
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

    df = get_ffp_clean_sheet_odds()

    if df is None or df.empty:
        st.warning("Could not load clean sheet odds data. The data source may be temporarily unavailable.")
        return

    st.markdown(f"#### GW{config.CURRENT_GAMEWEEK} Clean Sheet Probabilities")

    # Create horizontal bar chart visualization
    if 'CS Prob %' in df.columns and 'Team' in df.columns:
        # Sort by CS probability descending
        df_sorted = df.sort_values('CS Prob %', ascending=False).reset_index(drop=True)

        # Build HTML for horizontal bars
        bars_html = '<div style="display: flex; flex-direction: column; gap: 8px;">'

        for _, row in df_sorted.iterrows():
            team = row['Team']
            fixture = row.get('Fixture', '')
            prob = row['CS Prob %']

            if pd.isna(prob):
                prob = 0

            # Color based on probability (green gradient)
            if prob >= 50:
                bar_color = "#22c55e"  # Strong green
            elif prob >= 35:
                bar_color = "#86efac"  # Light green
            elif prob >= 25:
                bar_color = "#fde047"  # Yellow
            else:
                bar_color = "#fca5a5"  # Light red

            bars_html += f'''
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 100px; font-weight: 600; font-size: 14px;">{team}</div>
                <div style="flex: 1; background: #e5e7eb; border-radius: 4px; height: 24px; position: relative;">
                    <div style="width: {prob}%; background: {bar_color}; height: 100%; border-radius: 4px; transition: width 0.3s;"></div>
                    <span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: 500;">{prob:.0f}%</span>
                </div>
                <div style="width: 140px; font-size: 12px; color: #6b7280;">{fixture}</div>
            </div>
            '''

        bars_html += '</div>'

        st.markdown(bars_html, unsafe_allow_html=True)
        st.caption(f"Showing {len(df)} teams. Betting odds converted to probabilities.")
    else:
        st.warning("Data format not as expected.")


# =============================================================================
# Match Odds Tab (The Odds API)
# =============================================================================

def _get_odds_color(pct: float) -> str:
    """Get color based on win probability percentage."""
    if pct >= 60:
        return "#22c55e"  # Strong green - heavy favorite
    elif pct >= 45:
        return "#86efac"  # Light green - slight favorite
    elif pct >= 35:
        return "#fde047"  # Yellow - toss-up
    elif pct >= 25:
        return "#fca5a5"  # Light red - underdog
    else:
        return "#ef4444"  # Strong red - big underdog


def _render_match_card(home_team: str, away_team: str, kickoff: str, home_pct: float, draw_pct: float, away_pct: float):
    """Render a single match card with odds visualization."""
    # Format kickoff time
    try:
        kickoff_dt = pd.to_datetime(kickoff)
        kickoff_str = kickoff_dt.strftime("%a %b %d, %H:%M")
    except:
        kickoff_str = str(kickoff) if kickoff else ""

    home_color = _get_odds_color(home_pct)
    away_color = _get_odds_color(away_pct)

    card_html = f'''
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    ">
        <div style="text-align: center; font-size: 12px; color: #6b7280; margin-bottom: 12px;">
            {kickoff_str}
        </div>

        <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
            <!-- Home Team -->
            <div style="flex: 1; text-align: center;">
                <div style="font-weight: 700; font-size: 16px; margin-bottom: 8px;">{home_team}</div>
                <div style="
                    background: {home_color};
                    color: #ffffff;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: 700;
                ">{home_pct:.0f}%</div>
            </div>

            <!-- Draw -->
            <div style="text-align: center;">
                <div style="font-size: 12px; color: #9ca3af; margin-bottom: 8px;">Draw</div>
                <div style="
                    background: #f3f4f6;
                    color: #374151;
                    padding: 8px 12px;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 600;
                ">{draw_pct:.0f}%</div>
            </div>

            <!-- Away Team -->
            <div style="flex: 1; text-align: center;">
                <div style="font-weight: 700; font-size: 16px; margin-bottom: 8px;">{away_team}</div>
                <div style="
                    background: {away_color};
                    color: #ffffff;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: 700;
                ">{away_pct:.0f}%</div>
            </div>
        </div>
    </div>
    '''
    return card_html


def render_match_odds():
    """Render the match betting odds tab from The Odds API with card layout."""
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

            # Display by gameweek with card layout
            for gw in sorted(df['GW'].unique()):
                gw_df = df[df['GW'] == gw].copy()

                st.markdown(f"#### GW{gw} Match Odds")

                # Create 2-column layout for match cards
                cols = st.columns(2)
                col_idx = 0

                for _, row in gw_df.iterrows():
                    home_team = row.get('Home Team', '')
                    away_team = row.get('Away Team', '')
                    kickoff = row.get('Kickoff', '')
                    home_pct = row.get('Home Win %', 0) or 0
                    draw_pct = row.get('Draw %', 0) or 0
                    away_pct = row.get('Away Win %', 0) or 0

                    card_html = _render_match_card(home_team, away_team, kickoff, home_pct, draw_pct, away_pct)

                    with cols[col_idx]:
                        st.markdown(card_html, unsafe_allow_html=True)

                    col_idx = (col_idx + 1) % 2

                st.caption(f"{len(gw_df)} matches. UK bookmakers average odds.")

                if gw != max(df['GW'].unique()):
                    st.markdown("---")
    else:
        # Fallback if no Kickoff column - use simple card layout
        st.markdown(f"#### Match Odds")

        cols = st.columns(2)
        col_idx = 0

        for _, row in df.iterrows():
            home_team = row.get('Home Team', '')
            away_team = row.get('Away Team', '')
            kickoff = row.get('Kickoff', '')
            home_pct = row.get('Home Win %', 0) or 0
            draw_pct = row.get('Draw %', 0) or 0
            away_pct = row.get('Away Win %', 0) or 0

            card_html = _render_match_card(home_team, away_team, kickoff, home_pct, draw_pct, away_pct)

            with cols[col_idx]:
                st.markdown(card_html, unsafe_allow_html=True)

            col_idx = (col_idx + 1) % 2


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

        **Pos Rank** is colored within each position group (so GK rank 1 and MID rank 1 both show as green).

        ### Position Badges

        Player positions are displayed with color-coded badges:
        - **GK** (Orange) - Goalkeepers
        - **DEF** (Blue) - Defenders
        - **MID** (Green) - Midfielders
        - **FWD** (Red) - Forwards
        """)
