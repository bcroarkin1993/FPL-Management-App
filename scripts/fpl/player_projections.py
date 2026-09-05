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
from scripts.common.text_helpers import format_last_updated
from scripts.common.scraping import (
    FFP_CLEAN_SHEET_URL,
    FFP_GOAL_ASSIST_URL,
    FFP_POINTS_PREDICTOR_URL,
    get_ffp_feed,
    get_rotowire_article_updated,
)
from scripts.common.utils import (
    get_rotowire_player_projections,
    get_rotowire_rankings_url,
    get_ffp_goalscorer_odds,
    get_ffp_clean_sheet_odds,
    get_odds_api_match_odds,
)
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import to_display_name
from scripts.common.analytics import blend_projections_onto, merge_season_projections
from scripts.common.fpl_classic_api import get_classic_bootstrap_static


# =============================================================================
# Data Freshness
# =============================================================================

def _ffp_gate(feed):
    """Report FFP's own gameweek, and say plainly when it is not ours.

    What this replaces was a set-overlap test between FFP's fixture teams and
    the current gameweek's teams. All 20 clubs play every gameweek, so that set
    is the same every week: it scored 18/19 for GW2, GW3 and GW4 alike and could
    never fail. FFP's feed now states its gameweek outright.

    Returns True when the tab should render its table.
    """
    if not feed.ok:
        st.warning(
            "Could not load Fantasy Football Pundit data. "
            + (feed.note or "Neither their site nor their published spreadsheet responded.")
            + " This usually clears on its own — reload in a moment."
        )
        return False

    if feed.gameweek is None:
        st.info(
            "Fantasy Football Pundit data loaded, but which gameweek it covers could "
            "not be determined. Check the Fixture column before relying on it."
        )
        return True

    current_gw = config.CURRENT_GAMEWEEK
    if feed.is_stale(current_gw):
        st.warning(
            f"Fantasy Football Pundit has published **GW{feed.gameweek}**, but the "
            f"current gameweek is **GW{current_gw}**. The numbers below are for "
            f"GW{feed.gameweek} and are excluded from scoring elsewhere in the app."
        )
    return True


def _ffp_caption(feed):
    """One line naming the gameweek, the publish time and which source answered."""
    bits = []
    if feed.gameweek is not None:
        bits.append(f"Gameweek {feed.gameweek}")
    if feed.updated is not None:
        bits.append(f"published {format_last_updated(feed.updated)}")
    if feed.provenance == "sheet":
        bits.append("read from FFP's published spreadsheet (their site was unreachable)")
    return " · ".join(bits)


def _render_source_banner(source: str, description: str, bg_color: str, border_color: str,
                          url: str = None, updated=None):
    """Render a styled data source attribution banner.

    `updated` adds the source's own publish time. Worth showing: a projection
    table written before the last team-news cycle is materially less reliable
    than one written after it, and nothing else on the page reveals which you
    are looking at.
    """
    link_html = f'<a href="{url}" target="_blank" style="color: {border_color}; font-weight: 600;">{source}</a>' if url else f'<strong>{source}</strong>'
    updated_html = ""
    if updated is not None:
        updated_html = (
            f'<br><small style="opacity: 0.85;">'
            f'<strong>Updated:</strong> {format_last_updated(updated)}</small>'
        )
    st.markdown(f"""
    <div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>Data Source:</strong> {link_html}<br>
        <small style="opacity: 0.85;">{description}</small>{updated_html}
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


def is_rotowire_url_stale(url: str, current_gw: int) -> bool:
    """Check whether a Rotowire rankings article URL covers the current
    gameweek, so the app doesn't display a previous gameweek's projections
    as if they were current.

    Handles both a single-GW article ("...-gameweek-26-...") and a
    preseason range-preview article ("...-gameweeks-1-5-..."), where the
    article is fresh for any GW within its range. Returns False (not stale)
    if the URL doesn't match either pattern, since there's nothing to
    compare against — that's a different failure mode than "unknown but
    fine," but it's already handled separately by "no URL at all."
    """
    import re
    url_lower = url.lower()
    range_match = re.search(r'gameweeks-(\d+)-(\d+)', url_lower)
    if range_match:
        gw_start, gw_end = int(range_match.group(1)), int(range_match.group(2))
        return not (gw_start <= current_gw <= gw_end)
    single_match = re.search(r'gameweek-(\d+)', url_lower)
    if single_match:
        return int(single_match.group(1)) != current_gw
    return False


def render_rotowire_projections():
    """Render the Rotowire player projections tab."""
    # Resolve the URL before the banner so the banner can report when that
    # specific article was last updated.
    if config.ROTOWIRE_URL:
        url = config.ROTOWIRE_URL
    else:
        url = rotowire_url_selector()

    _render_source_banner(
        "Rotowire",
        "Weekly gameweek projections based on expert analysis, matchups, and form.",
        "#f0f9ff", "#0ea5e9",
        "https://www.rotowire.com/soccer/",
        updated=get_rotowire_article_updated(url) if url else None,
    )

    if not url:
        st.info("No Rotowire URL available. Please configure one above.")
        return

    current_gw = config.CURRENT_GAMEWEEK

    # Block stale data - don't show previous GW projections
    if is_rotowire_url_stale(url, current_gw):
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

    # Add value column if we have Points and Price. Rotowire occasionally lists a
    # player with no price (Price 0.0), so this must guard the division the same
    # way the scraper does -- an unguarded points/price yields inf, which is not a
    # value, sorts to the top, and used to crash the colour scale outright.
    if 'Points' in df.columns and 'Price' in df.columns:
        _price = pd.to_numeric(df['Price'], errors='coerce')
        df['Value'] = (pd.to_numeric(df['Points'], errors='coerce') / _price).where(_price > 0)

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
    display_gw = current_gw
    st.markdown(f"#### GW{display_gw} Player Projections")

    # Sort controls
    sortable_cols = [c for c in result.columns if c != 'Player']
    sort_col1, sort_col2 = st.columns([2, 1])
    with sort_col1:
        default_sort = 'Points' if 'Points' in sortable_cols else sortable_cols[0]
        sort_by = st.selectbox("Sort by", sortable_cols, index=sortable_cols.index(default_sort), key="rw_sort_col")
    with sort_col2:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="rw_sort_order")

    result = result.sort_values(sort_by, ascending=(sort_order == "Ascending"), na_position="last")

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
    feed = get_ffp_feed()
    _render_source_banner(
        "Fantasy Football Pundit",
        "Player data with start probability, ownership, and betting-derived probabilities.",
        "#fef3c7", "#f59e0b",
        FFP_POINTS_PREDICTOR_URL,
        updated=feed.updated,
    )

    if not _ffp_gate(feed):
        return
    raw_df = feed.df

    # Check which columns actually have data (non-zero values)
    prediction_cols = ['Predicted', 'StartingPredicted', 'Next2GWs', 'Next3GWs', 'Next6GWs']
    has_predictions = any(
        col in raw_df.columns and (raw_df[col] != 0).any()
        for col in prediction_cols
    )

    # Render the common name, not the legal one. FFP republishes the bootstrap's
    # full name ("Igor Thiago Nascimento Rodrigues"), which is what the merges
    # key on and not what anyone reads. The spreadsheet fallback carries no
    # Display_Name, so fall back to Name there.
    name_col = 'Display_Name' if 'Display_Name' in raw_df.columns else 'Name'
    base_cols = [name_col, 'Team', 'Position', 'Fixture', 'Price', 'Ownership', 'Start']

    if has_predictions:
        display_cols = base_cols + ['Predicted', 'StartingPredicted', 'Next2GWs', 'Next3GWs', 'Next6GWs']
        display_names = {
            name_col: 'Player', 'Ownership': 'Own %', 'Start': 'Start %',
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
            name_col: 'Player', 'Ownership': 'Own %', 'Start': 'Start %',
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
    st.markdown(f"#### GW{feed.gameweek or config.CURRENT_GAMEWEEK} Player Data")

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
    st.caption(f"Showing {len(result)} of {len(df)} players. Data from Fantasy Football Pundit"
               + (f" · {_ffp_caption(feed)}." if _ffp_caption(feed) else "."))


# =============================================================================
# Goal Scorer & Assist Odds Tab
# =============================================================================

def render_goalscorer_odds():
    """Render the goal scorer and assist odds tab."""
    feed = get_ffp_feed()
    _render_source_banner(
        "Fantasy Football Pundit (Betting Odds)",
        "Anytime goalscorer and assist probabilities converted from bookmaker odds.",
        "#fefce8", "#eab308",
        FFP_GOAL_ASSIST_URL,
        updated=feed.updated,
    )

    if not _ffp_gate(feed):
        return

    df = get_ffp_goalscorer_odds()

    if df is None or df.empty:
        st.warning("Goalscorer odds are not in the current Fantasy Football Pundit feed.")
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
    st.markdown(f"#### GW{feed.gameweek or config.CURRENT_GAMEWEEK} Goalscorer & Assist Probabilities")

    positive_cols = [c for c in ['Goal %', 'Assist %', 'Return %', 'Start %'] if c in display_df.columns]
    col_fmts = {col: '{:.0f}%' for col in positive_cols}

    render_styled_table(
        display_df,
        col_formats=col_fmts,
        positive_color_cols=positive_cols,
        max_height=600,
    )
    st.caption(f"Showing {len(result)} of {len(df)} players. Betting odds converted to probabilities"
               + (f" · {_ffp_caption(feed)}." if _ffp_caption(feed) else "."))


# =============================================================================
# Clean Sheet Odds Tab
# =============================================================================

def render_clean_sheet_odds():
    """Render the clean sheet odds tab with horizontal bar visualization."""
    feed = get_ffp_feed()
    _render_source_banner(
        "Fantasy Football Pundit (Clean Sheet Odds)",
        "Team clean sheet probabilities from betting markets.",
        "#f0fdf4", "#22c55e",
        FFP_CLEAN_SHEET_URL,
        updated=feed.updated,
    )

    if not _ffp_gate(feed):
        return

    df = get_ffp_clean_sheet_odds()

    if df is None or df.empty:
        st.warning("Clean sheet odds are not in the current Fantasy Football Pundit feed.")
        return

    st.markdown(f"#### GW{feed.gameweek or config.CURRENT_GAMEWEEK} Clean Sheet Probabilities")

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

        st.caption(f"Showing {len(df)} teams. Betting odds converted to probabilities"
                   + (f" · {_ffp_caption(feed)}." if _ffp_caption(feed) else "."))
    else:
        st.warning("Data format not as expected.")


# =============================================================================
# Match Odds Tab (The Odds API)
# =============================================================================

def _render_match_card_native(home_team: str, away_team: str, kickoff: str, home_pct: float, draw_pct: float, away_pct: float):
    """Render a single match card with dark theme and proportional probability bar."""
    from zoneinfo import ZoneInfo

    # Format kickoff time in EST with full format
    try:
        kickoff_dt = pd.to_datetime(kickoff)
        if kickoff_dt.tzinfo is None:
            kickoff_dt = kickoff_dt.replace(tzinfo=ZoneInfo('UTC'))
        kickoff_est = kickoff_dt.astimezone(ZoneInfo('America/New_York'))

        day_name = kickoff_est.strftime("%A")
        month = kickoff_est.strftime("%B")
        day = kickoff_est.day
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        time_str = kickoff_est.strftime("%I:%M%p").lstrip('0').lower()
        kickoff_str = f"{day_name}, {month} {day}{suffix}, {time_str} EST"
    except Exception:
        kickoff_str = str(kickoff) if kickoff else ""

    # Ensure percentages sum to ~100 and are positive
    total = max(home_pct + draw_pct + away_pct, 1)
    h_w = home_pct / total * 100
    d_w = draw_pct / total * 100
    a_w = away_pct / total * 100

    # Bar segment colors: home=emerald, draw=slate, away=indigo
    home_bar_color = "#10b981"
    draw_bar_color = "#64748b"
    away_bar_color = "#6366f1"

    # Font size: smaller for narrow segments
    h_font = "11px" if h_w < 18 else "14px"
    d_font = "11px" if d_w < 18 else "13px"
    a_font = "11px" if a_w < 18 else "14px"

    card_html = f'''
    <div style="border:1px solid #333;border-radius:10px;padding:16px;margin-bottom:14px;background:#1a1a2e;">
        <div style="text-align:center;color:#9ca3af;font-size:12px;margin-bottom:10px;">{kickoff_str}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div>
                <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.5px;">Home</div>
                <div style="font-weight:700;font-size:17px;color:#ffffff;">{home_team}</div>
            </div>
            <div style="color:#555;font-size:12px;">vs</div>
            <div style="text-align:right;">
                <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.5px;">Away</div>
                <div style="font-weight:700;font-size:17px;color:#ffffff;">{away_team}</div>
            </div>
        </div>
        <div style="display:flex;border-radius:6px;overflow:hidden;height:30px;">
            <div style="width:{h_w:.1f}%;background:{home_bar_color};display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:{h_font};">{home_pct:.0f}%</div>
            <div style="width:{d_w:.1f}%;background:{draw_bar_color};display:flex;align-items:center;justify-content:center;color:#fff;font-size:{d_font};">{draw_pct:.0f}%</div>
            <div style="width:{a_w:.1f}%;background:{away_bar_color};display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:{a_font};">{away_pct:.0f}%</div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7280;margin-top:4px;">
            <span style="color:{home_bar_color};">Home</span>
            <span style="color:{draw_bar_color};">Draw</span>
            <span style="color:{away_bar_color};">Away</span>
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
# Blended projections — the number the rest of the app actually uses
# =============================================================================

def _build_pool(bootstrap) -> pd.DataFrame:
    """The canonical player universe for the blend, from the FPL bootstrap.

    Carries both name forms on purpose: ``Player`` is the full legal name that
    projection sources are matched against, ``Display_Name`` is what a human
    calls him. Match on Player, display Display_Name -- swapping them silently
    degrades match rates.
    """
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    rows = []
    for e in bootstrap.get("elements", []):
        first, second = e.get("first_name", ""), e.get("second_name", "")
        rows.append({
            "Player_ID": e.get("id"),
            "Player": f"{first} {second}".strip(),
            "Display_Name": to_display_name(first, second, e.get("web_name", "")),
            "Web_Name": e.get("web_name", ""),
            "Team": teams.get(e.get("team"), ""),
            "Position": pos_map.get(e.get("element_type"), "M"),
            "ep_next": pd.to_numeric(e.get("ep_next"), errors="coerce"),
            "status": e.get("status"),
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
        })
    return pd.DataFrame(rows)


def _blended_source_status(rotowire_df, feed, blended, current_gw) -> None:
    """One row per source: is it up, how many players, how fresh, what weight.

    Modelled on the Initial Squad Optimizer's Data Sources panel, which exists
    because every fetch in this app degrades to an empty frame on failure --
    without a panel like this a broken source renders a page identical to a
    working one, just with every number quietly leaning on whatever survived.
    """
    weights = getattr(config, "PROJECTION_SOURCE_WEIGHTS", {}) or {}

    def _row(name, ok, rows, note, updated, weight_key):
        w = float(weights.get(weight_key, 0.0))
        if not ok or not rows:
            status = "🔴 Unavailable"
        elif note:
            status = f"🟡 {note}"
        else:
            status = "🟢 OK"
        return {
            "Source": name,
            "Status": status,
            "Players": int(rows or 0),
            "Weight": f"{w:.0%}" if w > 0 else "fallback only",
            "Updated": format_last_updated(updated) if updated else "—",
        }

    rows = []
    rw_ok = rotowire_df is not None and not rotowire_df.empty
    rows.append(_row(
        "Rotowire", rw_ok, len(rotowire_df) if rw_ok else 0,
        "" if rw_ok else "no rankings table for this gameweek",
        get_rotowire_article_updated(config.ROTOWIRE_URL) if config.ROTOWIRE_URL else None,
        "rotowire",
    ))
    ffp_note = ""
    if feed.ok and feed.is_stale(current_gw):
        ffp_note = f"published for GW{feed.gameweek} — excluded"
    elif feed.ok and feed.provenance == "sheet":
        ffp_note = "read from their spreadsheet"
    rows.append(_row(
        "Fantasy Football Pundit", feed.ok, len(feed.df) if feed.ok else 0,
        ffp_note, feed.updated, "ffp",
    ))
    _n_xp = int((blended["Proj_Src"] == "xP").sum()) if "Proj_Src" in blended.columns else 0
    rows.append(_row(
        "FPL expected points (ep_next)", True, _n_xp,
        "" if _n_xp else "not needed — every player priced by another source",
        None, "fpl_ep",
    ))

    st.markdown("##### Data Sources")
    render_styled_table(pd.DataFrame(rows))
    st.caption(
        "Weight is each source's share of the blend where it prices a player; "
        "weights are renormalised over whichever sources are actually available, "
        "so a missing source does not shrink the projection. FPL's own expected "
        "points fills only players nobody else priced."
    )


def render_blended_projections():
    """The blend, with the sources that produced it on the same row.

    This is the number every other page in the app now renders. Until this tab
    existed the Projections Hub showed Rotowire and FFP in separate tabs with no
    join between them, so the one figure the app actually scores on appeared
    nowhere, and a user could not see where two sources disagreed.
    """
    current_gw = config.CURRENT_GAMEWEEK
    rotowire_url = config.ROTOWIRE_URL

    rotowire_df = pd.DataFrame()
    if rotowire_url:
        rotowire_df = get_rotowire_player_projections(rotowire_url)
    feed = get_ffp_feed()

    if (rotowire_df is None or rotowire_df.empty) and not feed.ok:
        st.warning(
            f"No projections available for GW{current_gw}. Rotowire has not "
            "published a rankings table and Fantasy Football Pundit could not "
            "be read."
        )
        return

    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        st.error("Could not load the FPL player pool, so projections cannot be assembled.")
        return

    pool = _build_pool(bootstrap)
    if pool.empty:
        st.error("The FPL bootstrap carried no players.")
        return

    # Rotowire values land on the pool by name (merge_season_projections is the
    # app's tiered name-matched merge); FFP joins on the element id inside
    # blend_projections_onto.
    pool = merge_season_projections(pool, rotowire_df, output_col="Points")
    blended = blend_projections_onto(pool, feed.df, expected_gw=current_gw)

    _blended_source_status(rotowire_df, feed, blended, current_gw)

    st.markdown(f"#### GW{current_gw} Blended Projections")
    st.caption(
        "**Proj** is expected points — what he is worth allowing for the chance "
        "he does not start. **If Starts** is the same projection assuming he "
        "does. **Spread** is the gap between the sources: a large one means they "
        "disagree about this player, not that he is a bad pick."
    )

    display = pd.DataFrame({
        "Player": blended.get("Display_Name", blended["Player"]),
        "Team": blended["Team"],
        "Pos": blended["Position"],
        "Proj": pd.to_numeric(blended["Proj"], errors="coerce"),
        "If Starts": pd.to_numeric(blended["Proj_Start"], errors="coerce"),
        "Start %": pd.to_numeric(blended["Start_Pct"], errors="coerce") * 100,
        "Rotowire": pd.to_numeric(blended.get("Proj_Start__rotowire"), errors="coerce"),
        "FFP": pd.to_numeric(blended.get("Proj_Start__ffp"), errors="coerce"),
        "FPL xP": pd.to_numeric(blended.get("Proj_Start__fpl_ep"), errors="coerce"),
        "Spread": pd.to_numeric(blended.get("Proj_Spread"), errors="coerce"),
        "Src": blended["Proj_Src"],
    })
    display = display[display["Proj"].notna()]

    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("Search player", key="blend_search")
        with col2:
            positions = st.multiselect(
                "Position", ["G", "D", "M", "F"], default=[], key="blend_pos"
            )
        only_disagree = st.checkbox(
            "Only where sources disagree by 1.5+ points", key="blend_disagree",
            help="Where Rotowire and FFP are far apart the blend is doing real "
                 "work, and the pick deserves a second look.",
        )

    if search:
        display = display[display["Player"].str.contains(search, case=False, na=False)]
    if positions:
        display = display[display["Pos"].isin(positions)]
    if only_disagree:
        display = display[display["Spread"].fillna(0) >= 1.5]

    display = display.sort_values("Proj", ascending=False).reset_index(drop=True)
    display.insert(0, "#", range(1, len(display) + 1))

    render_styled_table(
        display.head(300),
        col_formats={
            "Proj": "{:.2f}", "If Starts": "{:.2f}", "Start %": "{:.0f}%",
            "Rotowire": "{:.2f}", "FFP": "{:.2f}", "FPL xP": "{:.2f}",
            "Spread": "{:.2f}",
        },
        positive_color_cols=["Proj", "If Starts"],
        negative_color_cols=["Spread"],
        max_height=600,
    )
    st.caption(
        f"Showing {min(len(display), 300)} of {len(display)} projected players. "
        "The per-source columns are on the *if he starts* basis, so they are "
        "directly comparable with each other and with **If Starts**."
    )


# =============================================================================

def show_player_projections_page():
    """Main projections hub page with tabbed interface."""
    st.title("Projections Hub")
    st.caption(
        "The **Blended** tab is the projection the rest of the app uses. The "
        "other tabs show each source on its own, unblended."
    )

    # Create tabs for different data sources
    # Blended leads. The individual sources stay one click away rather than
    # being hidden: the blend is the answer, but seeing what fed it is how you
    # judge whether to trust it on a particular player.
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Blended",
        "Rotowire",
        "FFP Data",
        "Goal/Assist Odds",
        "Clean Sheet Odds",
        "Match Odds"
    ])

    with tab0:
        render_blended_projections()

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
        - **Proportional probability bar** where segment width = likelihood:
          - Green = Home win probability
          - Grey = Draw probability
          - Purple = Away win probability
        - **Kickoff times** shown in Eastern Time (EST/EDT)
        """)
