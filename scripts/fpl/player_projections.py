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

import math

import config
import pandas as pd
import streamlit as st
from scripts.common.text_helpers import format_last_updated
from scripts.common.scraping import (
    FFP_CLEAN_SHEET_URL,
    FFP_GOAL_ASSIST_URL,
    FFP_POINTS_PREDICTOR_URL,
    get_ffp_feed,
    get_ffp_horizon,
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
from scripts.common.text_helpers import compact_html
from scripts.common.analytics import blend_projections_onto, merge_season_projections
from scripts.common.fpl_classic_api import get_classic_bootstrap_static
from scripts.common import projection_accuracy, projection_archive
from scripts.common.data_validation import check_ffp_horizon
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.projections_hub")


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
    elif feed.provenance == "archive":
        # Every other page says this via render_ffp_status(). The Hub did not,
        # so an archived table read here exactly like a live one -- and FFP
        # rolls its window forward mid-gameweek, so this is the normal state
        # for most of every week rather than an edge case.
        bits.append("from our archive of FFP's last publication for this gameweek")
        if feed.note:
            bits.append(feed.note)
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
# Horizon — FFP's six-gameweek forecasts, which nothing has ever shown
# =============================================================================

def _is_finite_number(val) -> bool:
    """True only for a real, finite number -- NaN and inf are not averageable."""
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _horizon_fixture_cell(fixtures, count, fdr=None) -> str:
    """One cell for the fixtures view: the opponent(s) and the difficulty.

    A blank gameweek must not render as an empty string beside a column of real
    fixtures -- it reads as missing data rather than as the fact that the club
    is not playing, which is the single most important thing a horizon can tell
    you. Same reasoning as ``attach_matchups()`` writing "No fixture" rather
    than leaving the cell to be read as "N/A".

    **The FDR is printed, not only coloured.** ``render_styled_table`` colours
    *text*, not the cell background, so difficulty encoded as colour alone is
    the weakest signal on the page and carries nothing at all for a colour-blind
    reader. The number is the data; the colour is the shortcut.
    """
    if not fixtures:
        return "—"
    label = " / ".join(fixtures)
    try:
        if int(count) > len(fixtures):
            # FFP priced a double but published one opponent for it.
            label += " ×%d" % int(count)
    except (TypeError, ValueError):
        pass
    try:
        if fdr is not None and math.isfinite(float(fdr)):
            label += "  %g" % round(float(fdr), 1)
    except (TypeError, ValueError):
        pass
    return label


def _horizon_frames(horizon, blended, current_gw):
    """Wide points / fixture / FDR grids keyed on element id.

    FFP's horizon rows carry ``Player_ID`` resolved from the bootstrap ``code``,
    and the blend is keyed on the element id too, so this whole path joins on an
    integer. That is the point of FFP publishing ``player_code`` at all -- every
    other cross-source join in this app is a name match with a tier ladder
    behind it, and this one needs none.
    """
    h = horizon.df.copy()
    h = h[h["Player_ID"].notna()]
    if h.empty:
        return None, None, None, None
    h["Player_ID"] = pd.to_numeric(h["Player_ID"], errors="coerce").astype("int64")
    h["FFP_GW"] = pd.to_numeric(h["FFP_GW"], errors="coerce").astype("int64")

    # Expected points, which is the chosen basis: `Predicted` already carries
    # the start discount (the site's `predicted_points_start`). Summed, so a
    # double gameweek counts twice rather than being averaged away.
    points = h.pivot_table(index="Player_ID", columns="FFP_GW",
                           values="Predicted", aggfunc="sum")
    fdr = h.pivot_table(index="Player_ID", columns="FFP_GW",
                        values="Difficulty", aggfunc="mean")
    counts = h.pivot_table(index="Player_ID", columns="FFP_GW",
                           values="Fixture_Count", aggfunc="max")
    fixtures = (h.groupby(["Player_ID", "FFP_GW"])["Fixture"]
                 .apply(lambda s: [x for x in dict.fromkeys(s.dropna()) if x])
                 .unstack())

    # The current gameweek's column comes from the engine, not from FFP. These
    # are the same two calls the Blended tab makes, so the first column *is*
    # the number the rest of the app scores on rather than a re-derivation of
    # it that can drift -- the failure this codebase has paid for three times.
    if current_gw in points.columns and blended is not None and not blended.empty:
        engine = blended[blended["Player_ID"].notna()].copy()
        engine["Player_ID"] = pd.to_numeric(engine["Player_ID"], errors="coerce").astype("int64")
        proj = pd.to_numeric(engine.set_index("Player_ID")["Proj"], errors="coerce")
        points[current_gw] = proj.reindex(points.index).combine_first(points[current_gw])

    return points, fixtures, fdr, counts


def render_horizon():
    """FFP's forecasts for the next six gameweeks, as a player x gameweek grid.

    **The data for this has been fetched and discarded on every page load for a
    year.** FFP's payload carries six gameweeks of forecasts per player -- about
    2,274 rows -- and ``to_sheet_schema()`` keeps one gameweek's 347 of them,
    folding the rest into ten relative-offset columns that only
    ``blend_multi_gw_projections`` reads, and only as ``Next3GWs``.

    What that cost is the question this tab answers. A single-gameweek
    projection is a noisy read on a player's value, because most of its
    variance is the fixture: live at GW7, Haaland is 7.6 at home to Ipswich and
    4.6 away at Arsenal five weeks later, and Mbeumo swings 3.9 to 7.0 the other
    way. Every other projection surface in the app shows one of those numbers
    and one three-week aggregate, with no way to see which weeks carry it.
    """
    current_gw = config.CURRENT_GAMEWEEK
    horizon = get_ffp_horizon()

    _render_source_banner(
        "Fantasy Football Pundit — multi-gameweek forecasts",
        "Expected points per gameweek for the next six gameweeks, with each "
        "week's fixture.",
        "#fef3c7", "#f59e0b",
        FFP_POINTS_PREDICTOR_URL,
        updated=horizon.updated,
    )

    if not horizon.ok:
        st.warning(
            "No multi-gameweek forecasts could be read. "
            + (horizon.note or "Fantasy Football Pundit's site was unreachable "
                               "and we hold no archived forecasts for this gameweek.")
        )
        return

    # Plausibility, logged rather than raised: a horizon that looks wrong should
    # degrade the tab, never take the page down. The contiguity check is the one
    # worth having -- a gap shifts every later column one week left of its own
    # heading and every number in the grid stays perfectly plausible.
    for issue in check_ffp_horizon(horizon.df, horizon.start_gw,
                                   horizon.per_gw_provenance):
        if issue.severity == "error":
            _logger.error("FFP horizon: %s — %s", issue.message, issue.hint)
        else:
            _logger.info("FFP horizon: %s", issue.message)

    gws = horizon.gameweeks()
    if not gws:
        st.warning("The forecasts carry no gameweek, so they cannot be laid out "
                   "in gameweek order.")
        return

    # Column 1 is the blend -- the very frame the Blended sub-tab renders, not a
    # second assembly of it, so a player's first column matches that tab to the
    # digit and the work is done once per page load.
    blended = _blended_pool(current_gw, config.ROTOWIRE_URL)
    if blended is None:
        blended = pd.DataFrame()

    built = _horizon_frames(horizon, blended, current_gw)
    if built[0] is None:
        st.warning("None of FFP's forecast rows resolved to an FPL player, so "
                   "there is nothing to join the grid onto.")
        return
    points, fixtures, fdr, counts = built

    # Identity comes from the blend where we have it (it carries Display_Name
    # for the whole pool) and from FFP's own rows otherwise.
    ident = (horizon.df[horizon.df["FFP_GW"] == gws[0]]
             .dropna(subset=["Player_ID"]).copy())
    ident["Player_ID"] = pd.to_numeric(ident["Player_ID"], errors="coerce").astype("int64")
    ident = ident.set_index("Player_ID")[["Display_Name", "Team", "Position"]]
    if not blended.empty:
        b = blended[blended["Player_ID"].notna()].copy()
        b["Player_ID"] = pd.to_numeric(b["Player_ID"], errors="coerce").astype("int64")
        b = b.set_index("Player_ID")
        ident = pd.DataFrame({
            "Display_Name": b.get("Display_Name", b.get("Player")).reindex(points.index)
                             .combine_first(ident["Display_Name"].reindex(points.index)),
            "Team": b["Team"].reindex(points.index).combine_first(
                ident["Team"].reindex(points.index)),
            "Position": b["Position"].reindex(points.index).combine_first(
                ident["Position"].reindex(points.index)),
        })
    ident = ident.reindex(points.index)

    # --- Controls ----------------------------------------------------------
    max_weeks = len(gws)
    c1, c2 = st.columns([1, 2])
    with c1:
        weeks = st.select_slider(
            "Gameweeks", options=list(range(2, max_weeks + 1)), value=max_weeks,
            key="horizon_weeks",
            help="How far ahead to show. The total is over exactly these weeks.",
        ) if max_weeks > 2 else max_weeks
    with c2:
        view = st.radio("Show", ["Points", "Fixtures"], horizontal=True,
                        key="horizon_view",
                        help="Points is expected points per gameweek. Fixtures "
                             "shows each week's opponent, coloured by difficulty.")

    shown = [g for g in gws if g in points.columns][:int(weeks)]
    gw_labels = {g: "GW%d" % g for g in shown}

    grid = pd.DataFrame({
        "Player": ident["Display_Name"].fillna(""),
        "Team": ident["Team"].fillna(""),
        "Pos": ident["Position"].fillna(""),
    }, index=points.index)
    for g in shown:
        grid[gw_labels[g]] = pd.to_numeric(points[g], errors="coerce")
    label_cols = [gw_labels[g] for g in shown]
    grid["Total"] = grid[label_cols].sum(axis=1, min_count=1)

    # A player with nothing but zeros across the whole window is a declared
    # non-starter, not a row worth a line on a board about long-term value.
    grid = grid[grid[label_cols].fillna(0).abs().sum(axis=1) > 0]

    # One colour range, taken over every gameweek column of the *whole* board
    # and fixed before any filtering. Two things depend on that. Per-column
    # ranges would make each week self-normalising, so a 6.8 in an easy week and
    # a 6.8 in a hard one would render differently -- which destroys the only
    # thing a grid is for, which is reading across a row. And a range re-derived
    # from the filtered rows would grade a selection against itself: filter to
    # one club and its best player goes pure green whatever he is worth, which
    # is the trap render_styled_table's own docstring documents.
    finite = pd.concat([pd.to_numeric(grid[c], errors="coerce")
                        for c in label_cols]).replace(
        [float("inf"), float("-inf")], float("nan")).dropna()
    span = (float(finite.min()), float(finite.max())) if not finite.empty else None

    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        search = f1.text_input("Search by name", placeholder="e.g., Salah",
                               key="horizon_search")
        positions = f2.multiselect(
            "Position", sorted(p for p in grid["Pos"].dropna().unique() if p),
            default=[], key="horizon_pos")
        teams = f3.multiselect(
            "Team", sorted(t for t in grid["Team"].dropna().unique() if t),
            default=[], key="horizon_team",
            help="Leave empty for every club.")
        s1, s2 = st.columns(2)
        sort_by = s1.selectbox("Sort by", ["Total"] + label_cols, key="horizon_sort")
        descending = s2.selectbox("Order", ["Highest first", "Lowest first"],
                                  key="horizon_order") == "Highest first"

    if search:
        grid = grid[grid["Player"].str.contains(search, case=False, na=False)]
    if positions:
        grid = grid[grid["Pos"].isin(positions)]
    if teams:
        grid = grid[grid["Team"].isin(teams)]

    grid = grid.sort_values(sort_by, ascending=not descending)
    if grid.empty:
        st.info("No players match those filters.")
        return

    ids = list(grid.index)

    st.markdown("#### GW%d–GW%d %s" % (
        shown[0], shown[-1],
        "expected points" if view == "Points" else "fixtures"))

    if view == "Points":
        display = grid.reset_index(drop=True)
        display.insert(0, "#", range(1, len(display) + 1))
        render_styled_table(
            display.head(300),
            col_formats={**{c: "{:.1f}" for c in label_cols}, "Total": "{:.1f}"},
            positive_color_cols=label_cols + ["Total"],
            color_range_overrides=({c: span for c in label_cols} if span else None),
            max_height=640,
        )
    else:
        rows = []
        colour = {c: [] for c in label_cols}
        for pid in ids:
            row = {"Player": grid.at[pid, "Player"], "Team": grid.at[pid, "Team"],
                   "Pos": grid.at[pid, "Pos"]}
            week_fdrs = []
            for g in shown:
                fx = fixtures[g].get(pid) if g in fixtures.columns else None
                cnt = counts[g].get(pid) if g in counts.columns else None
                this_fdr = fdr[g].get(pid) if g in fdr.columns else float("nan")
                row[gw_labels[g]] = _horizon_fixture_cell(
                    fx if isinstance(fx, list) else [], cnt, this_fdr)
                colour[gw_labels[g]].append(this_fdr)
                week_fdrs.append(this_fdr)
            # Both summaries, because this view answers two questions that can
            # disagree: how hard the run is, and what it is projected to be
            # worth. A soft run for a player who does not start is still worth
            # nothing, and a brutal run for a premium can still out-score it.
            finite_fdrs = [f for f in week_fdrs
                           if f is not None and _is_finite_number(f)]
            row["Avg FDR"] = (sum(finite_fdrs) / len(finite_fdrs)
                              if finite_fdrs else float("nan"))
            row["Total"] = grid.at[pid, "Total"]
            rows.append(row)
        display = pd.DataFrame(rows)
        display.insert(0, "#", range(1, len(display) + 1))
        display = display.head(300)
        render_styled_table(
            display,
            col_formats={"Total": "{:.1f}", "Avg FDR": "{:.2f}"},
            positive_color_cols=["Total"],
            # Colour comes from FFP's own published difficulty rather than the
            # text in the cell, and 1-5 is fixed so the scale means the same
            # thing in every column and on every rerun. Avg FDR shares that
            # scale, so a column average and a single week read alike.
            negative_color_cols=label_cols + ["Avg FDR"],
            color_values={c: colour[c][:len(display)] for c in label_cols},
            color_range_overrides={**{c: (1.0, 5.0) for c in label_cols},
                                   "Avg FDR": (1.0, 5.0)},
            text_align={c: "center" for c in label_cols},
            max_height=640,
        )
        st.caption("Each cell is the opponent, **H** home or **a** away, then "
                   "FFP's own fixture difficulty — 1 easiest, 5 hardest, green "
                   "to red. **—** is a blank gameweek; **×2** a double. "
                   "**Total** is the same projected points as the Points view, "
                   "and rows keep the order set above, so the two views read "
                   "against each other line for line.")

    bits = [
        "Showing %d of %d players." % (min(len(grid), 300), len(grid)),
        "**GW%d is the app's blended projection**, from Rotowire and FFP with "
        "FPL's own expected points filling players neither priced; GW%d onwards "
        "are FFP's forecasts alone, as the only source publishing beyond the "
        "next gameweek." % (shown[0], shown[0] + 1)
        if shown[0] == current_gw and len(shown) > 1 else
        "Every column is FFP's own forecast.",
        "Both are expected points — allowing for the chance he does not start — "
        "so a row sums. **Total** therefore differs slightly from FFP's own "
        "\"Next %d GW\" on the FFP Data tab, which uses FFP for the first week too."
        % len(shown),
    ]
    archived = horizon.archived_gameweeks()
    if archived:
        bits.append("From our archive: %s."
                    % ", ".join("GW%d" % g for g in archived))
    if horizon.updated is not None:
        bits.append("FFP published %s." % format_last_updated(horizon.updated))
    st.caption(" ".join(bits))


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
            # Carries the return date the horizon model reads -- "Expected back
            # 11 Oct" is three gameweeks at GW6.
            "news": e.get("news"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def _blended_pool(current_gw: int, rotowire_url: str):
    """The blended projection for ``current_gw``, built once per page load.

    Two tabs need this frame -- GW Projections renders it directly and Horizon
    takes its first column from it -- and ``st.tabs`` runs every tab body on
    every rerun, so without a cache the same blend is assembled twice on each
    interaction. Measured warm, that was 4.4s for the first and a further ~0.9s
    for the second; the name-matched ``merge_season_projections`` is most of it.

    Cached on ``(gameweek, rotowire_url)`` rather than on the frames it reads:
    both are small and hashable, and the fetches inside are themselves cached,
    so a miss here is cheap. The TTL matches ``get_ffp_feed``'s so the blend and
    the feed it was built from roll over together rather than the blend pinning
    a feed that has since been refreshed.

    ``st.cache_data`` returns a fresh object per call, so a caller mutating the
    result cannot corrupt the other's copy -- which ``cache_resource`` would.
    """
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        return pd.DataFrame()
    pool = _build_pool(bootstrap)
    if pool.empty:
        return pd.DataFrame()

    rotowire_df = pd.DataFrame()
    if rotowire_url:
        rotowire_df = get_rotowire_player_projections(rotowire_url)
    feed = get_ffp_feed()

    # Rotowire values land on the pool by name (merge_season_projections is the
    # app's tiered name-matched merge); FFP joins on the element id inside
    # blend_projections_onto.
    pool = merge_season_projections(pool, rotowire_df, output_col="Points")
    return blend_projections_onto(pool, feed.df, expected_gw=current_gw)


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
    elif feed.ok and feed.provenance == "archive":
        ffp_note = "from our archive of their last publication"
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

    blended = _blended_pool(current_gw, rotowire_url)
    if blended is None or blended.empty:
        st.error("Could not load the FPL player pool, so projections cannot be assembled.")
        return

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
        # FPL's own xP is deliberately not a column. It carries weight 0, so it
        # never moves the blend for a player either weighted source priced --
        # live, 29 of 667 -- and a column sitting beside Rotowire and FFP read
        # as a third contributor to it. It is still a declared source: `Src`
        # shows `xP` where it filled a player nobody else priced, the Data
        # Sources panel counts them, and the Accuracy tab scores it.
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
            "Rotowire": "{:.2f}", "FFP": "{:.2f}",
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

# =============================================================================
# Accuracy — how wrong has each source actually been?
# =============================================================================

_SOURCE_LABELS = {
    "blend": "Blended (what the app uses)",
    "rotowire": "Rotowire",
    "ffp": "Fantasy Football Pundit",
    "fpl_ep": "FPL expected points",
}


def render_accuracy():
    """Score archived projections against what actually happened.

    The 60/40 Rotowire/FFP split was an assumption for as long as this app has
    existed, because no projection was ever written down. This tab is what turns
    it into a measurement -- and it is deliberately cautious about saying so,
    because a confident-looking table built on two gameweeks would be worse than
    no table at all.
    """
    scored = projection_accuracy.score_archive(common_subset=True)
    note = projection_accuracy.confidence_note(scored)

    if scored.empty:
        st.info(note)
        pre = projection_archive.list_pre()
        act = projection_archive.list_actuals()
        st.caption(
            "Snapshots held: projections for %s, actual points for %s. Both "
            "halves are needed before a gameweek can be scored."
            % (_gw_list(pre), _gw_list(act))
        )
        return

    gws = sorted(scored["gameweek"].unique())
    if scored["gameweek"].nunique() < projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE:
        st.warning(note)
    else:
        st.caption(note)

    if (scored["capture"] == "backfill").any():
        backfilled = sorted(scored.loc[scored["capture"] == "backfill", "gameweek"].unique())
        st.caption(
            "⚠️ %s captured after the deadline. A backfill can see team news — in "
            "the limit, lineups — that no manager had when they picked, so it "
            "flatters every source in it. Shown here, but never used to fit "
            "blend weights." % _gw_list(backfilled)
        )

    _render_verdict()

    st.markdown("#### Error by source")
    st.caption(
        "**Starters** compares each source's *if he starts* projection against "
        "what players who actually started scored — it isolates the points model "
        "from the minutes model. Every source is scored on the same players, "
        "because Rotowire prices only expected starters and would otherwise be "
        "judged on a harder population than the rest. **MAE** is the average "
        "miss in points; **bias** is signed, so a source that is consistently "
        "high reads positive."
    )

    summary = projection_accuracy.summarise(scored)
    starters = summary[summary["scope"] == projection_accuracy.SCOPE_STARTERS].copy()
    if not starters.empty:
        show = pd.DataFrame({
            "Source": starters["source"].map(_SOURCE_LABELS).fillna(starters["source"]),
            "GWs": starters["gameweeks"],
            "Players scored": starters["n"],
            "Coverage": starters["coverage"] * 100,
            "MAE": starters["mae"],
            "RMSE": starters["rmse"],
            "Bias": starters["bias"],
            "Rank corr": starters["spearman"],
        })
        render_styled_table(
            show.reset_index(drop=True),
            col_formats={"Coverage": "{:.0f}%", "MAE": "{:.2f}", "RMSE": "{:.2f}",
                         "Bias": "{:+.2f}", "Rank corr": "{:.2f}"},
            negative_color_cols=["MAE", "RMSE"],
        )
        st.caption(
            "Coverage is the share of the player pool each source priced at all — "
            "a real property of the source, not a flaw. Rotowire's low number is "
            "it listing only expected starters."
        )

    ev = summary[summary["scope"] == projection_accuracy.SCOPE_ALL]
    if not ev.empty:
        row = ev.iloc[0]
        st.markdown("#### End-to-end")
        st.caption(
            "The blended **expected points** against actual points across every "
            "player, including those who never played and scored zero — a "
            "projection error like any other. This is the number the app is "
            "actually judged on."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{row['mae']:.2f}")
        c2.metric("Bias", f"{row['bias']:+.2f}")
        c3.metric("Rank correlation", f"{row['spearman']:.2f}")

    _render_start_model(scored)

    with st.expander("Per gameweek"):
        detail = scored.copy()
        detail["Source"] = detail["source"].map(_SOURCE_LABELS).fillna(detail["source"])
        detail = detail[["gameweek", "Source", "scope", "capture", "n",
                         "mae", "bias", "spearman"]]
        detail.columns = ["GW", "Source", "Scope", "Capture", "N", "MAE", "Bias", "Rank corr"]
        render_styled_table(
            detail.reset_index(drop=True),
            col_formats={"MAE": "{:.2f}", "Bias": "{:+.2f}", "Rank corr": "{:.2f}"},
        )

    _render_weight_fit(scored)


def _render_verdict():
    """Which source is actually best -- the answer, before the evidence.

    Four rows of numbers to three decimal places is not a conclusion, and the
    reader's question is "so which one should I believe". Sources are compared
    *pairwise* on the players both priced, because coverage runs from a third of
    the pool to all of it, and *paired* per player, which is what lets two
    gameweeks separate anything at all: differencing per player removes the
    shared difficulty of the week, so the interval on the difference is far
    narrower than the interval on either mean.
    """
    questions = [
        (projection_accuracy.SCOPE_START, "Who starts", "Brier", "{:.3f}"),
        (projection_accuracy.SCOPE_STARTERS, "Points if he starts", "MAE", "{:.2f}"),
    ]
    verdicts = [(label, metric, fmt, projection_accuracy.verdict(scope))
                for scope, label, metric, fmt in questions]
    if not any(v.get("best") for _, _, _, v in verdicts):
        return

    st.markdown("#### The short version")
    cols = st.columns(len(verdicts))
    for col, (label, metric, fmt, v) in zip(cols, verdicts):
        with col:
            if not v.get("best"):
                st.info(v.get("note") or "Not scoreable yet.")
                continue
            name = _SOURCE_LABELS.get(v["best"], v["best"])
            contenders = [_SOURCE_LABELS.get(c, c) for c in v["contenders"]]
            if contenders:
                tail = "level with " + _join(contenders)
            else:
                tail = "clear of every other source"
            st.markdown(
                compact_html(
                    f"""
                    <div style="background:#16213e;border:1px solid #2a2f52;
                                border-radius:10px;padding:14px 16px;color:#e0e0e0;">
                      <div style="font-size:0.78rem;letter-spacing:.06em;
                                  text-transform:uppercase;color:#8fa3c8;">{label}</div>
                      <div style="font-size:1.25rem;font-weight:700;margin:.25rem 0;
                                  color:#7ee787;">{name}</div>
                      <div style="font-size:0.85rem;color:#c8d2e4;">
                        {metric} {fmt.format(v['value'])} · {tail}
                      </div>
                      <div style="font-size:0.78rem;color:#8fa3c8;margin-top:.3rem;">
                        led in {v['leads_in']} of {v['gameweeks']}
                        gameweek{'' if v['gameweeks'] == 1 else 's'} ·
                        {v['n']:,} scored
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    st.caption(
        "Decided by comparing sources **in pairs, on the players both of them "
        "priced** — coverage runs from a third of the pool to all of it, so one "
        "common subset across all four would judge everybody on whatever the "
        "sparsest source happened to publish. Within a pair the comparison is "
        "**per player**, which is what lets so few gameweeks separate anything: "
        "differencing removes the shared difficulty of the week. \"Level with\" "
        "means the 95% interval on that difference still contains zero."
    )
    st.caption(
        "⚠️ The interval is across *players*, not across *gameweeks*. A source "
        "that happened to read these particular weeks well looks exactly like a "
        "better source, which is what the \"led in N of M\" line is there to "
        "temper — and why the blend weights are still left alone."
    )

    with st.expander("Head to head"):
        for scope, label in ((projection_accuracy.SCOPE_START, "Who starts"),
                             (projection_accuracy.SCOPE_STARTERS, "Points if he starts")):
            h2h = projection_accuracy.head_to_head(scope)
            if h2h.empty:
                continue
            st.markdown(f"**{label}**")
            render_styled_table(
                pd.DataFrame({
                    "Matchup": [
                        "%s  vs  %s" % (_SOURCE_LABELS.get(r["a"], r["a"]),
                                        _SOURCE_LABELS.get(r["b"], r["b"]))
                        for _, r in h2h.iterrows()],
                    "Shared players": h2h["n"],
                    "Better": [_SOURCE_LABELS.get(w, w) if w else "—"
                               for w in h2h["winner"]],
                    "By": h2h["difference"].abs(),
                    "95% interval": ["%+.3f to %+.3f" % (r["ci_low"], r["ci_high"])
                                     for _, r in h2h.iterrows()],
                }).reset_index(drop=True),
                col_formats={"By": "{:.3f}"},
            )


def _join(items) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def _render_start_model(scored):
    """How good is the app at the other half of the question -- who plays?

    `Proj = Proj_Start x Start_Pct`, so this half is on every board in the app,
    and until now nothing scored it. It is also where the evidence is: a start
    probability is checked against a binary outcome for every player in the pool,
    where the points model is only scoreable on the couple of hundred who started.
    """
    start = scored[scored["scope"] == projection_accuracy.SCOPE_START]
    if start.empty:
        return

    st.markdown("#### Will he start?")
    st.caption(
        "The minutes model on its own, scored against whether each player "
        "actually started. **Brier** is the mean squared error of a probability — "
        "lower is better, and 0.25 is what you get by answering 50% to "
        "everything. **AUC** is ordering alone: whether the players it rated "
        "higher did start more often. The two fail differently — a source can "
        "order players perfectly and still state every probability far too high — "
        "which is why **bias** is here as well."
    )

    summary = projection_accuracy.summarise(scored)
    rows = summary[summary["scope"] == projection_accuracy.SCOPE_START]
    if not rows.empty:
        render_styled_table(
            pd.DataFrame({
                "Source": rows["source"].map(_SOURCE_LABELS).fillna(rows["source"]),
                "GWs": rows["gameweeks"],
                "Players scored": rows["n"],
                "Coverage": rows["coverage"] * 100,
                "Brier": rows["brier"],
                "Bias": rows["bias"],
                "AUC": rows["auc"],
            }).reset_index(drop=True),
            col_formats={"Coverage": "{:.0f}%", "Brier": "{:.3f}",
                         "Bias": "{:+.3f}", "AUC": "{:.3f}"},
            negative_color_cols=["Brier"],
            positive_color_cols=["AUC"],
        )
        st.caption(
            "FFP and FPL are scored only on the players they publish a number "
            "for, which is why their coverage is well under 100% — the blend "
            "has an opinion about everybody."
        )

    calib = projection_accuracy.start_calibration()
    if not calib.empty:
        st.markdown("**Calibration**")
        st.caption(
            "Whether a stated probability means what it says. This is the table "
            "a single score cannot show you: in GW3 the app rendered 208 players "
            "at exactly 100% — not because any source said so, but because a "
            "fallback stood in for one that had stopped publishing — and a "
            "quarter of them started. Every individual number looked fine."
        )
        render_styled_table(
            pd.DataFrame({
                "Rated to start": calib["bucket"].map(_band_label),
                "Players": calib["n"],
                "Predicted": calib["predicted"] * 100,
                "Actually started": calib["actual"] * 100,
                "Gap": calib["gap"] * 100,
            }).reset_index(drop=True),
            col_formats={"Predicted": "{:.0f}%", "Actually started": "{:.0f}%",
                         "Gap": "{:+.0f}"},
        )

    cohorts = projection_accuracy.start_cohort_rates()
    if not cohorts.empty:
        with st.expander("What Rotowire's silence is worth"):
            st.caption(
                "Rotowire lists about eleven players a club, so it *omitting* "
                "someone is a lineup call rather than a missing value. These "
                "eight cells are what the engine's two tunable constants act "
                "on: a floor under the players it lists, an implied probability "
                "for the ones it does not. The cells differ in size by an order "
                "of magnitude, so each is retuned on its own sample — not on a "
                "gameweek count."
            )
            render_styled_table(
                pd.DataFrame({
                    "Position": cohorts["position"],
                    "Rotowire": cohorts["cohort"],
                    "Players": cohorts["n"],
                    "Predicted": cohorts["predicted"] * 100,
                    "Started": cohorts["actual"] * 100,
                    "Constant": cohorts["constant"] * 100,
                    "Enough to fit": cohorts["enough_rows"].map({True: "yes", False: "no"}),
                }).reset_index(drop=True),
                col_formats={"Predicted": "{:.1f}%", "Started": "{:.1f}%",
                             "Constant": "{:.0f}%"},
            )
            st.caption(
                "Backfills are excluded here, unlike the tables above: a "
                "snapshot taken after kickoff may have seen the team sheet, and "
                "a start model scored against the answer is not measuring "
                "anything."
            )


def _band_label(interval) -> str:
    """'70-85%' for a calibration bucket."""
    try:
        return "%d–%d%%" % (max(0, round(interval.left * 100)), round(interval.right * 100))
    except Exception:
        return str(interval)


def _render_weight_fit(scored):
    """What weights would have been best, and whether that is worth acting on."""
    st.markdown("#### Blend weights")
    fit = projection_accuracy.fit_blend_weights()

    if fit.get("fitted") is None:
        st.info(
            fit.get("note")
            or "Not enough pre-deadline history to fit weights yet."
        )
        st.caption(
            "Current weights: "
            + " · ".join(f"{_SOURCE_LABELS.get(k, k)} {v:.0%}"
                         for k, v in (config.PROJECTION_SOURCE_WEIGHTS or {}).items()
                         if v > 0)
        )
        return

    fitted, current = fit["fitted"], fit["current"]
    gain = fit["current_mae"] - fit["mae"]
    st.caption(
        "Fitted on %d gameweek%s (%d player-gameweeks), pre-deadline captures "
        "only." % (fit["gameweeks"], "" if fit["gameweeks"] == 1 else "s", fit["n"])
    )
    rows = [{
        "Source": _SOURCE_LABELS.get(k, k),
        "Current": current.get(k, 0.0) * 100,
        "Best fit": fitted.get(k, 0.0) * 100,
    } for k in fitted]
    render_styled_table(
        pd.DataFrame(rows),
        col_formats={"Current": "{:.0f}%", "Best fit": "{:.0f}%"},
    )
    st.caption(
        "Current weights give MAE **%.3f**; the best fit gives **%.3f** "
        "(%.3f better)." % (fit["current_mae"], fit["mae"], gain)
    )

    if fit["gameweeks"] < projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE:
        st.warning(
            "This fit is on %d gameweek%s. Weights chosen from a sample this "
            "small would be fitting noise — the app's configured weights are "
            "deliberately left alone until there are at least %d."
            % (fit["gameweeks"], "" if fit["gameweeks"] == 1 else "s",
               projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE)
        )


def _gw_list(gws) -> str:
    """'GW1, GW2 and GW3', or 'none'."""
    if not gws:
        return "none"
    labels = [f"GW{g}" for g in gws]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def show_player_projections_page():
    """Main projections hub page, grouped by the question each tab answers.

    Eight flat tabs grew into a list nobody could scan, and the grouping that
    fixes it is by *question*, not by source. "How many points this gameweek"
    had three tabs answering it -- and Rotowire and FFP are not alternatives to
    the blend, they are the detail behind the per-source columns the Blended
    tab already shows, so they belong underneath it. The three odds tabs were
    three more answering "what is likely to happen", which is a different
    question from "how many points" and reads as one group.

    **Accuracy stays top level deliberately.** It is not a projection for this
    gameweek: it scores every source after the fact, including the ones Horizon
    uses, and it is the page's credibility content. Nesting it under GW
    Projections would scope it to one of the things it measures.

    Note that ``st.tabs`` renders every tab body on each rerun -- the switch is
    client-side CSS -- so the grouping buys legibility, not load time. What buys
    load time is ``_blended_pool()``, which stops GW Projections and Horizon
    building the same blend twice.
    """
    st.title("Projections Hub")
    st.caption(
        "**GW Projections** is the projection the rest of the app uses, with "
        "the sources that fed it alongside. **Horizon** is the same question "
        "over the next six gameweeks, **Odds** is the betting markets, and "
        "**Accuracy** is how wrong each source has actually been."
    )

    tab_proj, tab_horizon, tab_odds, tab_accuracy = st.tabs([
        "GW Projections",
        "Horizon",
        "Odds",
        "Accuracy",
    ])

    with tab_proj:
        # Blended leads. The individual sources stay one click away rather than
        # being hidden: the blend is the answer, but seeing what fed it is how
        # you judge whether to trust it on a particular player.
        sub_blended, sub_rotowire, sub_ffp = st.tabs([
            "Blended", "Rotowire", "FFP Data",
        ])
        with sub_blended:
            render_blended_projections()
        with sub_rotowire:
            render_rotowire_projections()
        with sub_ffp:
            render_ffp_data()

    with tab_horizon:
        render_horizon()

    with tab_odds:
        sub_goals, sub_cs, sub_match = st.tabs([
            "Goal/Assist", "Clean Sheet", "Match Odds",
        ])
        with sub_goals:
            render_goalscorer_odds()
        with sub_cs:
            render_clean_sheet_odds()
        with sub_match:
            render_match_odds()

    with tab_accuracy:
        render_accuracy()

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
