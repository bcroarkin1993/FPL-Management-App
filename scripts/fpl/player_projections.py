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
import re

import config
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scripts.common.text_helpers import format_last_updated, relative_age
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
from scripts.common.styled_tables import render_styled_table, value_rgb
from scripts.common import cards, charts
from scripts.common.text_helpers import to_display_name
from scripts.common.text_helpers import compact_html
from scripts.common.text_helpers import _strip_accents
from scripts.common import analytics
from scripts.common.analytics import blend_projections_onto, merge_season_projections
from scripts.common.player_pool import blended_pool as _blended_pool
from scripts.common.fpl_classic_api import get_classic_bootstrap_static
from scripts.common import projection_accuracy, projection_archive
from scripts.common.data_validation import check_ffp_horizon
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.projections_hub")


def _positional_rank(df: pd.DataFrame, value_col: str,
                     position_col: str = "Pos") -> pd.Series:
    """1-based rank within position, best first, over *every* row of ``df``.

    A projection is only interpretable against the other players who could fill
    that slot: 4.2 points is a fine goalkeeper and a poor forward, and the
    overall ``#`` column cannot say which. This is the same quantity
    ``analytics.positional_rank()`` reports on the squad pages, as a bare
    integer rather than a "#8 MID" label, because here the position is already
    its own column and the value has to sort.

    **Call this before filtering.** A rank re-derived from the filtered rows
    ranks a selection against itself -- filter to one club and its best
    midfielder reads "1" whatever he is worth league-wide -- which is the trap
    ``render_styled_table``'s colour ranges document, one column over.

    Ties share a rank (``method="min"``), matching ``positional_rank()``'s
    "number of players strictly better, plus one". A row with no value ranks
    nowhere and comes back NaN, which renders as an empty cell rather than as a
    confident last place.
    """
    if value_col not in df.columns or position_col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    vals = pd.to_numeric(df[value_col], errors="coerce")
    return vals.groupby(df[position_col].astype("object"),
                        dropna=False).rank(method="min", ascending=False)


def _name_filter(frame: pd.DataFrame, needle: str,
                 cols=("Player", "_search_name")) -> pd.Series:
    """Rows whose name matches ``needle``, accents and legal names included.

    Two things a bare ``str.contains`` on the rendered name gets wrong, both
    reported from the app:

    * **The rendered name may not be the one you would type.** FPL abbreviates
      some surnames in ``web_name``, so searching "Nunes" matched nothing while
      the Horizon grid showed "Matheus N." -- the display bug
      ``to_display_name`` now fixes, but the search should not depend on having
      fixed it. Matching the full legal name as well means a surname always
      works.
    * **Accents.** "Guimaraes" typed on an ASCII keyboard does not contain-match
      "Guimarães", and no user is going to type the ã.

    Both sides are accent-stripped and lowercased, so the needle and the
    haystack meet in the middle.
    """
    needle = _strip_accents(str(needle or "")).lower().strip()
    if not needle:
        return pd.Series(True, index=frame.index)
    hay = pd.Series("", index=frame.index)
    for col in cols:
        if col in frame.columns:
            hay = hay.str.cat(
                frame[col].astype(str).map(lambda v: _strip_accents(v).lower()),
                sep=" ", na_rep="")
    return hay.str.contains(re.escape(needle), regex=True, na=False)


def _overall_rank(df: pd.DataFrame, value_col: str,
                  ascending: bool = False) -> pd.Series:
    """1-based rank over *every* row of ``df``, best first.

    The board's ``#`` column, and it obeys the same rule as
    ``_positional_rank`` above: **compute it before filtering.** It used to be
    ``range(1, len(display) + 1)`` stamped on after the search and position
    filters had run, so searching one player reported him as **#1** whatever he
    was worth -- the overall-rank twin of the trap that function documents.

    Ties share a rank and a row with no value comes back NaN, for the same
    reasons.
    """
    if value_col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    vals = pd.to_numeric(df[value_col], errors="coerce")
    return vals.rank(method="min", ascending=ascending)


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


#: Per-source accents, chosen to read on the dark ground. Kept distinct
#: because three of these banners sit within two clicks of each other, and the
#: colour is the fastest way to tell which source you are looking at.
ACCENT_ROTOWIRE = "#38bdf8"
ACCENT_FFP = "#fbbf24"
ACCENT_ODDS_API = "#a78bfa"
ACCENT_CLEAN_SHEET = "#34d399"


def _render_source_banner(source: str, description: str, accent: str,
                          url: str = None, updated=None):
    """Render a data-source attribution banner, on the app's own card shell.

    This used to paint a *light* panel -- `#f0f9ff`, `#fef3c7`, `#ede9fe` -- in
    an app where every other surface is `#1a1a2e`. The per-source accent
    survives as an edge on the house card, which keeps the identity and loses
    the clash; `cards.card` is the same shell Classic Home, the My Week strip
    and the league cards are built from, so these line up with everything else
    by construction rather than by anyone measuring.

    `updated` adds the source's own publish time. Worth showing: a projection
    table written before the last team-news cycle is materially less reliable
    than one written after it, and nothing else on the page reveals which you
    are looking at.
    """
    name = ('<a href="%s" target="_blank" style="color:%s;font-weight:700;'
            'text-decoration:none;">%s</a>' % (url, accent, source)
            if url else source)
    body = cards.eyebrow("Data source") + cards.headline(name, size="1.0em")
    body += cards.note(description, colour=cards.MUTED)
    if updated is not None:
        body += cards.note("Updated %s" % format_last_updated(updated))
    st.markdown(cards.card(body, accent=accent, padding="11px 14px"),
                unsafe_allow_html=True)
    # `cards.card` carries no outer margin, and on every tab that uses this a
    # heading follows immediately.
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


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
        ACCENT_ROTOWIRE,
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
    # Filters, inline rather than behind an expander: narrowing is the
    # primary interaction on a board this long.
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
        result = result[_name_filter(result, player_filter, cols=('Player',))]

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
        ACCENT_FFP,
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

    # Ranked on whatever this table is actually about -- FFP's points prediction
    # normally, its goal odds when they have published no predictions -- and
    # ranked here, over the whole board, so the user filters below narrow the
    # rows without re-ranking them against each other.
    if sort_col and 'Position' in df.columns:
        df.insert(df.columns.get_loc('Position') + 1, 'Pos RK',
                  _positional_rank(df, sort_col, position_col='Position'))

    # Filters
    # Filters, inline rather than behind an expander: narrowing is the
    # primary interaction on a board this long.
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
        result = result[_name_filter(result, player_filter,
                                     cols=('Player', 'Name', 'Display_Name'))]

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
    col_fmts = {'Price': '{:.1f}', 'Pos RK': '{:.0f}'}
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
               + (f" · {_ffp_caption(feed)}." if _ffp_caption(feed) else ".")
               + (f" **Pos RK** is his rank on **{sort_col}** among every player "
                  "at his position, and does not change when you filter."
                  if sort_col and 'Pos RK' in display_df.columns else ""))


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
        ACCENT_FFP,
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
    _ident_cols = ["Display_Name", "Team", "Position"]
    if "Name" in ident.columns:
        ident = ident.rename(columns={"Name": "Player"})
    if "Player" in ident.columns:
        _ident_cols.append("Player")
    ident = ident.set_index("Player_ID")[_ident_cols]
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
            "Player": (b["Player"].reindex(points.index) if "Player" in b.columns
                       else pd.Series(index=points.index, dtype="object")
                       ).combine_first(
                ident["Player"].reindex(points.index) if "Player" in ident.columns
                else pd.Series(index=points.index, dtype="object")),
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
        # Searched but never rendered: the full legal name, so a surname FPL
        # abbreviates out of the display name is still findable. Dropped before
        # the table is handed to the renderer.
        "_search_name": ident.get("Player", pd.Series("", index=ident.index)).fillna(""),
    }, index=points.index)
    for g in shown:
        grid[gw_labels[g]] = pd.to_numeric(points[g], errors="coerce")
    label_cols = [gw_labels[g] for g in shown]
    grid["Total"] = grid[label_cols].sum(axis=1, min_count=1)

    # A player with nothing but zeros across the whole window is a declared
    # non-starter, not a row worth a line on a board about long-term value.
    grid = grid[grid[label_cols].fillna(0).abs().sum(axis=1) > 0]

    # Rank on the window total, within position, over the whole board -- so it
    # answers "where does this run put him among the players who could fill
    # that slot", and narrowing to one club does not re-rank him against his
    # own team-mates. It is a rank over exactly the weeks shown, so shortening
    # the window legitimately changes it.
    grid.insert(3, "Pos RK", _positional_rank(grid, "Total"))

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

    # Filters, inline rather than behind an expander: narrowing is the
    # primary interaction on a board this long.
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

    # Like Pos RK and the colour range above, the overall rank is fixed before
    # the filters run -- it is a position on the whole board, so searching one
    # player must not report him as "1". It does move with the sort control,
    # which is the point of that control rather than a narrowing of the rows.
    grid.insert(0, "#", _overall_rank(grid, sort_by, ascending=not descending))

    if search:
        grid = grid[_name_filter(grid, search)]
    if positions:
        grid = grid[grid["Pos"].isin(positions)]
    if teams:
        grid = grid[grid["Team"].isin(teams)]
    grid = grid.drop(columns=["_search_name"], errors="ignore")

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
        render_styled_table(
            display.head(300),
            col_formats={**{c: "{:.1f}" for c in label_cols},
                         "Total": "{:.1f}", "#": "{:.0f}", "Pos RK": "{:.0f}"},
            positive_color_cols=label_cols + ["Total"],
            color_range_overrides=({c: span for c in label_cols} if span else None),
            max_height=640,
        )
    else:
        rows = []
        colour = {c: [] for c in label_cols}
        for pid in ids:
            row = {"#": grid.at[pid, "#"],
                   "Player": grid.at[pid, "Player"], "Team": grid.at[pid, "Team"],
                   "Pos": grid.at[pid, "Pos"],
                   "Pos RK": grid.at[pid, "Pos RK"]}
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
        display = pd.DataFrame(rows).head(300)
        render_styled_table(
            display,
            col_formats={"Total": "{:.1f}", "Avg FDR": "{:.2f}",
                         "#": "{:.0f}", "Pos RK": "{:.0f}"},
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
        "**#** ranks the sorted column and **Pos RK** ranks **Total** against "
        "every player at the same position, both over these same weeks and "
        "over the whole board — so they move when you change the window or the "
        "sort, but not when you filter.",
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
        ACCENT_FFP,
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
    # Filters, inline rather than behind an expander: narrowing is the
    # primary interaction on a board this long.
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
        ACCENT_CLEAN_SHEET,
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
        ACCENT_ODDS_API,
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

    # The legend the deleted footer panel used to carry, now beside the thing
    # it describes.
    st.caption(
        "Each card is one fixture -- **home** left, **away** right -- with a "
        "proportional bar where each segment's width is that outcome's "
        "likelihood: green the home win, grey the draw, purple the away win. "
        "Kickoffs are Eastern Time."
    )

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

# _build_pool() and blended_pool() moved to scripts/common/player_pool.py when
# the Classic Rival Tracker needed the same frame. blended_pool is imported
# above under its old private name, so the cache key -- and the tests that
# patch pp._blended_pool -- are unchanged.

def _num(val):
    """One value as a finite float, or None.

    `pd.to_numeric` wants a Series and the tiles read single cells, so this is
    the scalar form. None rather than NaN because every caller here has to
    branch on "is there a number" anyway, and `x != x` at four callsites is how
    a NaN leaks into a rendered string.
    """
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# =============================================================================
# Source health -- one derivation, two presentations
# =============================================================================

def _source_rows(rotowire_df, feed, blended, current_gw):
    """One record per source: up or down, how many players, how fresh, what weight.

    Modelled on the Initial Squad Optimizer's Data Sources panel, which exists
    because every fetch in this app degrades to an empty frame on failure --
    without a panel like this a broken source renders a page identical to a
    working one, just with every number quietly leaning on whatever survived.

    **This used to be the renderer.** It is now the data behind two
    presentations -- the one-line strip above the board and the full table
    inside `Source detail` -- and that split is the point: a strip that did its
    own reading of the same feeds would be a second implementation able to
    disagree with the table a click below it, which is the twin-drift failure
    this codebase keeps paying for.
    """
    weights = getattr(config, "PROJECTION_SOURCE_WEIGHTS", {}) or {}

    def _row(name, short, ok, rows, note, updated, weight_key):
        if weight_key == "__start_only__":
            weight, label = None, "start signal only"
        else:
            w = float(weights.get(weight_key, 0.0))
            weight, label = w, ("%.0f%%" % (w * 100) if w > 0 else "fallback only")
        if not ok or not rows:
            status = "\U0001F534 Unavailable"
        elif note:
            status = "\U0001F7E1 %s" % note
        else:
            status = "\U0001F7E2 OK"
        return {"name": name, "short": short, "ok": bool(ok and rows),
                "rows": int(rows or 0), "note": note or "", "updated": updated,
                "weight": weight, "weight_label": label, "status": status}

    rows = []
    rw_ok = rotowire_df is not None and not rotowire_df.empty
    rows.append(_row(
        "Rotowire", "Rotowire", rw_ok, len(rotowire_df) if rw_ok else 0,
        "" if rw_ok else "no rankings table for this gameweek",
        get_rotowire_article_updated(config.ROTOWIRE_URL) if config.ROTOWIRE_URL else None,
        "rotowire",
    ))
    ffp_note = ""
    if feed.ok and feed.is_stale(current_gw):
        ffp_note = "published for GW%s - excluded" % feed.gameweek
    elif feed.ok and feed.provenance == "sheet":
        ffp_note = "read from their spreadsheet"
    elif feed.ok and feed.provenance == "archive":
        ffp_note = "from our archive of their last publication"
    rows.append(_row(
        "Fantasy Football Pundit", "FFP", feed.ok, len(feed.df) if feed.ok else 0,
        ffp_note, feed.updated, "ffp",
    ))
    _n_xp = int((blended["Proj_Src"] == "xP").sum()) if "Proj_Src" in blended.columns else 0
    rows.append(_row(
        "FPL expected points (ep_next)", "FPL xP", True, _n_xp,
        "" if _n_xp else "not needed - every player priced by another source",
        None, "fpl_ep",
    ))

    # One row per publication. Listing them separately is the point: a single
    # "projected XIs" row would hide which of them is stale, and the engine
    # combines them unanimously-to-bench, so *how many* have an opinion about a
    # club decides whether anyone there can be benched at all.
    _xis = analytics._lineup_xis()
    for _src, _label, _short in (("rotowire", "Rotowire projected XIs", "RW XIs"),
                                 ("sportsgambler", "Sportsgambler projected XIs", "SG XIs")):
        xi = _xis.get(_src)
        n_starters = len(getattr(xi, "player_ids", ()) or ()) if xi else 0
        n_clubs = len(getattr(xi, "clubs", ()) or ()) if xi else 0
        if not n_starters:
            xi_note = "unavailable - the other XI source stands alone"
        elif n_clubs < 20:
            # A club short of a complete XI can promote its starters but bench
            # nobody, so say so rather than showing a green tick for a signal
            # that is working in one direction only.
            xi_note = "%d/20 clubs complete - no benchings at the rest" % n_clubs
        else:
            xi_note = ""
        rows.append(_row(_label, _short, bool(n_starters), n_starters,
                         xi_note, None, "__start_only__"))
    return rows


def _render_source_detail(rows) -> None:
    """The full audit, inside `Source detail`.

    Unchanged in content from when it led the page; it has moved, not shrunk.
    The update-frequency note at the end is what the page footer's second,
    hand-maintained Data Sources table used to carry -- that table had drifted
    to listing five sources with no weights and neither projected-XI feed, so
    it is gone and the live panel says this instead.
    """
    render_styled_table(pd.DataFrame([
        {"Source": r["name"], "Status": r["status"], "Players": r["rows"],
         "Weight": r["weight_label"],
         "Updated": format_last_updated(r["updated"]) if r["updated"] else "-"}
        for r in rows
    ]))
    st.caption(
        "Weight is each source's share of the blend where it prices a player; "
        "weights are renormalised over whichever sources are actually available, "
        "so a missing source does not shrink the projection. FPL's own expected "
        "points fills only players nobody else priced. **The projected XIs carry "
        "no points and no weight** - they decide *who starts*, which a weekly "
        "rankings article is too slow to track, so a benched player is priced as "
        "a substitute even where the article still ranks him. They are combined "
        "**unanimous-to-bench**: one source naming a player starts him, and he is "
        "benched only where every source with an opinion leaves him out, so two "
        "XIs disagreeing can never lower anybody."
    )
    st.caption(
        "**How often each refreshes.** Rotowire publishes a rankings article "
        "once a week and updates its projected XIs as team news breaks; FFP "
        "republishes through the gameweek; Sportsgambler updates its XIs daily; "
        "FPL's expected points move with every price and status change. The "
        "**Updated** column above is the source's own stamp where it publishes "
        "one, so it is the figure to trust over any of this."
    )


def _source_health_line(rows) -> str:
    """The audit as one line: is anything down, and how fresh is what is up.

    Derived from `_source_rows` rather than from the feeds, so it cannot
    contradict the table behind `Source detail`.

    What it shows is deliberately not everything: a source carrying real weight
    is always listed, and anything degraded or down is always listed, but a
    healthy zero-weight fallback is not -- it has nothing to say at a glance
    and would push the two numbers that matter off the end of the line.
    """
    if not rows:
        return ""

    down = [r for r in rows if not r["ok"]]
    degraded = [r for r in rows if r["ok"] and r["note"]]
    live = len(rows) - len(down)
    if down:
        colour, verdict = cards.BAD, "%d of %d sources down" % (len(down), len(rows))
    elif degraded:
        colour, verdict = cards.WARN, "All %d live, %d degraded" % (live, len(degraded))
    else:
        colour, verdict = cards.ACCENT, "All %d sources live" % live

    segs = ['<span style="color:%s;font-weight:700;white-space:nowrap;">&#9679; %s</span>'
            % (colour, verdict)]
    for r in rows:
        carries_weight = r["weight"] is not None and r["weight"] > 0
        if not (carries_weight or not r["ok"] or r["note"]):
            continue
        bits = ['<b style="color:%s;">%s</b>' % (cards.TEXT, r["short"])]
        if carries_weight:
            bits.append("%.0f%%" % (r["weight"] * 100))
        if not r["ok"]:
            bits.append('<span style="color:%s;">unavailable</span>' % cards.BAD)
        elif r["note"]:
            # The headline half only. "18/20 clubs complete - no benchings at
            # the rest" is the right text in the detail table and too long for
            # a one-line strip carrying five of them.
            bits.append('<span style="color:%s;">%s</span>'
                        % (cards.WARN, r["note"].split(" - ")[0]))
        elif r["updated"] is not None:
            bits.append(relative_age(r["updated"]) or "")
        segs.append('<span style="white-space:nowrap;">%s</span>'
                    % " ".join(b for b in bits if b))

    rule = '<span style="color:%s;">&#9474;</span>' % cards.RULE
    body = ('<div style="display:flex;flex-wrap:wrap;align-items:center;'
            'gap:4px 10px;font-size:0.84em;color:%s;line-height:1.5;">%s</div>'
            % (cards.MUTED, rule.join(segs)))
    return cards.card(body, padding="9px 13px")


# =============================================================================
# The insight rail -- where to look, before you have filtered anything
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def _trust_figures() -> dict:
    """The start model's Brier against the no-skill baseline, as primitives.

    Cached, and returning plain values rather than a frame, for two separate
    reasons. The archive read measured **0.61s** of what used to be a 0.92s
    rerun back when the Accuracy section was eager -- precisely the cost the
    lazy radio removed, and putting a tile on the board must not quietly bring
    it back on every search keystroke. And nothing unhashable enters a cache
    key here, which is the rule `my_week._assemble` already follows.

    **The gate is `confident_gameweeks`, not a second threshold.** The Accuracy
    section refuses to act on fewer than `MIN_GAMEWEEKS_FOR_CONFIDENCE`
    pre-deadline gameweeks, and a tile printing a confident percentage off a
    history that section disclaims would have the page contradict itself two
    clicks apart.
    """
    try:
        # `common_subset=False` deliberately, and this is the whole subtlety of
        # the tile. The common subset exists so that *sources* can be compared
        # on the players all of them priced -- which is right for the Accuracy
        # section's per-source table and wrong here, because this is an
        # absolute score against `no_skill_reference`, and that baseline is
        # computed over the full population. Measured live: the blend scores
        # Brier 0.085 over 5,308 player-gameweeks unrestricted, and 0.136 over
        # the 1,009 the subset leaves -- which are mostly the doubtful players
        # FPL bothered to rate. Scoring the model on the hard subset and the
        # baseline on everybody turns 62% skill into 39%, which is the
        # two-populations-one-comparison error this app has paid for elsewhere.
        # So the figure here legitimately differs from that table, and the note
        # on the tile says which basis it is on.
        scored = projection_accuracy.score_archive(common_subset=False)
    except Exception:
        _logger.warning("Trust tile: the projection archive could not be scored",
                        exc_info=True)
        return {"ok": False, "reason": "the projection archive could not be read"}
    if scored is None or scored.empty:
        return {"ok": False, "reason": "no gameweek has been scored yet"}

    gws = int(projection_accuracy.confident_gameweeks(scored))
    if gws < projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE:
        return {"ok": False, "gameweeks": gws,
                "reason": "only %d of %d pre-deadline gameweeks scored so far"
                          % (gws, projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE)}

    summary = projection_accuracy.summarise(scored)
    if summary is None or summary.empty:
        return {"ok": False, "gameweeks": gws, "reason": "nothing to summarise"}

    def _blend(scope, metric):
        rows = summary[(summary["scope"] == scope) & (summary["source"] == "blend")]
        return _num(rows[metric].iloc[0]) if not rows.empty and metric in rows else None

    brier = _blend(projection_accuracy.SCOPE_START, "brier")
    ref = projection_accuracy.no_skill_reference(projection_accuracy.SCOPE_START)
    skill = (projection_accuracy.skill_against_no_skill(brier, ref)
             if brier is not None else None)
    if skill is None:
        return {"ok": False, "gameweeks": gws,
                "reason": "the start model has no scored rows yet"}
    rows_scored = summary[(summary["scope"] == projection_accuracy.SCOPE_START)
                          & (summary["source"] == "blend")]["n"]
    return {"ok": True, "gameweeks": gws, "skill": float(skill), "brier": brier,
            "baseline": _num(ref.get("value")),
            "n": int(rows_scored.iloc[0]) if not rows_scored.empty else None}


def _tile_unknown(label: str, reason: str) -> str:
    """A tile that says it could not answer, rather than defaulting to calm.

    This is the `week_check.STATUS_UNKNOWN` discipline one surface over. A tile
    makes a claim at a glance, and "nothing to flag" and "we could not check"
    are opposite claims that render identically unless one of them says so.
    """
    return (cards.eyebrow(label)
            + cards.headline('<span style="color:%s;">Not available</span>'
                             % cards.MUTED, size="1.05em")
            + cards.note(reason))


def _tile_top(board) -> str:
    """The best projected player, which is what the page is for and used to
    take a scroll past the source audit to reach."""
    if board is None or board.empty:
        return _tile_unknown("Top of the board",
                             "no player is projected for this gameweek")
    row = board.iloc[0]
    proj = _num(row.get("Proj"))
    if proj is None:
        return _tile_unknown("Top of the board",
                             "the leading row carries no projection")
    sub = " \u00b7 ".join(str(v) for v in (row.get("Team"), row.get("Pos")) if v)
    start = _num(row.get("Start %"))
    if start is not None:
        sub += " \u00b7 %.0f%% to start" % start
    return (cards.eyebrow("Top of the board")
            + cards.headline('%s <span style="color:%s;">%.2f</span>'
                             % (row.get("Player", ""), cards.ACCENT, proj))
            + cards.note(sub))


def _tile_disagreement(board) -> str:
    """The player the blend is working hardest on.

    `Spread` is already a column, but it is one of twelve and the board is
    sorted on something else, so the row where the two sources are furthest
    apart never surfaced. It is worth surfacing precisely because a wide spread
    is not a verdict on the player -- it is a flag that his number is a
    compromise between two opinions, and worth a second look before you act on
    it.
    """
    label = "Sources disagree most"
    if board is None or board.empty or "Spread" not in board.columns:
        return _tile_unknown(label, "no spread was computed for this gameweek")
    spread = pd.to_numeric(board["Spread"], errors="coerce")
    if not spread.notna().any():
        return _tile_unknown(
            label, "only one source priced this gameweek, so there is nothing "
                   "to compare")
    row = board.loc[spread.idxmax()]
    gap = _num(row.get("Spread"))
    if gap is None:
        return _tile_unknown(label, "no spread was computed for this gameweek")
    rw, ffp = _num(row.get("Rotowire")), _num(row.get("FFP"))
    if rw is not None and ffp is not None:
        hi, lo = ("FFP", "Rotowire") if ffp > rw else ("Rotowire", "FFP")
        detail = "%s %.2f against %s %.2f" % (hi, max(rw, ffp), lo, min(rw, ffp))
    else:
        detail = "one source did not price him"
    wide = int((spread >= 1.5).sum())
    return (cards.eyebrow(label)
            + cards.headline('%s <span style="color:%s;">&plusmn;%.2f</span>'
                             % (row.get("Player", ""), cards.WARN, gap))
            + cards.note("%s \u00b7 %d player%s over 1.5"
                         % (detail, wide, "" if wide == 1 else "s")))


def _tile_trust() -> str:
    """How good the model that produced this board has actually been.

    The Accuracy section scores every source after the fact and is four clicks
    from here, so the board carried no indication of whether to believe it. One
    figure, on the half of the question with the evidence: a start probability
    is checked against a binary outcome for every player in the pool, where the
    points model is only scoreable on the couple of hundred who started.
    """
    figs = _trust_figures()
    if not figs.get("ok"):
        return _tile_unknown("Model trust",
                             figs.get("reason", "not yet measurable"))
    pct = figs["skill"] * 100
    colour = cards.ACCENT if pct >= 0 else cards.BAD
    # Deliberately one claim. A points MAE beside this would invite comparison
    # with the Accuracy section's per-source table, which is on a different
    # population by design -- see `_trust_figures`.
    note = ("Will-he-start model scores Brier %.3f against %.3f for one number "
            "for everybody" % (figs["brier"], figs["baseline"]))
    if figs.get("n"):
        note += " \u00b7 %s player-gameweeks" % format(figs["n"], ",")
    note += " over %d gameweeks" % figs["gameweeks"]
    return (cards.eyebrow("Model trust")
            + cards.headline('<span style="color:%s;">%.0f%%</span>'
                             '<span style="color:%s;font-size:0.68em;font-weight:600;">'
                             ' of the way to perfect</span>' % (colour, pct, cards.MUTED))
            + cards.note(note))


def _render_insight_tiles(board) -> None:
    """Three tiles above the board, as one grid row so they line up by
    construction rather than by anyone measuring a pixel height."""
    bodies = [_tile_top(board), _tile_disagreement(board), _tile_trust()]
    st.markdown(cards.band(*cards.card_row(bodies), columns="1fr 1fr 1fr"),
                unsafe_allow_html=True)


# =============================================================================
# The sparkline column, and the two charts under the board
# =============================================================================

def _sparkline_svg(values, lo, hi, width=64, height=20) -> str:
    """One player's horizon as an inline SVG, sized for a table cell.

    `render_styled_table` writes its cells as raw HTML, so this needs no change
    to the renderer. Six numbers in a row do not read as a shape; a 64px line
    does, and it is the only thing on this board that says *when* a player's
    points arrive rather than how many there are.

    **`lo`/`hi` are the board-wide range, passed in rather than derived per
    row.** A sparkline normalised to its own player makes every trajectory fill
    its cell, so a flat 0.3 and a flat 6.0 would draw identically -- the same
    trap as a colour range re-derived from filtered rows, which the Horizon
    grid's single fixed range already exists to avoid.
    """
    pts = [(i, v) for i, v in enumerate(values) if v == v]
    if len(pts) < 2:
        return ""
    span = (hi - lo) or 1.0
    steps = max(len(values) - 1, 1)
    pad = 2.5

    def _x(i):
        return pad + (width - 2 * pad) * (float(i) / steps)

    def _y(v):
        frac = min(1.0, max(0.0, (v - lo) / span))
        return (height - pad) - (height - 2 * pad) * frac

    poly = " ".join("%.1f,%.1f" % (_x(i), _y(v)) for i, v in pts)
    mean = sum(v for _, v in pts) / len(pts)
    colour = value_rgb(mean, lo, hi) or cards.MUTED
    return ('<svg viewBox="0 0 %d %d" width="%d" height="%d" '
            'style="display:block;margin:0 auto;" aria-hidden="true">'
            '<polyline points="%s" fill="none" stroke="%s" stroke-width="1.6" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
            '<circle cx="%.1f" cy="%.1f" r="2.2" fill="%s"/></svg>'
            % (width, height, width, height, poly, colour,
               _x(pts[-1][0]), _y(pts[-1][1]), colour))


def _sparkline_column(blended, current_gw):
    """`{element_id: svg}` for the forecast window, or `{}`.

    Built from `_horizon_frames`, the same assembly the Horizon section uses,
    for one reason that matters: that function replaces the current gameweek's
    column with the engine's own `Proj`. So the first point of every line here
    agrees with the `Proj` column beside it by construction. Re-deriving the
    series from FFP alone would put a forecast and the blend on one row and let
    them disagree, which is this page's recurring failure in miniature.

    Degrades to `{}` on any failure -- a missing sparkline costs a column, and
    nothing else on the board depends on it.
    """
    try:
        horizon = get_ffp_horizon()
    except Exception:
        _logger.warning("Sparklines: the FFP horizon could not be read", exc_info=True)
        return {}
    if horizon is None or not getattr(horizon, "ok", False):
        return {}
    gws = horizon.gameweeks()
    if not gws:
        return {}
    try:
        points = _horizon_frames(horizon, blended, current_gw)[0]
    except Exception:
        _logger.warning("Sparklines: the horizon grid could not be built", exc_info=True)
        return {}
    if points is None or points.empty:
        return {}
    cols = [g for g in gws if g in points.columns]
    if len(cols) < 2:
        return {}

    series = points[cols].apply(pd.to_numeric, errors="coerce")
    finite = series.to_numpy(dtype="float64").ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    lo, hi = float(finite.min()), float(finite.max())

    out = {}
    for pid, row in series.iterrows():
        svg = _sparkline_svg(row.tolist(), lo, hi)
        if svg:
            out[int(pid)] = svg
    return out


def _agreement_chart(board, shown) -> None:
    """Rotowire against FFP, with the agreement diagonal drawn.

    `Spread` gives the *size* of a disagreement and not its direction, so the
    board can tell you two sources are 1.5 points apart and never whose number
    you are taking. Which side of the diagonal a player falls is the whole
    content of this chart.

    Drawn from the **whole board** with the current filter ringed, never from
    the filtered rows: a scatter of one club against itself has no diagonal
    worth reading, and grading a selection against itself is the trap this
    page's fixed colour ranges and pre-filter ranks already avoid twice over.
    """
    if board is None or board.empty:
        return
    x = pd.to_numeric(board.get("Rotowire"), errors="coerce")
    y = pd.to_numeric(board.get("FFP"), errors="coerce")
    both = x.notna() & y.notna()
    if int(both.sum()) < 10:
        st.caption("Too few players are priced by both sources to compare them "
                   "this gameweek.")
        return

    bx, by = x[both], y[both]
    names = board.loc[both, "Player"].astype(str)
    far = (bx - by).abs() >= 1.5
    top = float(max(bx.max(), by.max())) * 1.03
    hover = "%{text}<br>Rotowire %{x:.2f} \u00b7 FFP %{y:.2f}<extra></extra>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, top], y=[0, top], mode="lines", name="They agree",
        line=dict(color=cards.ACCENT_ALT, width=1.4, dash="dash"),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=bx[~far], y=by[~far], mode="markers", name="Within 1.5",
        marker=dict(size=6, color="#5d5a7a"), text=names[~far],
        hovertemplate=hover))
    if int(far.sum()):
        fig.add_trace(go.Scatter(
            x=bx[far], y=by[far], mode="markers", name="Apart by 1.5+",
            marker=dict(size=9, color=cards.WARN), text=names[far],
            hovertemplate=hover))

    if shown is not None and 0 < len(shown) < len(board):
        sx = pd.to_numeric(shown.get("Rotowire"), errors="coerce")
        sy = pd.to_numeric(shown.get("FFP"), errors="coerce")
        m = sx.notna() & sy.notna()
        if 0 < int(m.sum()) <= 60:
            fig.add_trace(go.Scatter(
                x=sx[m], y=sy[m], mode="markers", name="Your filter",
                marker=dict(size=13, color="rgba(0,0,0,0)",
                            line=dict(width=2, color=cards.ACCENT)),
                text=shown.loc[m, "Player"].astype(str),
                hovertemplate="%{text}<extra></extra>"))

    fig.update_layout(**charts.dark_layout(
        height=310, margin=dict(l=10, r=10, t=8, b=10),
        legend=dict(orientation="h", y=1.14, x=0,
                    font=dict(color="#ffffff", size=11)),
        xaxis=dict(title="Rotowire - points if he starts", rangemode="tozero"),
        yaxis=dict(title="FFP - points if he starts", rangemode="tozero"),
    ))
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _position_strip_chart(board, shown) -> None:
    """The shape of each positional field, which a rank cannot show.

    `Proj` of 4.2 is a leading goalkeeper and a replaceable forward. `Pos RK`
    fixes the *ranking* but says nothing about how far clear the top of a
    position is or how flat its middle runs -- and the four pools are not
    comparable populations: live they hold roughly 73 / 217 / 298 / 79 players.
    """
    if board is None or board.empty or "Pos" not in board.columns:
        return

    fig = go.Figure()
    drawn = 0
    # Reversed, because plotly stacks the first horizontal trace lowest and
    # goalkeeper-first is the order every other table on this page uses.
    for pos in ("F", "M", "D", "G"):
        vals = pd.to_numeric(board.loc[board["Pos"] == pos, "Proj"],
                             errors="coerce").dropna()
        if vals.empty:
            continue
        drawn += 1
        fig.add_trace(go.Box(
            x=vals, name=pos, orientation="h", boxpoints="all", jitter=0.55,
            pointpos=0, marker=dict(size=4, color="#6f6b90", opacity=0.5),
            line=dict(color=cards.ACCENT_ALT, width=1.3),
            fillcolor="rgba(78,204,163,0.08)", showlegend=False,
            hovertemplate=pos + " \u00b7 %{x:.2f}<extra></extra>"))
    if not drawn:
        return

    if shown is not None and 0 < len(shown) < len(board) and len(shown) <= 25:
        sp = pd.to_numeric(shown.get("Proj"), errors="coerce")
        m = sp.notna()
        if int(m.sum()):
            fig.add_trace(go.Scatter(
                x=sp[m], y=shown.loc[m, "Pos"].astype(str), mode="markers",
                name="Your filter", showlegend=False,
                marker=dict(size=12, symbol="diamond", color=cards.ACCENT,
                            line=dict(width=1, color="#06220f")),
                text=shown.loc[m, "Player"].astype(str),
                hovertemplate="%{text}<br>%{x:.2f}<extra></extra>"))

    fig.update_layout(**charts.dark_layout(
        height=310, margin=dict(l=10, r=10, t=8, b=10), showlegend=False,
        xaxis=dict(title="Proj - expected points this gameweek",
                   rangemode="tozero"),
        yaxis=dict(title=""),
    ))
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _render_board_charts(board, shown) -> None:
    """The two charts, under the board they describe."""
    st.markdown("##### What the table cannot say")
    left, right = st.columns(2)
    with left:
        _agreement_chart(board, shown)
    with right:
        _position_strip_chart(board, shown)
    st.caption(
        "Both are drawn from the **whole board** with your current filter "
        "ringed, not from the rows left after filtering - a scatter of one club "
        "against itself has no diagonal worth reading, and a field re-derived "
        "from a selection grades that selection against itself."
    )


def _blend_filter_bar(board):
    """The board's filters, as a control row rather than a collapsed expander.

    Narrowing a 670-row board is the primary interaction on this tab and it
    used to cost a click to reach. **Team** is new here, and matches the filter
    Horizon has had all along.

    **Returns the values; it must not apply them.** Both ranks are taken over
    the whole board before this runs, which is what stops a search reporting
    its single hit as "#1" -- the trap `_positional_rank` and `_overall_rank`
    each document at length.
    """
    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.4, 2.2])
    search = c1.text_input("Search player", key="blend_search",
                           placeholder="e.g. Mbeumo")
    positions = c2.multiselect("Position", ["G", "D", "M", "F"], default=[],
                               key="blend_pos")
    teams_available = sorted(
        t for t in board.get("Team", pd.Series(dtype=object)).dropna().unique() if t)
    teams = c3.multiselect("Team", teams_available, default=[], key="blend_team",
                           help="Leave empty for every club.")
    only_disagree = c4.checkbox(
        "Only where sources disagree by 1.5+", key="blend_disagree",
        help="Where Rotowire and FFP are far apart the blend is doing real "
             "work, and the pick deserves a second look.")
    return search, positions, teams, only_disagree


def render_blended_projections():
    """The blend, with the sources that produced it on the same row.

    This is the number every other page in the app now renders. Until this tab
    existed the Projections Hub showed Rotowire and FFP in separate tabs with no
    join between them, so the one figure the app actually scores on appeared
    nowhere, and a user could not see where two sources disagreed.

    **The order here is the point.** The source audit used to render first -- a
    five-row table and a hundred words about weight renormalisation, above the
    board. It is a *check*, not a finding: it exists because every fetch in this
    app degrades to an empty frame and a broken source renders a page identical
    to a working one. So it is now one line, with the detail a click away, and
    the three tiles that answer "where should I look" take the top of the page
    instead.
    """
    current_gw = config.CURRENT_GAMEWEEK
    rotowire_url = config.ROTOWIRE_URL

    rotowire_df = pd.DataFrame()
    if rotowire_url:
        rotowire_df = get_rotowire_player_projections(rotowire_url)
    feed = get_ffp_feed()

    if (rotowire_df is None or rotowire_df.empty) and not feed.ok:
        st.warning(
            "No projections available for GW%d. Rotowire has not published a "
            "rankings table and Fantasy Football Pundit could not be read."
            % current_gw
        )
        return

    blended = _blended_pool(current_gw, rotowire_url)
    if blended is None or blended.empty:
        st.error("Could not load the FPL player pool, so projections cannot be assembled.")
        return

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
        # shows `xP` where it filled a player nobody else priced, the source
        # detail counts them, and the Accuracy section scores it.
        "Spread": pd.to_numeric(blended.get("Proj_Spread"), errors="coerce"),
        "Src": blended["Proj_Src"],
        # Searched and joined on, never rendered -- both dropped before render.
        "_search_name": blended["Player"],
        "_pid": pd.to_numeric(blended.get("Player_ID"), errors="coerce"),
    })
    display = display[display["Proj"].notna()]

    # Both ranks are taken here, over the whole board, so filtering to a club
    # or a position narrows the rows without re-ranking them against each other.
    display.insert(3, "Pos RK", _positional_rank(display, "Proj"))
    display.insert(0, "#", _overall_rank(display, "Proj"))
    board = display.sort_values("Proj", ascending=False).reset_index(drop=True)

    _render_insight_tiles(board)

    rows = _source_rows(rotowire_df, feed, blended, current_gw)
    health = _source_health_line(rows)
    if health:
        st.markdown(health, unsafe_allow_html=True)
    with st.expander("Source detail"):
        _render_source_detail(rows)

    st.markdown("#### GW%d Blended Projections" % current_gw)
    st.caption(
        "**Proj** is expected points - what he is worth allowing for the chance "
        "he does not start. **If Starts** is the same projection assuming he "
        "does. **Spread** is the gap between the sources: a large one means they "
        "disagree about this player, not that he is a bad pick."
    )

    search, positions, teams, only_disagree = _blend_filter_bar(board)

    shown = board
    if search:
        shown = shown[_name_filter(shown, search)]
    if positions:
        shown = shown[shown["Pos"].isin(positions)]
    if teams:
        shown = shown[shown["Team"].isin(teams)]
    if only_disagree:
        shown = shown[shown["Spread"].fillna(0) >= 1.5]

    table = shown.copy()
    sparks = _sparkline_column(blended, current_gw)
    if sparks:
        pid = pd.to_numeric(table["_pid"], errors="coerce")
        table["Next 6"] = [sparks.get(int(v), "") if v == v else "" for v in pid]
    table = table.drop(columns=["_search_name", "_pid"], errors="ignore")

    if table.empty:
        st.info("No players match those filters.")
    else:
        render_styled_table(
            table.head(300),
            col_formats={
                "#": "{:.0f}", "Pos RK": "{:.0f}",
                "Proj": "{:.2f}", "If Starts": "{:.2f}", "Start %": "{:.0f}%",
                "Rotowire": "{:.2f}", "FFP": "{:.2f}",
                "Spread": "{:.2f}",
            },
            positive_color_cols=["Proj", "If Starts"],
            negative_color_cols=["Spread"],
            text_align={"Next 6": "center"},
            max_height=600,
        )
        caption = (
            "Showing %d of %d projected players. "
            "**#** is his rank on **Proj** across the whole board and **Pos RK** "
            "his rank among every player at his position, both league-wide - "
            "neither changes when you filter. "
            "The per-source columns are on the *if he starts* basis, so they are "
            "directly comparable with each other and with **If Starts**."
            % (min(len(table), 300), len(table))
        )
        if sparks:
            caption += (" **Next 6** is his forecast over the next six "
                        "gameweeks, on one scale for the whole board, so two "
                        "lines are comparable. Its first point is the **Proj** "
                        "on the same row.")
        st.caption(caption)

    _render_board_charts(board, shown)

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
    # The same count `confidence_note` reports, so the headline and the warning
    # cannot disagree -- and pre-deadline only, because a backfill must not buy
    # the page confidence it is refused for fitting weights.
    if (projection_accuracy.confident_gameweeks(scored)
            < projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE):
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
    _render_glossary()

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

    _render_horizon_length(scored)
    _render_weight_fit(scored)


def _render_horizon_length(scored):
    """How far ahead is worth projecting -- the question behind ROS's 40%.

    A waiver claim or a transfer is a lasting decision, so a longer forecast
    window sounds like it should be worth more. FFP publishes six gameweeks and
    the scoring model percentiles three of them, and nothing had ever measured
    whether the extra weeks help. They do not, and both reasons are on the page
    rather than only the conclusion -- the answer is counter-intuitive enough
    that it needs its evidence next to it.
    """
    st.markdown("#### How far ahead is worth projecting?")

    lengths = projection_accuracy.score_horizon_lengths()
    if lengths.empty:
        st.info(
            "No archived FFP gameweek has a complete run of actual points after "
            "it yet, so horizon length cannot be scored. This needs an FFP table "
            "carrying player ids — the GW3 archive is the spreadsheet recovery, "
            "which publishes names only."
        )
        return

    verdict = projection_accuracy.horizon_verdict()
    best = verdict.get("best_weeks", projection_accuracy.ROS_HORIZON_WEEKS)
    st.markdown(
        compact_html(
            """
            <div style="background:#16213e;border:1px solid #2a2f52;
                        border-radius:10px;padding:14px 16px;color:#e0e0e0;">
              <div style="font-size:0.78rem;letter-spacing:.06em;
                          text-transform:uppercase;color:#8fa3c8;">Best window</div>
              <div style="font-size:1.25rem;font-weight:700;margin:.25rem 0;
                          color:#7ee787;">%d gameweek%s</div>
              <div style="font-size:0.85rem;color:#c8d2e4;">%s</div>
            </div>
            """ % (best, "" if best == 1 else "s", verdict.get("note", ""))
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        "**ROS weights a 3-gameweek total at 40%**, so this asks whether that "
        "window is the right length. Each anchor is one FFP publication, scored "
        "against the longest complete run of actual points available after it — "
        "so a longer window is judged over a longer outcome rather than being "
        "marked against a single week. Rank correlation, because the scoring "
        "model percentiles this term: the ordering is what matters."
    )

    pivot = lengths.pivot_table(index="horizon_weeks", columns="anchor_gw",
                               values="rho")
    table = pd.DataFrame({
        "Window": ["%d GW%s" % (w, " ← ROS" if w == projection_accuracy.ROS_HORIZON_WEEKS
                                else "")
                   for w in pivot.index],
    })
    for anchor in pivot.columns:
        weeks_scored = int(lengths.loc[lengths["anchor_gw"] == anchor,
                                       "weeks_scored"].iloc[0])
        table["GW%d (%dw)" % (anchor, weeks_scored)] = pivot[anchor].values
    table["Mean"] = pivot.mean(axis=1).values
    render_styled_table(
        table,
        col_formats={c: "{:.3f}" for c in table.columns if c != "Window"},
        positive_color_cols=[c for c in table.columns if c != "Window"],
    )
    st.caption(
        "Column headings are the anchor gameweek and how many weeks of actual "
        "points it was scored over. n=%d players per cell."
        % int(lengths["n"].max())
    )

    decay = projection_accuracy.score_horizon_decay()
    if not decay.empty and decay["lead_weeks"].nunique() > 1:
        st.markdown("##### Why: a forecast decays fast")
        d = decay.rename(columns={
            "target_gw": "Gameweek", "lead_weeks": "Weeks early",
            "rho": "Rank corr", "n": "N"})[
            ["Gameweek", "Weeks early", "Rank corr", "N"]]
        render_styled_table(
            d.reset_index(drop=True),
            col_formats={"Rank corr": "{:.3f}"},
            positive_color_cols=["Rank corr"],
            negative_color_cols=["Weeks early"],
        )
        st.caption(
            "One FFP publication against itself, so the model and its vintage "
            "are held constant and only the distance to the target week varies. "
            "Each further week of a six-week window is worth less than the one "
            "before it, which is why lengthening the total does not pay."
        )

    overlap = projection_accuracy.score_horizon_overlap()
    if not overlap.empty:
        st.markdown("##### Why: a long window is mostly player quality")
        agg = (overlap.groupby("horizon_weeks")[["rho_vs_season", "rho_vs_ros"]]
               .mean().reset_index())
        agg.columns = ["Window (GW)", "vs season points", "vs the 3GW window"]
        render_styled_table(
            agg,
            col_formats={"vs season points": "{:.3f}",
                         "vs the 3GW window": "{:.3f}"},
            positive_color_cols=["vs season points", "vs the 3GW window"],
        )
        st.caption(
            "FFP publishes **one** start probability per player for all six "
            "forecast weeks, so the only thing varying across a horizon is the "
            "fixture — and over six weeks fixtures average out. The longer the "
            "window, the more it just restates who is a good player, which ROS "
            "already weights at 30–45% as season quality and measures from "
            "*actual* points rather than a forecast. The right-hand column is "
            "the other half of it: a window that correlates ~0.99 with the one "
            "in use would barely change a single ranking."
        )

    anchors = int(verdict.get("anchors", 0))
    if anchors < 2 or lengths["weeks_scored"].max() < 4:
        st.warning(
            "Thin evidence: %d anchor%s, and the longest outcome window is %d "
            "gameweek%s. A six-week horizon cannot really be judged against "
            "fewer than six weeks of results, so read this as *no reason to "
            "look further ahead* rather than as proof that %d is optimal."
            % (anchors, "" if anchors == 1 else "s",
               int(lengths["weeks_scored"].max()),
               "" if lengths["weeks_scored"].max() == 1 else "s",
               projection_accuracy.ROS_HORIZON_WEEKS)
        )


def _render_glossary():
    """What every number on this tab means, in one collapsed place.

    The tab reported Brier scores, AUC, MAE, bias and rank correlation to three
    decimal places and defined none of them where a reader would look. Each
    definition here comes with the direction that is good and -- the part that
    actually makes a number readable -- what it should be compared against,
    computed from the archive rather than asserted.
    """
    refs = {
        "Who starts": projection_accuracy.no_skill_reference(
            projection_accuracy.SCOPE_START),
        "Points if he starts": projection_accuracy.no_skill_reference(
            projection_accuracy.SCOPE_STARTERS),
    }
    with st.expander("What do these numbers mean?"):
        rows = []
        for key, (name, direction, blurb) in projection_accuracy.METRIC_GLOSSARY.items():
            rows.append({
                "Metric": name,
                "Good is": {"lower": "lower", "higher": "higher",
                            "toward zero": "near 0"}.get(direction, direction),
                "What it measures": blurb,
            })
        render_styled_table(pd.DataFrame(rows))
        st.caption(
            "**A score needs something to be compared against**, which is what "
            "makes the headline cards above readable. The baseline is the "
            "dumbest forecast that still respects the data — one number for "
            "everybody:"
        )
        for label, ref in refs.items():
            if not ref:
                continue
            st.caption(
                "· **%s** — %s (%s) scores %s **%.3f** over %s player-gameweeks."
                % (label, ref["label"], ref["detail"],
                   projection_accuracy.METRIC_GLOSSARY[ref["metric"]][0],
                   ref["value"], f"{ref['n']:,}")
            )
        st.caption(
            "**Reading the colours.** Every table in this app shades its "
            "numbers on one red-to-green scale -- green is a good value, "
            "yellow middling, red poor -- and the scale is fixed to the whole "
            "population being shown, never re-derived from the rows left after "
            "a filter. So a number keeps its colour when you narrow the board."
        )
        st.caption(
            "So on **who starts** the app is well clear of guessing. On "
            "**points if he starts** no source beats that constant on average "
            "miss — FPL returns among starters have a standard deviation of "
            "about 3.2 points and most of that is irreducible, so MAE is a weak "
            "way to choose between sources. **Rank correlation** is the one "
            "that separates them, and it is why the fitted blend weights are "
            "not simply whatever minimises MAE."
        )


def _plain_skill(value, reference) -> str:
    """A raw error score as a sentence, anchored on a no-information baseline.

    "Brier 0.076" is uninterpretable without knowing what good looks like, and
    nothing on this page said. Against the dumbest forecast that still respects
    the data -- one constant for everybody -- it becomes readable, and on points
    it becomes *informative*: no source currently beats that constant on average
    miss, which is the single most useful thing this tab can tell a reader.
    """
    if not reference:
        return ""
    skill = projection_accuracy.skill_against_no_skill(value, reference)
    if skill is None:
        return ""
    baseline = reference.get("label", "no information")
    if skill >= 0.02:
        return ("<b>%.0f%%</b> of the way from <i>%s</i> to perfect"
                % (skill * 100, baseline))
    if skill <= -0.02:
        return ("<b>%.0f%% worse</b> than simply <i>%s</i> — the ordering is "
                "what separates the sources" % (abs(skill) * 100, baseline))
    return "<b>level with</b> simply <i>%s</i>" % baseline


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
    verdicts = [(label, metric, fmt, projection_accuracy.verdict(scope),
                 projection_accuracy.no_skill_reference(scope))
                for scope, label, metric, fmt in questions]
    if not any(v.get("best") for *_, v, _ref in verdicts):
        return

    st.markdown("#### The short version")
    cols = st.columns(len(verdicts))
    for col, (label, metric, fmt, v, ref) in zip(cols, verdicts):
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
                      <div style="font-size:0.9rem;color:#e0e0e0;">
                        {_plain_skill(v['value'], ref)}
                      </div>
                      <div style="font-size:0.8rem;color:#8fa3c8;margin-top:.2rem;">
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
        "actually started. **Brier** is the average squared miss on a "
        "probability: say 100% for a player who starts and you score 0, say it "
        "for one who does not and you score 1 — so it rewards being right *and* "
        "being appropriately unsure, and a source that says 100% for everybody "
        "scores badly rather than looking good. **AUC** is ordering alone: the "
        "chance a player who started was rated above one who did not, where 0.5 "
        "is a coin flip. A source can order perfectly and still state every "
        "probability far too high, which is why **bias** is here as well."
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
    recommended = fit.get("recommended") or current
    rows = [{
        "Source": _SOURCE_LABELS.get(k, k),
        "Current": current.get(k, 0.0) * 100,
        "Best fit": fitted.get(k, 0.0) * 100,
        "Recommended": recommended.get(k, 0.0) * 100,
    } for k in fitted]
    render_styled_table(
        pd.DataFrame(rows),
        col_formats={"Current": "{:.0f}%", "Best fit": "{:.0f}%",
                     "Recommended": "{:.0f}%"},
    )
    st.caption(
        "Current weights give MAE **%.3f**; the best fit gives **%.3f** "
        "(%.3f better)." % (fit["current_mae"], fit["mae"], gain)
    )

    # **The best fit is not the recommendation, and saying so is the point.**
    # An argmin always names something: this surface is monotone with its
    # minimum on the grid edge, and a whole range of weights sits inside the
    # tie band, so reporting one winner overstates what the sample can tell.
    band = fit.get("tie_band") or []
    if len(band) > 1:
        st.caption(
            "**%d weightings are within %.1f%% of that best fit**, so this "
            "sample cannot choose between them. *Recommended* is the smallest "
            "change that reaches the band — which also avoids adopting a "
            "weight of 0%% or 100%%, where the grid simply ran out of room "
            "rather than finding an optimum."
            % (len(band), projection_accuracy.WEIGHT_FIT_TIE_BAND * 100)
        )
    if fit.get("configured_in_band"):
        st.success(
            "The configured weights are already inside the band — there is no "
            "change this history supports making."
        )
    elif recommended != current:
        st.info(
            "Configured weights sit outside the band. The smallest supported "
            "change is "
            + " · ".join("%s %.0f%%" % (_SOURCE_LABELS.get(k, k), v * 100)
                         for k, v in recommended.items() if v > 0)
            + "."
        )

    if fit["gameweeks"] < projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE:
        st.warning(
            "This fit is on %d gameweek%s. Weights chosen from a sample this "
            "small would be fitting noise — the app's configured weights are "
            "deliberately left alone until there are at least %d."
            % (fit["gameweeks"], "" if fit["gameweeks"] == 1 else "s",
               projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE)
        )

    # **"Would adding an unused source help?" is a different question from
    # "which weighting is best".** The grid search answers the second, and its
    # argmin always names something -- a one-step improvement is exactly what
    # noise looks like. A source sitting at zero gets the two tests the rest of
    # this tab trusts instead: the paired interval, and consistency across
    # gameweeks.
    unused = [k for k, v in current.items() if not v]
    marginals = [projection_accuracy.marginal_source_value(k) for k in unused]
    marginals = [m for m in marginals if m]
    if marginals:
        st.markdown("##### Would adding an unused source help?")
        render_styled_table(pd.DataFrame([{
            "Source": _SOURCE_LABELS.get(m["source"], m["source"]),
            "At weight": m["weight"] * 100,
            "MAE change": -m["mean"],
            "95% interval": "%+.3f to %+.3f" % (-m["ci_high"], -m["ci_low"]),
            "Helped in": "%d of %d GW" % (m["gameweeks_helped"], m["gameweeks"]),
            "Verdict": "worth adding" if m["supported"] else "no evidence",
        } for m in marginals]), col_formats={"At weight": "{:.0f}%",
                                             "MAE change": "{:+.3f}"})
        st.caption(
            "A negative *MAE change* would mean the blend got more accurate. "
            "**No evidence** means the interval still contains zero, or it "
            "helped in half the gameweeks or fewer — either way the sample "
            "cannot tell the source apart from noise, so its weight stays 0. "
            "That is a result, not a gap: a source can be the worst of the "
            "four on its own and still carry information none of the others "
            "do, and this is the test that would find it."
        )


def _gw_list(gws) -> str:
    """'GW1, GW2 and GW3', or 'none'."""
    if not gws:
        return "none"
    labels = [f"GW{g}" for g in gws]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


#: The Hub's top-level sections, in order. Only the selected one is rendered --
#: see :func:`show_player_projections_page`.
HUB_SECTIONS = ("GW Projections", "Horizon", "Odds", "Accuracy")


def _render_hub_section(section: str) -> None:
    """Render one top-level section. This is the lazy unit.

    Split out from the page function so each section can be rendered -- and
    tested -- on its own. That matters for coverage: the nav is a radio, and a
    mocked radio returns only its first option, so without a seam here three of
    the four sections would silently lose their smoke test the moment the page
    stopped rendering all of them.

    The *inner* grouping stays ``st.tabs``, which renders all of its bodies.
    That is deliberate: a whole inner group costs 0.02-0.11s warm because the
    three sub-tabs share one cached blend and one cached FFP payload, so making
    them lazy too would trade a native-looking tab strip for nothing.
    """
    if section == "Horizon":
        render_horizon()
        return

    if section == "Accuracy":
        render_accuracy()
        return

    if section == "Odds":
        sub_goals, sub_cs, sub_match = st.tabs([
            "Goal/Assist", "Clean Sheet", "Match Odds",
        ])
        with sub_goals:
            render_goalscorer_odds()
        with sub_cs:
            render_clean_sheet_odds()
        with sub_match:
            render_match_odds()
        return

    # Default: GW Projections. Blended leads. The individual sources stay one
    # click away rather than being hidden: the blend is the answer, but seeing
    # what fed it is how you judge whether to trust it on a particular player.
    sub_blended, sub_rotowire, sub_ffp = st.tabs([
        "Blended", "Rotowire", "FFP Data",
    ])
    with sub_blended:
        render_blended_projections()
    with sub_rotowire:
        render_rotowire_projections()
    with sub_ffp:
        render_ffp_data()


def show_player_projections_page():
    """Main projections hub page, grouped by the question each section answers.

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

    **The top level is a radio rather than ``st.tabs`` so that it can be lazy.**
    ``st.tabs`` renders every one of its bodies on every rerun -- the switch is
    client-side CSS -- so the page paid for all four sections on every
    interaction regardless of which one was open. Measured warm, Accuracy alone
    was 0.61s of a 0.92s rerun, entirely for a report nobody was looking at.
    Rendering only the selected section takes a rerun on the section you *are*
    looking at to 0.11-0.18s, which is what every filter change and search
    keystroke costs.

    The trade is that a radio does not look like a tab strip. It is the only
    lazy option here: ``st.segmented_control`` and ``st.pills`` arrived in
    Streamlit 1.40 and this app pins 1.38, and styling the radio to look like
    tabs means a blind selector against Streamlit's internal DOM that would also
    restyle the Horizon view toggle on this same page.
    """
    st.title("Projections Hub")
    # One sentence. The four section names say what they are, and the radio's
    # own `help` carries the note about only the open section being computed --
    # this used to restate both at four clauses, above the board.
    st.caption("What every player is worth this week, and how much to believe it.")

    section = st.radio(
        "Section", HUB_SECTIONS, horizontal=True, key="hub_section",
        label_visibility="collapsed",
        help="Only the section you are viewing is computed, so switching costs "
             "a load and staying put does not.",
    )
    st.markdown("---")

    # A mocked or unset radio can hand back None; fall back rather than render
    # a blank page.
    _render_hub_section(section if section in HUB_SECTIONS else HUB_SECTIONS[0])

    # The footer's "About Data Sources" panel is deliberately gone. It was a
    # second, hand-maintained answer to a question the live Source detail panel
    # already answers, and it had drifted: five sources, no weights, and
    # neither projected-XI feed. Its three pieces of real content moved to
    # where each is actually needed -- update frequencies into that panel, the
    # colour scale into the Accuracy glossary, the match-card legend under the
    # match cards.
