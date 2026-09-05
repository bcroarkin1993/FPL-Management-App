"""
Projection source adapters — pure, Streamlit-free.

Every projection source in the app is fetched here and returned as a
:class:`SourceResult`: a frame plus a *declared basis*. The declaration is the
point. Historically each caller had to remember whether a number was already
multiplied by a start probability, and the app got it wrong three separate
times -- FFP's two prediction columns, ``blend_fixture_projections`` blending
the unconditional one, and the dead ``apply_availability_penalty``. A source
that states its own basis lets :mod:`scripts.common.projection_engine` do the
conversion once, in one place, so no caller can double-discount.

Streamlit-free is a hard constraint: the GitHub Actions snapshot collector
imports this module, and that workflow installs requirements best-effort. The
cached wrappers live in ``scraping.py``, which delegates here.
"""

import logging
import re
from datetime import datetime
from typing import NamedTuple, Optional

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

from scripts.common import ffp_feed

# Named in the repo's ``fpl_app.*`` convention, but via plain logging:
# ``error_helpers.get_logger`` imports Streamlit, and this module must not.
_logger = logging.getLogger("fpl_app.projections")

_HEADERS = {"User-Agent": "Mozilla/5.0"}

#: Floor on the start rate used to recover a conditional projection from an
#: unconditional one. Without it a 2% start probability turns a 0.1-point
#: projection into 5 -- the divisor, not the numerator, decides the blow-up.
FFP_START_RECOVERY_FLOOR = 0.05


# =============================================================================
# The source contract
# =============================================================================

#: Points *if he starts*. Rotowire and FFP's ``StartingPredicted`` are on this
#: basis -- an expert lineup projection is inherently conditional on selection.
BASIS_CONDITIONAL = "conditional"

#: Expected points, start probability already priced in. FPL's ``ep_next`` and
#: FFP's ``Predicted`` are on this basis.
BASIS_UNCONDITIONAL = "unconditional"

#: The source prices only players it expects to start, so absence from it is a
#: signal ("not expected to start"), not a missing value.
COVERS_STARTERS = "starters_only"

#: The source prices everyone, so absence means it simply has no row.
COVERS_ALL = "all_players"


class SourceResult(NamedTuple):
    """One projection source's output, with everything needed to blend it.

    ``df`` is keyed on ``Player_ID`` (the FPL element id) and carries whichever
    of ``Proj_Start`` / ``Start_Pct`` / ``Proj_Next3`` the source can supply.
    The engine fills in the rest from the declared ``basis``.

    Mirrors the shape of ``FFPFeed`` (``scraping.py``), which already proved
    that carrying provenance beside the numbers is what lets the UI say where a
    projection came from and how old it is.
    """

    name: str
    df: Optional[pd.DataFrame]
    basis: str
    covers: str
    gameweek: Optional[int] = None
    updated: Optional[datetime] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.df.empty

    def is_stale(self, expected_gw: Optional[int]) -> bool:
        """True only when both gameweeks are known and disagree.

        An *unknown* gameweek is not a wrong one -- same rule as
        ``FFPFeed.is_stale`` and ``ffp_gameweek_matches``.
        """
        if expected_gw is None or self.gameweek is None:
            return False
        return int(self.gameweek) != int(expected_gw)



# Rotowire publishes at least two distinct table shapes under the same CSS
# classes: the standard weekly rankings table (12 columns, includes
# per-position rank breakdown and a Matchup/TSB% column) and a preseason
# "best picks" preview table (6 columns: Rank, Player, Team, Pos, Price,
# Adj Total -- no per-position ranks, no matchup). A fixed column-count/order
# assumption silently drops every row of whichever shape it wasn't written
# for. Since both shapes include a real <thead> with clear column names,
# map by header text instead -- resilient to Rotowire adding, removing, or
# reordering columns, not just these two known shapes.
_ROTOWIRE_HEADER_ALIASES = {
    'Overall Rank': ('overall rank', 'rank'),
    'FW Rank': ('fw rank',),
    'MID Rank': ('mid rank',),
    'DEF Rank': ('def rank',),
    'GK Rank': ('gk rank',),
    'Player': ('player',),
    'Team': ('team',),
    'Matchup': ('matchup',),
    'Position': ('pos', 'position'),
    'Price': ('price',),
    'TSB %': ('tsb%', 'tsb %'),
    'Points': ('pts', 'points', 'adj total'),
}


def _map_rotowire_header_row(header_cells):
    """Map a <thead> row's header text to canonical field -> column index."""
    lower_headers = [h.strip().lower() for h in header_cells]
    mapping = {}
    for canonical, aliases in _ROTOWIRE_HEADER_ALIASES.items():
        for alias in aliases:
            if alias in lower_headers:
                mapping[canonical] = lower_headers.index(alias)
                break
    return mapping


def _rotowire_row_from_header_map(cells, header_map, safe_numeric):
    """Build a canonical row dict from a <td> row using a header->index map.
    Returns None if the row can't be resolved to at least Player + Team
    (e.g. a stray header/subtitle row inside <tbody>)."""
    def get(field, default=""):
        idx = header_map.get(field)
        return cells[idx] if idx is not None and idx < len(cells) else default

    player, team = get('Player'), get('Team')
    if not player or not team:
        return None

    return {
        'Overall Rank': get('Overall Rank'),
        'FW Rank': safe_numeric(get('FW Rank')),
        'MID Rank': safe_numeric(get('MID Rank')),
        'DEF Rank': safe_numeric(get('DEF Rank')),
        'GK Rank': safe_numeric(get('GK Rank')),
        'Player': player,
        'Team': team,
        'Matchup': get('Matchup'),
        'Position': get('Position'),
        'Price': safe_numeric(get('Price')),
        'TSB %': get('TSB %'),
        'Points': safe_numeric(get('Points')),
    }


# Rotowire's "best picks for gameweeks X-Y" articles are not a projection source at
# all. Their points column is headed "Adj Total" and holds an adjusted value metric
# accumulated over the whole range -- not projected points for any single gameweek.
# Scaling it down by the number of gameweeks would not recover a projection, it would
# just make a fabricated number look plausible, so these are refused outright.
# config._discover_rotowire_article() already declines to select them; this is the
# backstop for a URL pinned by hand via ROTOWIRE_URL.
_ROTOWIRE_RANGE_ARTICLE_RE = re.compile(r"best-fpl-picks-for-gameweeks-(\d+)-(\d+)-")

# Above this, a "single gameweek" projection table is not credible (the highest realistic
# single-GW Rotowire projection is well under 10) — used as a tripwire, not a correction.
_ROTOWIRE_MAX_PLAUSIBLE_MEDIAN = 10


def fetch_rotowire_projections(url, limit=None):
    """
    Fetches fantasy rankings and projected points for players from RotoWire.

    Parameters:
    - url (str): URL to fetch the data from.
    - limit (int, optional): Number of players to display. Defaults to None (displays all players).

    Returns:
    - DataFrame: A DataFrame containing player rankings, projected points, and calculated value.
                 Returns empty DataFrame on error.
    """
    LEGACY_EXPECTED_COLUMNS = 12  # Fallback positional layout, used only if no <thead> is found

    # Helper to safely convert to numeric
    def _safe_numeric(val, default=0):
        if val is None:
            return default
        s = str(val).strip()
        if s in {"#N/A", "N/A", "", "-", "—"}:
            return default
        s = re.sub(r"[£$,%]", "", s)  # Strip currency/formatting
        s = s.replace("\u200b", "").replace("\xa0", "").strip()
        try:
            return float(s)
        except ValueError:
            return default

    # Download the page
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        _logger.warning("Failed to fetch Rotowire projections from %s: %s", url, e)
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find table with fallback selectors (most specific to least specific)
    table = soup.select_one("table.article-table__tablesorter.article-table__standard.article-table__figure")
    if table is None:
        table = soup.select_one("table.article-table__tablesorter")
        if table:
            _logger.info("Rotowire: Using fallback table selector (article-table__tablesorter)")
    if table is None:
        table = soup.select_one("table.ck-table-resized")
        if table:
            _logger.info("Rotowire: Using ck-table-resized selector")
    if table is None:
        table = soup.find("table")
        if table:
            _logger.info("Rotowire: Using generic table selector")
    if table is None:
        _logger.warning("Rotowire: Could not locate any table on page %s", url)
        return pd.DataFrame()

    # Extract rows from table body or table directly
    try:
        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")
    except AttributeError as e:
        _logger.warning("Rotowire: Error extracting table rows: %s", e)
        return pd.DataFrame()

    thead = table.find("thead")
    header_map = {}
    if thead:
        header_cells = [th.get_text(strip=True) for th in thead.find_all("th")]
        header_map = _map_rotowire_header_row(header_cells)
        if 'Player' not in header_map or 'Team' not in header_map:
            _logger.warning(
                "Rotowire: <thead> found but couldn't resolve Player/Team columns from %s -- "
                "falling back to legacy positional parsing.",
                header_cells,
            )
            header_map = {}

    # Parse each row
    data = []
    skipped_rows = 0
    for tr in rows:
        tds = tr.find_all("td")
        if not tds:
            skipped_rows += 1
            continue
        cells = [td.get_text(strip=True) for td in tds]

        if header_map:
            row_data = _rotowire_row_from_header_map(cells, header_map, _safe_numeric)
            if row_data is None:
                skipped_rows += 1
                continue
            data.append(row_data)
            continue

        # Legacy fallback: fixed 12-column positional layout (no <thead> on the page)
        if len(cells) < LEGACY_EXPECTED_COLUMNS:
            skipped_rows += 1
            continue
        cells = cells[:LEGACY_EXPECTED_COLUMNS]
        try:
            row_data = {
                'Overall Rank': cells[0],
                'FW Rank': _safe_numeric(cells[1]),
                'MID Rank': _safe_numeric(cells[2]),
                'DEF Rank': _safe_numeric(cells[3]),
                'GK Rank': _safe_numeric(cells[4]),
                'Player': cells[5],
                'Team': cells[6],
                'Matchup': cells[7],
                'Position': cells[8],
                'Price': _safe_numeric(cells[9]),
                'TSB %': cells[10],
                'Points': _safe_numeric(cells[11]),
            }
            data.append(row_data)
        except IndexError as e:
            _logger.warning("Rotowire: Error parsing row, skipping: %s", e)
            skipped_rows += 1
            continue

    if skipped_rows > 0:
        _logger.debug("Rotowire: Skipped %d rows with unexpected structure", skipped_rows)

    if not data:
        _logger.warning("Rotowire: No valid player data extracted from %s", url)
        return pd.DataFrame()

    # Create DataFrame
    player_rankings = pd.DataFrame(data)

    # Refuse multi-gameweek range articles outright -- see _ROTOWIRE_RANGE_ARTICLE_RE.
    # Returning nothing surfaces the app's existing "projections unavailable" warning,
    # which is a far better outcome than plausible-looking invented numbers.
    if _ROTOWIRE_RANGE_ARTICLE_RE.search(url or ""):
        _logger.error(
            "Rotowire: %s is a multi-gameweek 'best picks' article, not a gameweek "
            "projection table (its column is an adjusted value total, not points). "
            "Refusing to use it. Pin a single-gameweek rankings article via "
            "ROTOWIRE_URL, or leave it unset to let discovery find one.",
            url,
        )
        return pd.DataFrame()

    # Tripwire for the next slug shape Rotowire invents: if the table still doesn't look
    # like single-gameweek data, say so loudly rather than silently inflating every score.
    _median_points = player_rankings['Points'].median()
    if pd.notna(_median_points) and _median_points > _ROTOWIRE_MAX_PLAUSIBLE_MEDIAN:
        _logger.warning(
            "Rotowire: median Points is %.1f for %s -- implausible for a single gameweek. "
            "This article may be a multi-GW or season-long table; check ROTOWIRE_URL discovery.",
            _median_points, url,
        )

    # Create 'Pos Rank' by summing the four position ranks
    player_rankings['Pos Rank'] = (
        player_rankings['FW Rank'] + player_rankings['MID Rank'] +
        player_rankings['DEF Rank'] + player_rankings['GK Rank']
    ).astype(int)

    # Drop individual position rank columns
    player_rankings.drop(columns=['FW Rank', 'MID Rank', 'DEF Rank', 'GK Rank'], inplace=True)

    # Create the 'Value' column by dividing 'Points' by 'Price'
    player_rankings['Value'] = player_rankings.apply(
        lambda row: row['Points'] / row['Price'] if row['Price'] > 0 else float('nan'), axis=1
    )

    # If a limit is provided, return only the top 'limit' players
    if limit:
        player_rankings = player_rankings.head(limit)

    # Format the DataFrame to remove the index and reset it with a starting value of 1
    player_rankings.reset_index(drop=True, inplace=True)
    player_rankings.index = player_rankings.index + 1

    _logger.debug("Rotowire: Successfully parsed %d players from %s", len(player_rankings), url)
    return player_rankings


def ffp_conditional_points(predicted, starting_predicted, start_pct):
    """FFP's *conditional* projection -- points if he starts.

    FFP publishes both bases and they differ by exactly the start probability
    (verified live at r=0.9998, MAE 0.0003). Which column you take is the whole
    ball game: taking ``Predicted`` and then applying start likelihood is the
    double discount that ran the FFP term ~44% low at a 60% median start rate,
    silently, because every value involved stayed plausible.

    Prefer ``StartingPredicted``; where FFP published only ``Predicted``, divide
    the start rate back out. A published 0 means "no prediction", not a
    prediction of zero.

    Args:
        predicted: FFP ``Predicted`` (unconditional), or None.
        starting_predicted: FFP ``StartingPredicted`` (conditional), or None.
        start_pct: start probability as a 0-1 fraction.

    Returns:
        A Series of conditional points, NaN where FFP said nothing.
    """
    # Callers reach this with ``df.get("FFP_Predicted")``, which returns **None**
    # for an absent column -- pd.to_numeric(None) is then a scalar and every
    # Series method below raises. That is the documented `DataFrame.get()` trap,
    # and it only fires on a degraded upstream, which is exactly when it must not.
    start_pct = _as_series(start_pct, index=None)
    index = start_pct.index
    conditional = _as_series(starting_predicted, index=index)
    unconditional = _as_series(predicted, index=index)

    recovered = unconditional / start_pct.clip(lower=FFP_START_RECOVERY_FLOOR)
    conditional = conditional.fillna(recovered.where(start_pct.gt(0), unconditional))
    return conditional.where(conditional.gt(0))


def _column(df: pd.DataFrame, col: str) -> pd.Series:
    """A numeric Series for ``col``, all-NaN when the column is absent.

    The same guard as ``analytics.numeric_col``, repeated here because this
    module must not import from analytics (Streamlit).
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _as_series(value, index) -> pd.Series:
    """A numeric Series for ``value``, all-NaN when the column was absent."""
    if value is None:
        return pd.Series(np.nan, index=index if index is not None else pd.RangeIndex(0),
                         dtype="float64")
    if not isinstance(value, pd.Series):
        value = pd.Series(value, index=index)
    return pd.to_numeric(value, errors="coerce")


# =============================================================================
# Source adapters
# =============================================================================
#
# Each returns a SourceResult keyed on Player_ID where it can resolve one, and
# on (Player, Team) where it cannot. Rotowire is the only source that publishes
# names alone; the engine resolves it through ReferenceMatcher.


def rotowire_source(url, limit=None) -> SourceResult:
    """Rotowire's weekly rankings table.

    Basis is **conditional**: Rotowire projects what a player scores if he
    starts, and it only lists players it expects to start -- 20 clubs x 11.
    That coverage is itself the signal ``covers=starters_only`` records: a
    player absent from this table is not un-priced, he is not expected to
    start, which is why the engine applies a start prior rather than treating
    the gap as missing data.

    Rotowire publishes no start percentage at all, so ``Start_Pct`` is absent
    here and comes from FFP or the FPL bootstrap.
    """
    if not url:
        return SourceResult("rotowire", None, BASIS_CONDITIONAL, COVERS_STARTERS,
                            note="No Rotowire article URL resolved for this gameweek.")
    df = fetch_rotowire_projections(url, limit=limit)
    if df is None or df.empty:
        return SourceResult("rotowire", None, BASIS_CONDITIONAL, COVERS_STARTERS,
                            note="Rotowire returned no rows (unreachable, or the "
                                 "article is not a single-gameweek table).")
    out = df[["Player", "Team", "Position", "Points"]].copy()
    out = out.rename(columns={"Points": "Proj_Start"})
    out["Proj_Start"] = pd.to_numeric(out["Proj_Start"], errors="coerce")
    return SourceResult("rotowire", out, BASIS_CONDITIONAL, COVERS_STARTERS)


def fetch_ffp_table(timeout: int = ffp_feed.DEFAULT_TIMEOUT, gameweek=None):
    """FFP's table for ``gameweek``: live payload, then archive, then sheet.

    Returns ``(df, gameweek, updated, provenance, note)``. This is the pure body
    behind ``scraping.get_ffp_feed()`` -- that function wraps it in
    ``@st.cache_data`` and packs the result into an ``FFPFeed``. Kept in one
    place so the Actions collector and the app read FFP identically.

    **The archive step is what keeps FFP available for a whole gameweek.** FFP
    publishes GW N+1 as soon as GW N kicks off, so for most of every gameweek
    the live payload describes a week the app is not scoring, and the gameweek
    gate correctly discards it -- silently taking FFP off every page partway
    through the week. Measured 2026-09-05: the payload had rolled to a GW4-GW9
    window while the app was still scoring GW3, and because FFP was persisted
    nowhere, its GW3 numbers no longer existed anywhere.

    Every fetch therefore archives all six gameweeks of the payload before
    returning, and a request for a gameweek the live payload does not cover
    falls back to the archive. See :mod:`scripts.common.ffp_archive`.

    A failure is never reported as a success: on total failure ``df`` is None
    and ``provenance`` is ``"none"`` with the reason in ``note``.
    """
    from scripts.common import ffp_archive

    live_gw = None
    try:
        rows, gw, updated = ffp_feed.fetch_points_predictor()
        if rows:
            live_gw = gw
            code_to_id = ffp_feed.bootstrap_code_to_id(timeout=timeout)
            # Archive the whole window first, so the data is kept even when this
            # call ends up serving a different gameweek from the archive below.
            try:
                ffp_archive.archive_payload(rows, gw, updated, code_to_id)
            except Exception as e:                  # never let archiving break a read
                _logger.warning("FFP: could not archive payload: %s", e)

            if gameweek is None or gw == gameweek:
                df = ffp_feed.to_sheet_schema(rows, gw, code_to_id)
                if df is not None and not df.empty:
                    goal_rows, _, _ = ffp_feed.fetch_goal_assist()
                    cs_rows, _, _ = ffp_feed.fetch_clean_sheet()
                    df = ffp_feed.attach_odds_columns(df, goal_rows, cs_rows, gw)
                    return df, gw, updated, "site", ""
    except Exception as e:
        _logger.warning("FFP site feed failed, falling back to the sheet: %s", e)

    # The live payload is for another gameweek (or unreachable). Serve the
    # archived table for the one actually being scored.
    if gameweek is not None:
        archived, meta = ffp_archive.load_gameweek(gameweek)
        if archived is not None and not archived.empty:
            note = ""
            if meta.get("provenance") == ffp_archive.PROV_SHEET_OFFSET:
                note = ("recovered from FFP's spreadsheet, which publishes no "
                        "revision time")
            elif live_gw is not None:
                note = f"archived — FFP's live payload has moved on to GW{live_gw}"
            else:
                note = "archived — FFP's site was unreachable"
            # Back to a datetime: FFPFeed.updated is consumed by
            # format_last_updated(), which needs one. JSON only carries strings.
            return (archived, gameweek, ffp_archive.parse_updated(meta.get("updated")),
                    "archive", note)

    try:
        sheet = ffp_feed.fetch_sheet()
    except Exception as e:
        _logger.warning("FFP sheet fetch failed: %s", e)
        sheet = None

    if sheet is None or sheet.empty:
        return (None, None, None, "none",
                "Neither the Fantasy Football Pundit site nor its published "
                "spreadsheet could be read.")

    sheet_gw = ffp_feed.resolve_ffp_gameweek(sheet)
    if sheet_gw is not None:
        sheet = sheet.copy()
        sheet["FFP_GW"] = sheet_gw
    return (sheet, sheet_gw, None, "sheet",
            "Read from FFP's published spreadsheet; their site payload was "
            "unavailable.")


def ffp_source(ffp_df=None, gameweek=None, updated=None, note="") -> SourceResult:
    """Fantasy Football Pundit's points predictor.

    Basis is **conditional**, because we take ``StartingPredicted``. FFP
    publishes both bases and they differ by exactly the start probability
    (verified live at r=0.9998), so which column you take is the whole ball
    game -- taking ``Predicted`` and then applying start likelihood is the
    double-discount that ran the FFP term ~44% low. Where FFP has published only
    the unconditional column, the conditional one is recovered by dividing the
    start rate back out.

    Pass ``ffp_df`` to reuse an already-fetched frame (the app has one cached);
    omit it to fetch.
    """
    if ffp_df is None:
        ffp_df, gameweek, updated, provenance, note = fetch_ffp_table()
        if provenance == "none":
            return SourceResult("ffp", None, BASIS_CONDITIONAL, COVERS_ALL, note=note)

    if ffp_df is None or ffp_df.empty:
        return SourceResult("ffp", None, BASIS_CONDITIONAL, COVERS_ALL,
                            note=note or "FFP returned no rows.")

    out = pd.DataFrame(index=ffp_df.index)
    name_col = "Name" if "Name" in ffp_df.columns else "Player"
    # Every read here goes through a column-or-NaN helper. `df.get(missing)`
    # returns **None**, and pd.to_numeric(None) is a scalar -- so the natural
    # spelling either raises on the next Series method or, worse, broadcasts one
    # value to every row. An archived table recovered from the spreadsheet has no
    # Next3GWs at all, so this is the normal case, not a degraded one.
    out["Player"] = ffp_df[name_col] if name_col in ffp_df.columns else None
    out["Team"] = ffp_df["Team"] if "Team" in ffp_df.columns else None
    out["Position"] = ffp_df["Position"] if "Position" in ffp_df.columns else None
    if "Player_ID" in ffp_df.columns:
        out["Player_ID"] = pd.to_numeric(ffp_df["Player_ID"], errors="coerce")
    if "Display_Name" in ffp_df.columns:
        out["Display_Name"] = ffp_df["Display_Name"]

    start = _column(ffp_df, "Start") / 100.0
    out["Start_Pct"] = start.clip(0, 1)

    out["Proj_Start"] = ffp_conditional_points(
        ffp_df.get("Predicted"), ffp_df.get("StartingPredicted"), start
    )

    next3 = _column(ffp_df, "Next3GWs")
    out["Proj_Next3"] = next3.where(next3.gt(0))

    if gameweek is None and "FFP_GW" in ffp_df.columns:
        gws = pd.to_numeric(ffp_df["FFP_GW"], errors="coerce").dropna().unique()
        gameweek = int(gws[0]) if len(gws) == 1 else None

    return SourceResult("ffp", out, BASIS_CONDITIONAL, COVERS_ALL,
                        gameweek=gameweek, updated=updated, note=note)


def fpl_ep_source(bootstrap=None, gameweek=None, timeout: int = 20) -> SourceResult:
    """FPL's own expected points (``ep_next``) from the bootstrap.

    Basis is **unconditional** -- FPL's expected points already price in the
    chance of playing, which is why blending it against Rotowire without saying
    so was wrong. It was already in the app: ``classic/transfers.py`` wrote it
    into the Rotowire column when Rotowire had not published, so it silently
    took Rotowire's blend weight while being labelled Rotowire. Declaring it a
    source with its own basis and weight is the fix.

    It covers every player in the game, which makes it the natural accuracy
    baseline: a blend that cannot beat FPL's own number is worth knowing about.
    """
    if bootstrap is None:
        try:
            resp = requests.get(ffp_feed.BOOTSTRAP_URL, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            bootstrap = resp.json()
        except Exception as e:
            _logger.warning("FPL bootstrap fetch failed: %s", e)
            return SourceResult("fpl_ep", None, BASIS_UNCONDITIONAL, COVERS_ALL,
                                note=f"FPL bootstrap unreachable: {e}")

    elements = (bootstrap or {}).get("elements") or []
    if not elements:
        return SourceResult("fpl_ep", None, BASIS_UNCONDITIONAL, COVERS_ALL,
                            note="FPL bootstrap carried no elements.")

    # FPL publishes two expectations and they describe different gameweeks:
    # `ep_this` is the event flagged `is_current`, `ep_next` the one flagged
    # `is_next`. Taking `ep_next` unconditionally makes this source describe
    # GW+1 the moment a deadline passes, and the engine's gameweek gate then
    # (correctly) throws it away -- so the source silently vanishes for most of
    # every gameweek. Pick the column that matches the gameweek being asked for.
    current_gw = next_gw = None
    for ev in (bootstrap or {}).get("events") or []:
        if ev.get("is_current"):
            current_gw = int(ev.get("id"))
        if ev.get("is_next"):
            next_gw = int(ev.get("id"))

    if gameweek is not None and current_gw == int(gameweek):
        ep_field, source_gw = "ep_this", current_gw
    elif gameweek is not None and next_gw == int(gameweek):
        ep_field, source_gw = "ep_next", next_gw
    else:
        # Cannot tell which gameweek is wanted: default to `ep_next` and report
        # the gameweek it really describes, so the gate can judge it.
        ep_field, source_gw = "ep_next", next_gw

    pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    rows = []
    for e in elements:
        rows.append({
            "Player_ID": int(e.get("id")),
            "Position": pos_map.get(e.get("element_type")),
            "Proj": pd.to_numeric(e.get(ep_field), errors="coerce"),
            "Chance_Of_Playing": e.get("chance_of_playing_next_round"),
        })
    out = pd.DataFrame(rows)
    out["Proj"] = pd.to_numeric(out["Proj"], errors="coerce")
    out["Proj"] = out["Proj"].where(out["Proj"].gt(0))
    # The bootstrap's own availability read, used as the Start_Pct fallback when
    # FFP has not published. A null here means "no news", i.e. fully available.
    chance = pd.to_numeric(out["Chance_Of_Playing"], errors="coerce") / 100.0
    out["Start_Pct"] = chance.clip(0, 1)
    out = out.drop(columns=["Chance_Of_Playing"])

    return SourceResult("fpl_ep", out, BASIS_UNCONDITIONAL, COVERS_ALL,
                        gameweek=source_gw,
                        note=f"FPL {ep_field} for GW{source_gw}" if source_gw else "")
