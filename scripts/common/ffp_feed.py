"""
Fantasy Football Pundit feed — the site's own payload, with the sheet as fallback.

FFP publishes the same numbers twice, and the two disagree.

The published Google Sheet this app read for a year has stopped keeping pace.
Measured 2026-09-04, with the app and Rotowire both on GW3, the sheet's
``Fixture`` column still described **GW2** — and not even consistently: Aston
Villa and Man City each carried a leftover GW1 fixture string alongside the GW2
one. Meanwhile fantasyfootballpundit.com said "Updated for GW3 - 4 September at
16:18". The app was blending Rotowire GW3 at 60% with FFP GW2 at 40% and
applying GW2 start probabilities, silently, because every individual value was
plausible.

The site is a Next.js app that server-embeds its tables in the RSC flight
payload (``self.__next_f.push([1,"..."])``). Concatenating those strings,
JSON-unescaping each and ``raw_decode``-ing from ``"rows":[`` yields clean
records that are strictly better than the sheet in four ways:

1. **``gw`` is on every row.** Which gameweek a table describes stops being an
   inference. A set-overlap test on team names can never answer it — all 20
   clubs play every gameweek, so the old check scored 18/19 for GW2, GW3 *and*
   GW4 and was a no-op.
2. **``player_code`` is the FPL bootstrap ``code``** (368/368 resolved live), so
   FFP joins on an integer id instead of a name.
3. **``fixture_count``** makes doubles and blanks representable.
4. Six gameweeks of forecasts per player, not five relative-offset columns.

**The two point columns are named the opposite way round from the sheet.**
Verified over all 2208 live rows: ``predicted_points_start == predicted_points *
start_pct/100`` at MAE 0.0003, while the reverse relation is off by 2.31. So the
site's ``predicted_points`` is the *conditional* value (the sheet's
``StartingPredicted``) and ``predicted_points_start`` is the *unconditional* one
(the sheet's ``Predicted``). Mapping these across by name rather than by basis
re-introduces the double-discount bug recorded in CLAUDE.md under "FFP has two
prediction bases".

``to_sheet_schema()`` is the seam that keeps every existing consumer unchanged:
it emits the legacy sheet column names, so the merges in ``analytics.py`` and
all seven page callsites carry on working.

No Streamlit and no caching here — ``data_source_checks.py`` imports this module
and GitHub Actions imports that. Caching wrappers live in ``scraping.py``.
"""

import json
import os
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from scripts.common.text_helpers import TEAM_FULL_TO_SHORT, to_display_name

try:                                    # zoneinfo is 3.9+ with tzdata present
    from zoneinfo import ZoneInfo
    _TZ_LONDON = ZoneInfo("Europe/London")
except Exception:                       # pragma: no cover - fallback for bare envs
    _TZ_LONDON = None

_logger = logging.getLogger(__name__)

# Every FFP address lives here. These used to be spread across scraping.py and
# data_source_checks.py, which meant a URL change had to be made twice.
# Env-overridable, like every Rotowire URL in config.py. FFP had none, so an
# endpoint change could only be fixed by editing code -- awkward for the Actions
# snapshot collector, which cannot be hand-patched between runs. Read via
# os.getenv rather than config so this module stays importable on its own.
FFP_BASE = "https://www.fantasyfootballpundit.com"
FFP_POINTS_PREDICTOR_URL = (os.getenv("FFP_POINTS_PREDICTOR_URL")
                            or FFP_BASE + "/fpl-points-predictor/")
FFP_GOAL_ASSIST_URL = FFP_BASE + "/premier-league-goalscorer-assist-odds/"
FFP_CLEAN_SHEET_URL = FFP_BASE + "/premier-league-clean-sheet-odds/"
FFP_SHEET_URL = os.getenv("FFP_SHEET_URL") or (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaiTmUKjtQ7MxiGibN2GAZ8m9NHF3"
    "IA2U-yE0PhBpCOXHewhs57PrjZO7GQzZvrEGGBW7HFEE43yX0/pub?output=csv"
)

FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

_HEADERS = {"User-Agent": "Mozilla/5.0"}

# A real fetch of the Google Sheet was observed to exceed the app's old 15s
# timeout. A slow read must not read as "FFP is unavailable".
DEFAULT_TIMEOUT = 20
DEFAULT_ATTEMPTS = 3

#: How many forecast gameweeks the sheet schema carries beyond the current one.
_HORIZON = 5

_FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')
#: "Updated for GW",3," - 4 September at 16:18"  (an RSC-rendered React child list)
_UPDATED_RE = re.compile(r'"Updated for GW",(\d+),"\s*[^\w"]{0,4}\s*([^"]*)"')

_ELEMENT_TYPE_TO_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# =============================================================================
# In-process memo
# =============================================================================
#
# Streamlit's cache is not available here -- this module is Streamlit-free by
# hard constraint, because the Actions snapshot collector imports it. But the
# two fetches below are each called by several *separately* cached Streamlit
# wrappers, so on one cold page load the same page was downloaded more than
# once: measured, the FFP payload twice and bootstrap-static twice, and the
# payload fetch also re-ran `archive_payload`, writing six JSON files each time.
#
#: How long a successful fetch is reused within one process.
#:
#: Deliberately far shorter than the 300s ``@st.cache_data`` TTL on
#: ``get_ffp_feed`` / ``get_ffp_horizon`` that sits above it. That ordering is
#: what makes this memo invisible: it can never serve a payload older than the
#: Streamlit cache above it would have served anyway, so it cannot make any page
#: staler, and it cannot outlive a Refresh click in a way that matters. A cache
#: surviving the very click meant to escape it is a bug this app has already
#: paid for once -- see ``purge_cache_prefix`` -- and the fix there was a
#: *permanent* (``ttl=None``) entry, which this is not.
MEMO_TTL_SECONDS = 60

_memo: Dict[str, tuple] = {}


def _memoised(key: str, fn, ttl: int = MEMO_TTL_SECONDS):
    """Reuse ``fn()``'s result for ``ttl`` seconds -- **successes only**.

    Caching a failure is the bug this app has already paid for once:
    ``get_ffp_feed`` returning a bare ``None`` from ``@st.cache_data`` pinned a
    transient timeout for five minutes, and FFP read "temporarily unavailable"
    in the app while the website plainly worked. So an empty result is returned
    to the caller and *not* stored, and the next caller retries.
    """
    now = time.time()
    hit = _memo.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    if _memo_worth_keeping(value):
        _memo[key] = (now, value)
    return value


def _memo_worth_keeping(value) -> bool:
    """True for a result that actually carries data."""
    if value is None:
        return False
    if isinstance(value, tuple):
        return bool(value and value[0])
    return bool(value)


def clear_memo() -> None:
    """Drop the in-process memo. For tests and for a manual refresh."""
    _memo.clear()


# =============================================================================
# Transport
# =============================================================================

def _get(url: str, timeout: int = DEFAULT_TIMEOUT, attempts: int = DEFAULT_ATTEMPTS,
         **kwargs) -> Optional[requests.Response]:
    """GET with a short backoff. Returns None rather than raising.

    A transient timeout used to surface as "the data source may be temporarily
    unavailable" and — because the caller cached the ``None`` — stayed that way
    for five minutes while the website worked fine.
    """
    delay = 0.5
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == attempts:
                _logger.warning("FFP: giving up on %s after %d attempts: %s",
                                url, attempts, exc)
                return None
            _logger.info("FFP: retrying %s (attempt %d/%d): %s", url, attempt, attempts, exc)
            time.sleep(delay)
            delay *= 2
    return None


# =============================================================================
# RSC flight payload extraction
# =============================================================================

def flight_payload(html: str) -> str:
    """Concatenate the page's RSC flight chunks into one string.

    Each chunk is a JSON string literal, so it is unescaped with ``json.loads``
    rather than by hand — the payload is full of escaped quotes and unicode.
    """
    if not html:
        return ""
    out = []
    for chunk in _FLIGHT_RE.findall(html):
        try:
            out.append(json.loads('"' + chunk + '"'))
        except ValueError:
            continue
    return "".join(out)


def extract_rows(flight: str, key: str = '"rows":[') -> List[dict]:
    """Decode the first JSON array following ``key``.

    ``raw_decode`` stops at the end of the array, so the surrounding React tree
    does not need to be parsed or even be valid JSON.
    """
    if not flight:
        return []
    idx = flight.find(key)
    if idx < 0:
        return []
    start = idx + len(key) - 1          # point at the '['
    try:
        rows, _ = json.JSONDecoder().raw_decode(flight[start:])
    except ValueError as exc:
        _logger.warning("FFP: could not decode rows payload: %s", exc)
        return []
    return rows if isinstance(rows, list) else []


def parse_updated(flight: str) -> Tuple[Optional[int], Optional[datetime]]:
    """Read the page's "Updated for GW N - <date>" stamp.

    Returns ``(gameweek, datetime)``; either may be None. FFP prints a UK
    wall-clock time with no year, so the year is inferred as the one that puts
    the stamp closest to now — a stamp is never more than a season old in
    practice, and guessing forward across New Year would render "11 months ago".
    """
    match = _UPDATED_RE.search(flight or "")
    if not match:
        return None, None
    gw = int(match.group(1))
    when = _parse_uk_datetime(match.group(2))
    return gw, when


def _parse_uk_datetime(text: str) -> Optional[datetime]:
    """Parse "4 September at 16:18" into an aware datetime, or None."""
    if not text:
        return None
    cleaned = " ".join(text.replace(" at ", " ").split())
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{1,2}):(\d{2})", cleaned)
    if not match:
        return None
    day, month_name, hour, minute = match.groups()
    now = datetime.now(_TZ_LONDON) if _TZ_LONDON else datetime.now()
    for year in (now.year, now.year - 1, now.year + 1):
        try:
            naive = datetime.strptime(
                "%s %s %s %s:%s" % (day, month_name[:3], year, hour, minute),
                "%d %b %Y %H:%M",
            )
        except ValueError:
            continue
        stamped = naive.replace(tzinfo=_TZ_LONDON) if _TZ_LONDON else naive
        if abs((now - stamped).days) <= 180:
            return stamped
    return None


# =============================================================================
# The three FFP pages
# =============================================================================

def _fetch_page(url: str, key: str = '"rows":[') -> Tuple[List[dict], Optional[int], Optional[datetime]]:
    resp = _get(url)
    if resp is None:
        return [], None, None
    flight = flight_payload(resp.text)
    rows = extract_rows(flight, key)
    gw, updated = parse_updated(flight)

    # The header string and the row data are independent claims about the same
    # thing. Trust the rows: they are what gets consumed.
    row_gws = sorted({r["gw"] for r in rows if isinstance(r, dict) and isinstance(r.get("gw"), int)})
    if row_gws:
        if gw is not None and gw != row_gws[0]:
            _logger.warning("FFP %s: header says GW%s but rows start at GW%s — using GW%s",
                            url, gw, row_gws[0], row_gws[0])
        gw = row_gws[0]
    return rows, gw, updated


def fetch_points_predictor():
    """Rows for the points predictor, plus its gameweek and publish time.

    Memoised, because two separately-cached Streamlit wrappers need this one
    payload: ``fetch_ffp_table`` for the gameweek being scored and
    ``fetch_ffp_horizon`` for the window. Without it a cold page load downloaded
    the same ~400KB page twice *and* ran ``archive_payload`` twice, writing six
    JSON files each time.
    """
    return _memoised("points_predictor",
                     lambda: _fetch_page(FFP_POINTS_PREDICTOR_URL))


def fetch_goal_assist():
    """Anytime goal/assist/return probabilities. Rows carry no ``gw`` field."""
    return _fetch_page(FFP_GOAL_ASSIST_URL)


def fetch_clean_sheet():
    """Team clean-sheet odds; each row holds a ``cells`` list keyed by gameweek."""
    return _fetch_page(FFP_CLEAN_SHEET_URL)


# =============================================================================
# Sheet-schema projection
# =============================================================================

def _num(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """``col`` as a numeric Series aligned to ``df``, all-NaN when absent.

    ``DataFrame.get("missing")`` returns None, not a Series, so the natural
    ``pd.to_numeric(df.get(col)).fillna(...)`` raises exactly when the column is
    missing -- which is the case it was written for. See "``DataFrame.get()`` is
    not a safe accessor" in CLAUDE.md.
    """
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _prepare_frame(rows: List[dict]) -> Optional[pd.DataFrame]:
    """The payload as a frame, with every numeric column coerced.

    Returns None when the payload cannot carry a projection at all — no rows, or
    missing the two fields every downstream path keys on.
    """
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    if "gw" not in frame.columns or "player_code" not in frame.columns:
        _logger.warning("FFP: payload missing gw/player_code — cannot build schema")
        return None

    frame["gw"] = pd.to_numeric(frame["gw"], errors="coerce")
    for col in ("predicted_points", "predicted_points_start", "start_pct",
                "price", "selected_by_percent", "difficulty"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _gameweek_block(frame: pd.DataFrame, gw: int,
                    code_to_id: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """One row per player for a single **absolute** gameweek, keyed on player_code.

    This is the sole implementation of the two things FFP has made this app get
    wrong before: the **basis inversion** — the site's two point columns are
    named the opposite way round from the sheet's — and the **start_pct
    recovery**, since FFP stops publishing it for the gameweek it is currently
    on. ``to_sheet_schema()`` and ``to_long_schema()`` both go through here, so
    there is exactly one copy of each.

    That matters more than the line count suggests. This codebase has three
    times paid for two implementations of one piece of arithmetic drifting
    apart (the FFP double discount, ``compute_player_scores`` against
    ``blend_fixture_projections``, the ``Proj_Next3`` assembly). A second
    hand-written copy of a basis conversion is that failure set up again.
    """
    current = frame[frame["gw"] == gw].copy()
    if current.empty:
        _logger.warning("FFP: payload carries no rows for GW%s", gw)
        return pd.DataFrame()

    out = pd.DataFrame(index=current.index)
    first = current.get("first_name", pd.Series("", index=current.index)).fillna("")
    second = current.get("second_name", pd.Series("", index=current.index)).fillna("")
    out["Name"] = (first.astype(str) + " " + second.astype(str)).str.strip()
    out["Web_Name"] = current.get("web_name")

    # FFP republishes the FPL bootstrap's full legal name, which is not what
    # anybody says out loud -- "Igor Thiago Nascimento Rodrigues" for a player
    # everyone calls Igor Thiago. `Name` stays the legal one because that is what
    # the merges key on; `Display_Name` is what the UI renders. Match on Name,
    # display Display_Name -- swapping them silently degrades match rates.
    web = current.get("web_name", pd.Series("", index=current.index)).fillna("")
    out["Display_Name"] = [
        to_display_name(f, s_, w) for f, s_, w in zip(first, second, web)
    ]
    out["Team"] = current.get("team_name")
    out["Position"] = pd.to_numeric(current.get("element_type"), errors="coerce").map(
        _ELEMENT_TYPE_TO_POS)

    home = current.get("is_home", pd.Series(False, index=current.index)).fillna(False)
    opponent = current.get("opponent_abbr", pd.Series("", index=current.index)).fillna("")
    out["Fixture"] = [
        "%s (%s)" % (opp, "H" if bool(is_home) else "a")
        for opp, is_home in zip(opponent, home)
    ]

    out["Price"] = current.get("price")
    out["Ownership"] = current.get("selected_by_percent")

    # **FFP stops publishing `start_pct` for the gameweek it is currently on.**
    # Measured 2026-09-25 with the app on GW6: null on all 372 current-week
    # rows, published in full (372 each, mean 58.5%) for GW7 through GW11. So
    # the one week the app actually scores is the one week the app's *primary*
    # start-probability source says nothing about -- and `Start` is primary
    # precisely because it is the only continuous 0-100 start model any source
    # gives us.
    #
    # Nothing downstream could tell. `blend_aligned` falls through FFP, then
    # FPL's `chance_of_playing` (set for 46 of 667 players), then "no news means
    # he plays" -- so every player Rotowire listed resolved to exactly 1.00. On
    # the Projections Hub that was 208 of 220 listed players at 100%, against an
    # FFP view of the same players with a median of 70% and a maximum of 95%:
    # FFP never rates anybody certain, so every 100% on that page was the app's
    # fallback rather than anyone's opinion.
    #
    # The number is recoverable exactly, because the two point columns are the
    # same forecast with and without the discount: `predicted_points_start ==
    # predicted_points x start_pct/100`, the relation `check_ffp_feed()` already
    # guards. Inverted against the 1,860 live rows where FFP *does* publish
    # `start_pct`, the ratio reproduces it to 0.008 percentage points (max
    # 0.05), and the recovered GW6 distribution is identical to GW7's published
    # one to every quartile.
    start = _numeric(current, "start_pct")
    cond_points = _numeric(current, "predicted_points")
    uncond_points = _numeric(current, "predicted_points_start")
    recovered = 100.0 * uncond_points / cond_points.where(cond_points.gt(0))
    out["Start"] = start.fillna(recovered).clip(0, 100)

    # The basis inversion. Site `predicted_points` is conditional on starting,
    # which is the sheet's `StartingPredicted`; site `predicted_points_start`
    # already carries the start discount, which is the sheet's `Predicted`.
    out["StartingPredicted"] = current.get("predicted_points")
    out["Predicted"] = current.get("predicted_points_start")

    out["FFP_GW"] = int(gw)
    out["Player_Code"] = pd.to_numeric(current["player_code"], errors="coerce").astype("Int64")
    out["Fixture_Count"] = pd.to_numeric(
        current.get("fixture_count", 1), errors="coerce").fillna(1).astype(int)
    if code_to_id:
        out["Player_ID"] = out["Player_Code"].map(code_to_id).astype("Int64")

    # The payload has published an FDR per fixture all along and nothing has
    # ever read it. The horizon grid colours its fixtures view with it, which
    # saves the six live HTTP calls `get_fixture_difficulty_grid()` would cost.
    out["Difficulty"] = _numeric(current, "difficulty")

    return out.set_index(pd.Index(current["player_code"].astype("int64"),
                                  name="player_code"))


def to_long_schema(rows: List[dict],
                   code_to_id: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """Every gameweek in the payload, one row per (player, gameweek).

    ``to_sheet_schema()`` keeps one gameweek and folds the other five into ten
    *relative-offset* columns, discarding ~1,895 of the payload's ~2,274 rows on
    every fetch. An offset cannot say which absolute gameweek it describes
    without also knowing the window it was built from — and inferring a
    gameweek is precisely what made FFP wrong in this app before.

    So here ``FFP_GW`` is the **absolute** gameweek and no consumer has to
    resolve anything. Same columns as the sheet block plus ``Difficulty``, and
    none of the offset or cumulative columns: a long frame needs neither.
    """
    frame = _prepare_frame(rows)
    if frame is None:
        return pd.DataFrame()

    blocks = []
    for gw in sorted(frame["gw"].dropna().unique()):
        block = _gameweek_block(frame, int(gw), code_to_id)
        if not block.empty:
            blocks.append(block.reset_index(drop=True))
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def to_sheet_schema(rows: List[dict], gw: Optional[int] = None,
                    code_to_id: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """Project the site payload onto the legacy Google Sheet column names.

    This is the compatibility seam — the same role ``normalise_my_team()`` plays
    for the Classic squad resolver. Every downstream consumer keeps reading
    ``Name`` / ``Team`` / ``Predicted`` / ``StartingPredicted`` / ``Start`` /
    ``NextNGWs`` and needs no change.

    Sheet semantics, pinned against live data so the two sources agree:
      * ``GW2..GW6`` are **relative offsets** — the 2nd..6th gameweek of the
        window, on the conditional (if-he-starts) basis.
      * ``GW2s..GW6s`` are those same weeks discounted by start likelihood.
      * ``NextNGWs`` **includes** the current gameweek:
        ``Next2GWsStart == StartingPredicted + GW2`` (MAE 0.029), not
        ``GW2 + GW3`` (MAE 0.45).

    ``LongStart`` is deliberately **not** emitted. The site publishes one
    ``start_pct`` per player, identical across all six forecast gameweeks, so
    there is no independent long-run rate to report.  ``compute_player_scores()``
    already falls back to the FPL ``starts`` count for start consistency;
    emitting a copy of ``Start`` would instead spend 10% of ROS on a signal that
    1GW already carries.
    """
    frame = _prepare_frame(rows)
    if frame is None:
        return pd.DataFrame()
    if gw is None:
        gw = int(frame["gw"].min())

    out = _gameweek_block(frame, gw, code_to_id)
    if out.empty:
        return pd.DataFrame()

    # `Difficulty` is carried by the block for the horizon grid. This function
    # is the compatibility seam every existing consumer reads and the archive
    # serialises, so its column list stays exactly as it was.
    out = out.drop(columns=["Difficulty"])

    # Forward gameweeks, on both bases.
    cond = frame.pivot_table(index="player_code", columns="gw",
                             values="predicted_points", aggfunc="sum")
    uncond = frame.pivot_table(index="player_code", columns="gw",
                               values="predicted_points_start", aggfunc="sum")
    for offset in range(1, _HORIZON + 1):
        target = gw + offset
        label = "GW%d" % (offset + 1)
        out[label] = cond[target].reindex(out.index) if target in cond.columns else float("nan")
        out[label + "s"] = uncond[target].reindex(out.index) if target in uncond.columns else float("nan")

    # Cumulative windows, current gameweek included.
    running_cond = out["StartingPredicted"].astype(float).copy()
    running_uncond = out["Predicted"].astype(float).copy()
    for offset in range(1, _HORIZON + 1):
        label = "GW%d" % (offset + 1)
        running_cond = running_cond + out[label].astype(float).fillna(0)
        running_uncond = running_uncond + out[label + "s"].astype(float).fillna(0)
        out["Next%dGWsStart" % (offset + 1)] = running_cond.round(2)
        out["Next%dGWs" % (offset + 1)] = running_uncond.round(2)

    return out.reset_index(drop=True)


def attach_odds_columns(df: pd.DataFrame, goal_rows: List[dict],
                        cs_rows: List[dict], gw: Optional[int]) -> pd.DataFrame:
    """Add CS / AnytimeGoal / AnytimeAssist / AnytimeReturn, as the sheet did.

    Joined on ``player_code`` and ``team_code`` — no name matching on either side.
    Probabilities arrive as 0-1 fractions and are stored as percentages, matching
    the sheet's scale so consumers and validation thresholds are unchanged.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["AnytimeGoal"] = float("nan")
    out["AnytimeAssist"] = float("nan")
    out["AnytimeReturn"] = float("nan")
    out["CS"] = float("nan")

    if goal_rows and "Player_Code" in out.columns:
        goals = pd.DataFrame(goal_rows)
        if "player_code" in goals.columns:
            goals = goals.set_index(pd.to_numeric(goals["player_code"], errors="coerce"))
            for src, dst in (("goal_prob", "AnytimeGoal"),
                             ("assist_prob", "AnytimeAssist"),
                             ("return_prob", "AnytimeReturn")):
                if src in goals.columns:
                    mapped = out["Player_Code"].map(
                        pd.to_numeric(goals[src], errors="coerce"))
                    out[dst] = (mapped.astype(float) * 100).round(1)

    if cs_rows and "Team" in out.columns:
        by_team = {}
        for row in cs_rows:
            name = row.get("team_name")
            for cell in row.get("cells") or []:
                if gw is None or cell.get("gw") == gw:
                    by_team[name] = _num(cell.get("cs_prob"))
                    break
        if by_team:
            out["CS"] = (out["Team"].map(by_team).astype(float) * 100).round(1)

    return out


# =============================================================================
# Which gameweek is a sheet-shaped frame for?
# =============================================================================

_FIXTURE_RE = re.compile(r"^\s*(.*?)\s*\(\s*([aAhH])\s*\)\s*$")

#: A minority of rows can be leftovers from an earlier gameweek — the live sheet
#: carried two clubs still on GW1 fixtures while the rest had moved to GW2 — so
#: this is a vote, not a unanimity test.
_GW_VOTE_THRESHOLD = 0.60


def _fpl_fixture_pairs(timeout: int = DEFAULT_TIMEOUT) -> Dict[int, set]:
    """``{gameweek: {(home_short, away_short), ...}}`` from the FPL fixture list."""
    boot = _get(BOOTSTRAP_URL, timeout=timeout)
    fixtures = _get(FIXTURES_URL, timeout=timeout)
    if boot is None or fixtures is None:
        return {}
    try:
        short = {t["id"]: t["short_name"] for t in boot.json().get("teams", [])}
        out: Dict[int, set] = {}
        for fixture in fixtures.json():
            event = fixture.get("event")
            if not event:
                continue
            home, away = short.get(fixture.get("team_h")), short.get(fixture.get("team_a"))
            if home and away:
                out.setdefault(int(event), set()).add((home, away))
        return out
    except Exception as exc:
        _logger.warning("FFP: could not build FPL fixture pairs: %s", exc)
        return {}


def resolve_ffp_gameweek(df: Optional[pd.DataFrame],
                         fixture_pairs: Optional[Dict[int, set]] = None) -> Optional[int]:
    """Work out which gameweek a sheet-shaped FFP frame describes.

    The site payload states its gameweek outright; this is for the Google Sheet
    fallback, which does not. It reconstructs ``(home, away)`` club pairs from
    ``Team`` + ``Fixture`` and votes them against the real fixture list.

    Club labels are resolved through ``TEAM_FULL_TO_SHORT`` rather than compared
    as raw strings — FFP writes "Notts Forest" where the bootstrap writes
    "Nott'm Forest", and an unmapped label is not cosmetic here.

    Returns None when no gameweek carries a clear majority, which is the honest
    answer: better to report "unknown" than to assert a wrong week.

    The check this replaces compared *sets of team names* and so scored 18/19
    for GW2, GW3 and GW4 alike — all 20 clubs play every gameweek, so team
    identity cannot separate adjacent weeks. Ordered pairs can: on the live
    sheet this scores GW2 at 0.83 and GW3 at exactly 0.0.
    """
    if df is None or getattr(df, "empty", True):
        return None
    if "FFP_GW" in df.columns:
        stated = pd.to_numeric(df["FFP_GW"], errors="coerce").dropna()
        if not stated.empty:
            return int(stated.mode().iloc[0])
    if "Team" not in df.columns or "Fixture" not in df.columns:
        return None

    pairs = set()
    for team, fixture in df[["Team", "Fixture"]].dropna().drop_duplicates().itertuples(index=False):
        match = _FIXTURE_RE.match(str(fixture))
        if not match:
            continue
        opponent, venue = match.group(1).strip(), match.group(2).lower()
        team_code = TEAM_FULL_TO_SHORT.get(str(team).strip())
        opp_code = TEAM_FULL_TO_SHORT.get(opponent)
        if not team_code or not opp_code:
            _logger.info("FFP: unmapped club label in fixture %r / %r", team, fixture)
            continue
        pairs.add((team_code, opp_code) if venue == "h" else (opp_code, team_code))

    if not pairs:
        return None

    if fixture_pairs is None:
        fixture_pairs = _fpl_fixture_pairs()
    if not fixture_pairs:
        return None

    scored = sorted(
        ((len(pairs & known) / len(pairs), gw) for gw, known in fixture_pairs.items()),
        reverse=True,
    )
    if not scored or scored[0][0] < _GW_VOTE_THRESHOLD:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None                     # ambiguous resolves to no answer
    return int(scored[0][1])


# =============================================================================
# Sheet fallback
# =============================================================================

_SHEET_PCT_COLS = ("Ownership", "Start", "LongStart", "CS",
                   "AnytimeAssist", "AnytimeGoal", "AnytimeReturn")
_SHEET_NUMERIC_COLS = (
    ["Predicted", "StartingPredicted"]
    + ["GW%d" % n for n in range(2, 7)] + ["GW%ds" % n for n in range(2, 7)]
    + ["Next%dGWs" % n for n in range(2, 7)]
    + ["Next%dGWsStart" % n for n in range(2, 7)]
)


def fetch_sheet() -> Optional[pd.DataFrame]:
    """The published Google Sheet, cleaned into the schema the app has always used.

    Kept as a fallback only: FFP no longer keeps it in step with their site.
    """
    from io import StringIO

    resp = _get(FFP_SHEET_URL)
    if resp is None:
        return None
    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception as exc:
        _logger.warning("FFP: could not parse sheet CSV: %s", exc)
        return None
    if df.empty:
        _logger.warning("FFP: sheet returned an empty table")
        return None

    for col in _SHEET_PCT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce")
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(
            df["Price"].astype(str).str.replace("£", "", regex=False)
                                  .str.replace("m", "", regex=False).str.strip(),
            errors="coerce")
    for col in _SHEET_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bootstrap_code_to_id(timeout: int = DEFAULT_TIMEOUT) -> Dict[int, int]:
    """``{FPL player code: FPL element id}``.

    FFP publishes the stable ``code``; every frame in this app keys on the
    per-season element ``id``.

    Memoised: both ``fetch_ffp_table`` and ``fetch_ffp_horizon`` need this map,
    and each used to download the whole 667-player bootstrap for it.
    """
    return _memoised("bootstrap_code_to_id",
                     lambda: _bootstrap_code_to_id_uncached(timeout))


def _bootstrap_code_to_id_uncached(timeout: int = DEFAULT_TIMEOUT) -> Dict[int, int]:
    resp = _get(BOOTSTRAP_URL, timeout=timeout)
    if resp is None:
        return {}
    try:
        return {int(e["code"]): int(e["id"]) for e in resp.json().get("elements", [])}
    except Exception as exc:
        _logger.warning("FFP: could not build code->id map: %s", exc)
        return {}
