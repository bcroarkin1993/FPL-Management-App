"""The Waiver Wire's claim engine, separated from the page that renders it.

Everything here computes; nothing here renders. The split exists because
`show_waiver_wire_page()` was the only place in the app that could answer "is
there a claim worth making", and a second surface now needs that answer -- the
Draft My Week strip on Home. Two implementations of one claim engine is the
twin-drift failure this codebase keeps paying for, and this module is the
alternative to it.

**It was already a de-facto shared module.** `scripts/draft/trade_analyzer.py`
imports six of these helpers and `scripts/classic/transfers.py` imports five
more, both reaching into a Streamlit page to do it. And
`scripts/test_scoring.py` hand-rolled the whole pipeline behind its own
Streamlit stub, where it drifted six ways -- no `_attach_minutes_dependency`,
no `AvgFDRNextN` on the reference pool, `element_states` omitted from
`_available_from_projections`, so every locked player scored as claimable.
That file tunes ROS weights.

**`import streamlit` here is for `@st.cache_data` and nothing else.** Six of
these functions are memoised and the decorator is how this codebase memoises.
What a module like this must never do is *render* or *halt*: `st.stop()` is an
instruction to Streamlit's script runner, so a helper that calls it can end a
page it has never heard of -- which `_available_from_projections` could do,
from inside a function whose job is to filter a DataFrame. Those paths now
return a status and a note, the shape `get_ffp_feed` and
`fetch_odds_index_with_status` already use, and the *page* decides what to show
and whether to stop. `tests/draft/test_waiver_core_is_renderless.py` pins it.
"""

# waiver_wire.py
import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, replace
from datetime import datetime

import config
from openai import OpenAI
from scripts.common.injury_helpers import (
    ABSENCE_FROM_NEWS,
    OUT_OF_SQUAD_STATUSES,
    absence_estimate,
    departed_mask,
    estimate_games_to_miss,
)
from scripts.common.projection_engine import DEPUTY_MAX_START_RATE
from scripts.common import return_watch as return_watch_module
from scripts.common.scraping import get_prior_season_rates
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.draft.waiver_wire")


# ---------------------------------------------------------------------------
# How this module reports trouble, now that it may not render it
# ---------------------------------------------------------------------------
#: Terminal statuses. The engine cannot produce a board; the *page* decides what
#: to show and whether to stop, and the My Week strip turns the same status into
#: an `unknown` row rather than a reassuring one.
BOARD_OK = "ok"
BOARD_NO_PROJECTIONS = "no_projections"
BOARD_BAD_SCHEMA = "bad_schema"
BOARD_POOL_FAILED = "pool_failed"


class WaiverDataUnavailable(Exception):
    """No board can be built, and the prose explaining why.

    Raised rather than rendered because `st.stop()` is an instruction to
    Streamlit's script runner: called from in here it ends a page this module
    has never heard of. `_available_from_projections` -- a function whose job
    is to filter a DataFrame -- could do precisely that.
    """

    def __init__(self, message: str, status: str = BOARD_NO_PROJECTIONS):
        super().__init__(message)
        self.message = message
        self.status = status


def _note(notes, message: str, severity: str = "warning") -> None:
    """Record something the reader should see, for the caller to render.

    An explicit sink rather than a module-level list: the board is cached, and
    global mutable state behind a cache is how a warning comes to be shown on
    the cold call and silently dropped on every warm one.
    """
    _logger.warning(message)
    if notes is not None:
        notes.append((severity, message))


from scripts.common.utils import (
    get_current_gameweek,
    merge_fpl_players_and_projections,
    get_league_player_ownership,
    get_league_entries,
    get_fpl_player_mapping,
    get_rotowire_player_projections,
    pull_fpl_player_stats,
    normalize_fpl_players_to_rotowire_schema,
    normalize_rotowire_players,
    _bootstrap_teams_df,
    _norm_text,
    TEAM_FULL_TO_SHORT
)
from scripts.common.fpl_draft_api import (
    get_league_element_states,
    get_draft_transaction_window,
    get_league_waiver_order,
    ELEMENT_STATE_LOCKED,
    ELEMENT_STATE_AVAILABLE,
    ELEMENT_STATE_OWNED,
    TRANSACTION_MODE_FREE_AGENCY,
)
from scripts.common.claim_value import (
    HORIZON_GWS,
    MIN_FLOOR_POOL,
    LANE_HOLD,
    LANE_RENT,
    recommended_lane,
    release_cost as _release_cost,
    rental_gain,
    waiver_floors,
)
from scripts.common.waiver_priority import (
    BAND_CONTESTED,
    BAND_LIKELY,
    BAND_LONG_SHOT,
    claim_outlook,
    expected_gone_before,
    expected_taken_at_position,
    managers_ahead,
    my_waiver_pick,
    rank_claim_plan,
)
from scripts.common.display_names import (
    attach_display_names,
    display_of,
    display_series,
)
from scripts.common.player_matching import canonical_normalize
from scripts.common.text_helpers import (compact_html, format_deadline_stamp,
                                         format_time_until)
from scripts.common.styled_tables import render_styled_table
from scripts.common.transfer_sanity import (horizon_column, horizon_points,
                                             sanity_check_suggestion)
from scripts.common.analytics import (
    gameweek_deadlines,
    compute_player_scores,
    compute_dynamic_alpha,
    compute_healthy_form,
    compute_positional_depth,
    compute_transfer_urgency,
    blend_multi_gw_projections,
    positional_rank,
    dampen_form_by_starts,
    merge_season_projections,
    merge_ffp_single_gw_data,
    enrich_reference_with_projections,
    numeric_col,
    attach_reference_ids,
)
from scripts.common.scraping import get_ffp_feed, get_rotowire_season_rankings, render_ffp_status

POSITION_LABEL_TO_CODE = {"GK": "G", "DEF": "D", "MID": "M", "FWD": "F"}

POSITION_CODE_TO_LABEL = {c: l for l, c in POSITION_LABEL_TO_CODE.items()}

#: Smallest rental gain worth listing, in points over the window. A **display**
#: floor, not a significance claim: the projections this is built from carry an
#: MAE around 2.5 points per player, so a 0.6-point edge is not a fact. The card
#: prints the number and the release cost so the call stays the manager's -- but
#: without some floor the lane lists every marginal swap on the board.
RENTAL_MIN_GAIN = 0.5

@st.cache_data(show_spinner=False)
def _load_bootstrap() -> Dict[str, Any]:
    """bootstrap-static with players & teams (cached).

    The **Draft** bootstrap, and only its `teams` are read -- team ids and short
    names are identical in both APIs, so that is safe. Do not take `elements`
    from here: the two APIs number 64 of 669 players differently, and this
    page's frames are Classic-keyed. See `scripts/common/element_ids.py`.
    """
    url = "https://draft.premierleague.com/api/bootstrap-static"
    return requests.get(url, timeout=30).json()

def _teams_map_short() -> Dict[int, str]:
    """Map FPL team id -> short_name (e.g., 13 -> 'MCI')."""
    data = _load_bootstrap()
    return {t['id']: t['short_name'] for t in data.get('teams', [])}

def _team_id_by_short(short_name: str) -> Optional[int]:
    """Reverse lookup: short_name -> team id."""
    m = {v: k for k, v in _teams_map_short().items()}
    return m.get(short_name)

# Display names live in `scripts/common/display_names.py`. This page had its own
# bootstrap->display map, its own `Display_Name or Player` fallback and its own
# vectorised wrapper; two other pages had the same fallback inline and two more
# re-derived first/second names by splitting a legal name on whitespace. One
# implementation now, resolving by player code first -- which is the only id
# both FPL APIs agree on, so a frame cannot be labelled by an id that crossed
# API spaces. That is the bug that gave a tenth of the pool another player's
# name here.
#
# These three names are kept as thin aliases because this module's call sites
# and tests are written against them.
_attach_display_names = attach_display_names

_display_of = display_of

_display_names = display_series

@st.cache_data(show_spinner=False)
def _load_future_fixtures() -> pd.DataFrame:
    """
    Returns future fixtures with difficulties.
    Columns: event, kickoff_time, team_h, team_a, team_h_difficulty, team_a_difficulty
    """
    url = "https://fantasy.premierleague.com/api/fixtures/?future=1"
    fx = requests.get(url, timeout=30).json()
    df = pd.DataFrame(fx)
    keep = ["event", "kickoff_time", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df

@st.cache_data(show_spinner=False)
def _element_history_df(player_id: int) -> Optional[pd.DataFrame]:
    """Return per-GW history for a player (None on failure)."""
    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        js = requests.get(url, timeout=30).json()
        hist = pd.DataFrame(js.get("history", []))
        return hist if not hist.empty else None
    except Exception:
        return None

_POS_NUM_TO_LETTER = {1: "G", 2: "D", 3: "M", 4: "F"}

_POS_WORD_TO_LETTER = {
    "GK": "G", "GKP": "G", "Goalkeeper": "G",
    "DEF": "D", "Defender": "D",
    "MID": "M", "Midfielder": "M",
    "FWD": "F", "FW": "F", "Forward": "F"
}

def _to_position_letter(val) -> Optional[str]:
    """Map whatever we get to G/D/M/F."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        return _POS_NUM_TO_LETTER.get(int(val))
    v = str(val).strip()
    if v in {"G", "D", "M", "F"}:
        return v
    return _POS_WORD_TO_LETTER.get(v)

def _series_to_short_team(s: pd.Series, teams_df: pd.DataFrame) -> pd.Series:
    """
    Convert a Series of team identifiers to short codes (e.g., 'MCI').
    Handles numeric ids or strings; falls back gracefully.
    """
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s):
        id2short = dict(zip(teams_df["id"], teams_df["short_name"]))
        return s.map(id2short)
    s = s.astype(str)
    mask_short = s.str.len() == 3
    out = s.where(mask_short)
    # Use TEAM_FULL_TO_SHORT from utils.py for team name mapping
    if TEAM_FULL_TO_SHORT:
        out = out.fillna(s.map(TEAM_FULL_TO_SHORT))
    return out.fillna(s)

def _enforce_rw_schema_fpl(fpl_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure FPL stats DF has: Player, Team (short), Position (G/D/M/F), Player_ID, Season_Points.
    """
    df = fpl_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Player_ID", "Season_Points"])

    idx = {c.lower(): c for c in df.columns}

    # --- Player ---
    if "player" in idx:
        df["Player"] = df[idx["player"]]
    elif "player_name" in idx:
        df["Player"] = df[idx["player_name"]]
    elif "web_name" in idx:
        df["Player"] = df[idx["web_name"]]
    elif "first_name" in idx and "second_name" in idx:
        df["Player"] = df[idx["first_name"]].astype(str) + " " + df[idx["second_name"]].astype(str)
    elif "name" in idx:
        df["Player"] = df[idx["name"]]
    else:
        df["Player"] = df.index.astype(str)

    # --- Team (short) ---
    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    elif "team_name" in idx:
        df["Team"] = _series_to_short_team(df[idx["team_name"]], teams_df)
    else:
        df["Team"] = np.nan

    # --- Position (G/D/M/F) ---
    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "position_abbrv" in idx:  # <— NEW: FPL stats has 'FWD/MID/DEF/GK'
        df["Position"] = df[idx["position_abbrv"]].map(_to_position_letter)
    elif "element_type" in idx:
        df["Position"] = df[idx["element_type"]].map(_to_position_letter)
    elif "pos" in idx:
        df["Position"] = df[idx["pos"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    # --- Player_ID ---
    if "player_id" in idx:
        df["Player_ID"] = pd.to_numeric(df[idx["player_id"]], errors="coerce")
    elif "id" in idx:
        df["Player_ID"] = pd.to_numeric(df[idx["id"]], errors="coerce")
    else:
        df["Player_ID"] = np.nan

    # --- Season_Points ---
    if "season_points" in idx:
        df["Season_Points"] = pd.to_numeric(df[idx["season_points"]], errors="coerce")
    elif "total_points" in idx:
        df["Season_Points"] = pd.to_numeric(df[idx["total_points"]], errors="coerce")
    else:
        df["Season_Points"] = 0

    df["Team"] = df["Team"].astype(str)
    df["Position"] = df["Position"].astype(str)

    keep = ["Player", "Team", "Position", "Player_ID", "Season_Points"]
    extras = [c for c in df.columns if c not in keep]
    return df[keep + extras]

def _enforce_rw_schema_proj(proj_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure projections DF has Player, Team (short), Position (G/D/M/F), Points.
    """
    df = proj_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])

    idx = {c.lower(): c for c in df.columns}

    # Player
    df["Player"] = df[idx["player"]] if "player" in idx else (
        df[idx["name"]] if "name" in idx else df.index.astype(str)
    )

    # Team
    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    else:
        df["Team"] = np.nan

    # Position
    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "pos" in idx:
        df["Position"] = df[idx["pos"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    # Points
    df["Points"] = pd.to_numeric(df[idx["points"]], errors="coerce") if "points" in idx else np.nan

    return df

def _avg_fdr_for_team(short_name: str, current_gw: int, n_weeks: int) -> Optional[float]:
    """Average FDR over next n_weeks for a team (short_name)."""
    team_id = _team_id_by_short(short_name)
    if team_id is None:
        return None

    fixtures = _load_future_fixtures()
    fixtures = fixtures.dropna(subset=["event"])
    fixtures["event"] = fixtures["event"].astype(int)

    future_slice = fixtures[(fixtures["event"] >= current_gw) &
                            (fixtures["event"] < current_gw + n_weeks)].copy()
    if future_slice.empty:
        return None

    def _row_team_fdr(row):
        if row.get("team_h") == team_id:
            return row.get("team_h_difficulty")
        if row.get("team_a") == team_id:
            return row.get("team_a_difficulty")
        return None

    future_slice["team_fdr"] = future_slice.apply(_row_team_fdr, axis=1)
    d = pd.to_numeric(future_slice["team_fdr"], errors="coerce").dropna()
    return float(d.mean()) if not d.empty else None

def _avg_fdr_by_short_name(current_gw: int, n_weeks: int) -> pd.Series:
    """Average FDR over the next ``n_weeks``, one value per club, by short code.

    **FDR is a property of the club.** ``_avg_fdr_for_team`` was applied per row
    -- each call a dropna, an astype, a filter and a per-row apply over the whole
    fixture table -- to produce twenty distinct answers for ~700 players. This is
    the same optimisation Classic already made (``transfers._avg_fdr_by_team``:
    "Twenty clubs, not seven hundred players").

    The reason it exists rather than being a tidy-up, though, is that the
    **reference pool needs this column and never had it**. ``AvgFDRNextN`` was
    written onto the scored frames and not onto ``fpl_stats``, so the ROS fixture
    term silently min-maxed each frame against itself -- see
    ``analytics._referenced_percentile``. One cheap club-indexed Series can be
    mapped onto both, which is what makes the fix a ``.map()`` at two callsites
    rather than 700 more fixture scans.

    Built on the same ``_load_future_fixtures()`` window as before, so the values
    are identical to what the scored frame already received. ``None``/missing
    stays NaN so ``numeric_col(..., 3)`` downstream still defaults correctly.
    """
    out: Dict[str, float] = {}
    try:
        fixtures = _load_future_fixtures()
    except Exception:
        _logger.warning("Unable to load fixtures for club FDR", exc_info=True)
        return pd.Series(dtype="float64")
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return pd.Series(dtype="float64")

    fixtures = fixtures.dropna(subset=["event"]).copy()
    fixtures["event"] = fixtures["event"].astype(int)
    window = fixtures[(fixtures["event"] >= current_gw) &
                      (fixtures["event"] < current_gw + n_weeks)]
    if window.empty:
        return pd.Series(dtype="float64")

    teams = _bootstrap_teams_df()
    if teams is None or teams.empty:
        return pd.Series(dtype="float64")

    for _, trow in teams.iterrows():
        tid = trow.get("id")
        short = trow.get("short_name")
        if tid is None or not short:
            continue
        home = window.loc[window["team_h"] == tid, "team_h_difficulty"]
        away = window.loc[window["team_a"] == tid, "team_a_difficulty"]
        vals = pd.to_numeric(pd.concat([home, away]), errors="coerce").dropna()
        if not vals.empty:
            out[str(short)] = float(vals.mean())

    return pd.Series(out, dtype="float64")

def _min_max_norm(series: pd.Series) -> pd.Series:
    """Min-max to [0,1], safe for constants/NaN."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)

def _flatten_owned_names(league_ownership: Dict[int, Dict[str, Any]]) -> List[str]:
    """
    Supports structures like:
      { team_id: { 'team_name': str, 'players': {'G':[...],'D':[...],'M':[...],'F':[...]} } }
    or { team_id: { 'team_name': str, 'players': [<names>...] } }
    or even { team_id: { 'G':[...],'D':[...],'M':[...],'F':[...]} }  (older variants)
    Falls back to scanning any nested iterables of strings.
    """
    out: List[str] = []

    def _extend_from(obj):
        if obj is None:
            return
        # dict of positions -> list of names
        if isinstance(obj, dict):
            for v in obj.values():
                _extend_from(v)
        # iterable of names
        elif isinstance(obj, (list, set, tuple)):
            for x in obj:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, (list, set, tuple, dict)):
                    _extend_from(x)
        # single string
        elif isinstance(obj, str):
            out.append(obj)

    for _, team_blob in league_ownership.items():
        if not isinstance(team_blob, dict):
            continue
        # primary path
        if "players" in team_blob:
            _extend_from(team_blob["players"])
            continue
        # fallback: maybe positions live at top level
        pos_like = {k: v for k, v in team_blob.items() if k in {"G", "D", "M", "F"}}
        if pos_like:
            _extend_from(pos_like)
            continue
        # absolute fallback: scan everything in blob
        _extend_from(team_blob)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

#: Identity is preserved through every schema coercion on this page. Rotowire
#: publishes names only, so these are resolved once at the fetch boundary
#: (`scraping._attach_projection_ids`); a coercion that drops them sends the
#: whole page back to name matching, silently.
_IDENTITY_COLS = ("Player_ID", "Player_Code")

def _prepare_proj_for_merge(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce projections to the exact schema the fuzzy merge expects:
    ['Player','Team','Matchup','Position','Points','Pos Rank'], plus identity.

    - Renames common variants
    - Creates missing columns with safe defaults
    - Returns only the expected columns (and correct dtypes)
    """
    df = df_in.copy()

    # Rename common variants -> canonical
    # Track which canonical names we've already mapped to avoid duplicates
    rename_map = {}
    mapped_targets = set()

    for c in df.columns:
        lc = c.strip().lower()
        target = None
        if lc in ("player", "name", "player_name"):
            target = "Player"
        elif lc in ("team", "team_short", "teamname", "team_name"):
            target = "Team"
        elif lc in ("matchup", "fixture", "opp", "opponent"):
            target = "Matchup"
        elif lc in ("position", "pos"):
            target = "Position"
        elif lc in ("points", "point", "proj", "projection"):
            target = "Points"
        elif lc in ("pos rank", "pos_rank", "position rank", "position_rank"):
            target = "Pos Rank"

        # Only add to rename map if we haven't already mapped something to this target
        if target and target not in mapped_targets:
            rename_map[c] = target
            mapped_targets.add(target)

    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop any remaining duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Ensure required columns exist
    required = ["Player", "Team", "Matchup", "Position", "Points", "Pos Rank"]
    defaults = {"Player": None, "Team": None, "Matchup": "", "Position": None, "Points": np.nan, "Pos Rank": "NA"}
    for k in required:
        if k not in df.columns:
            df[k] = defaults[k]

    # Dtypes
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

    # The columns the merge helper uses, plus identity. Returning `required`
    # alone discarded `Player_ID`/`Player_Code` -- which `scraping` now resolves
    # once at the boundary -- so every consumer downstream was back to matching
    # on names. Same mistake as the merge helper's own, one layer up.
    keep = required + [c for c in _IDENTITY_COLS if c in df.columns]
    return df[keep]

def _codes_in_state(element_states, status) -> set:
    """Player codes of every element in ``status``, via the Draft player map.

    `element-status` is the league's own ground truth, keyed on the *Draft*
    element id, and `get_fpl_player_mapping` carries the `Player_Code` for each
    -- so this crosses from the Draft space to the app's join key without any
    name matching and without the element-id translation step. See
    `scripts/common/element_ids.py`.
    """
    if not element_states:
        return set()
    try:
        player_map = get_fpl_player_mapping()
    except Exception:
        _logger.warning("Could not read the Draft player map for %r state", status,
                        exc_info=True)
        return set()
    out = set()
    for element_id, state in (element_states or {}).items():
        if (state or {}).get("status") != status:
            continue
        code = (player_map.get(element_id) or {}).get("Player_Code")
        if code is not None and not pd.isna(code):
            out.add(int(code))
    return out

def _names_for_codes(proj_for_merge: pd.DataFrame, codes: set) -> set:
    """The projection table's own names for a set of player codes."""
    if not codes or "Player_Code" not in proj_for_merge.columns:
        return set()
    pc = pd.to_numeric(proj_for_merge["Player_Code"], errors="coerce")
    hit = proj_for_merge.loc[pc.isin(codes), "Player"]
    return set(hit.dropna().unique().tolist())

def _locked_rotowire_names(
    element_states: Optional[Dict[int, Dict[str, Any]]],
    proj_for_merge: pd.DataFrame,
) -> set:
    """Rotowire canonical names of players currently in the Draft *locked* state.

    A locked player is on nobody's roster but cannot be acquired — most often
    because another manager just dropped them. They only become claimable at the
    next waiver deadline, so suggesting one as a free-agency pickup is advice the
    manager cannot act on.

    Locked element IDs are mapped onto Rotowire names through the *same*
    merge_fpl_players_and_projections() path the ownership anti-join below uses,
    so locked-matching is exactly as accurate as the owned-matching this page
    already depends on. Anything that fails to match is simply not returned:
    leaving a genuine free agent flagged as available is a far cheaper error than
    hiding one.
    """
    if not element_states:
        return set()

    player_map = get_fpl_player_mapping()
    rows = []
    for element_id, state in element_states.items():
        if state.get("status") != ELEMENT_STATE_LOCKED:
            continue
        info = player_map.get(element_id)
        if not info:
            continue
        rows.append({
            "Player": info.get("Player"),
            "Team": info.get("Team"),
            "Position": info.get("Position"),
        })

    if not rows:
        return set()

    locked_fpl = pd.DataFrame(rows)
    locked_fpl["Team"] = locked_fpl["Team"].replace(TEAM_FULL_TO_SHORT)

    try:
        mapped = merge_fpl_players_and_projections(locked_fpl, proj_for_merge)
    except Exception:
        # Never take the page down over a badge. Unknown state -> treat as available.
        _logger.warning("Failed to map locked players onto projections", exc_info=True)
        return set()

    by_name = set(mapped["Player"].dropna().unique().tolist())

    # Union with the id path, never a replacement. The name path can only ever
    # *fail to flag* a locked player (its unmatched branch returns the FPL name,
    # which matches no Rotowire row and so does nothing), so adding the code
    # resolution can only flag more -- it cannot un-flag anyone the name path
    # got right. Measured live at GW9 the two agree exactly, on the same 4 of
    # 14 locked players who appear in the Rotowire table at all.
    return by_name | _names_for_codes(
        proj_for_merge, _codes_in_state(element_states, ELEMENT_STATE_LOCKED))

def _available_from_projections(
    projections_df: pd.DataFrame,
    fpl_player_statistics_df: pd.DataFrame,
    league_ownership: Dict[int, Dict[str, Any]],
    element_states: Optional[Dict[int, Dict[str, Any]]] = None,
    notes=None,
) -> pd.DataFrame:
    """
    Anti-join to remove league-owned players using your fuzzy merge helper.
    If ownership is empty/unavailable (preseason, API issues), return normalized projections unchanged.

    Adds a `Draft_State` column ('a' available / 'l' locked). Owned players are
    dropped outright — they are not waiver targets at all — whereas locked players
    are kept and flagged, so the manager can still see who is coming to the next
    waiver round. `_compute_transfer_suggestions()` excludes the locked ones.
    """
    if projections_df is None or projections_df.empty:
        # Genuinely nothing to rank players by (Rotowire hasn't published a usable
        # rankings table yet — common early preseason). Distinct from the
        # ownership-empty case below: there, we can still show *something*
        # (everyone available); here there's no projection data to show at all,
        # so stop cleanly with an honest message instead of crashing downstream.
        raise WaiverDataUnavailable(
            "📊 Player projections aren't available yet — Rotowire hasn't published a "
            "usable rankings table for this gameweek (common in the preseason). Waiver "
            "suggestions need projected points to rank players; check back closer to "
            "the gameweek deadline."
        )

    # Normalize projections to the exact schema the fuzzy merge expects
    proj_for_merge = _prepare_proj_for_merge(projections_df)

    # Locked players are unclaimable right now regardless of ownership, so resolve
    # them before the ownership branch below.
    locked_names = _locked_rotowire_names(element_states, proj_for_merge)

    def _stamp_state(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Draft_State"] = np.where(
            df["Player"].isin(locked_names), ELEMENT_STATE_LOCKED, ELEMENT_STATE_AVAILABLE
        )
        return df

    # 1) Try to build owned names
    owned_names = _flatten_owned_names(league_ownership)
    if len(owned_names) == 0:
        # Nothing owned -> everyone available
        _note(notes, "⚠️ No ownership data detected (preseason or API). "
                     "Treating all players as available.", "info")
        return _stamp_state(proj_for_merge)

    # 2) Enrich owned with Team/Position (for better fuzzy accuracy)
    # Use normalized names for the merge to handle accent mismatches
    fpl_cols = {c.lower(): c for c in fpl_player_statistics_df.columns}
    need = {"player", "team", "position"}
    miss = need - set(fpl_cols.keys())
    if miss:
        raise ValueError(
            "fpl_player_statistics_df must contain ['Player','Team','Position'] after normalization. "
            f"Missing (case-insensitive): {sorted(miss)}. "
            f"Got: {list(fpl_player_statistics_df.columns)}"
        )

    # Prepare FPL stats with normalized names for matching
    fpl_stats_for_merge = fpl_player_statistics_df[
        [fpl_cols["player"], fpl_cols["team"], fpl_cols["position"]]
    ].rename(columns={
        fpl_cols["player"]:   "Player",
        fpl_cols["team"]:     "Team",
        fpl_cols["position"]: "Position"
    }).copy()
    fpl_stats_for_merge["__norm_name"] = fpl_stats_for_merge["Player"].apply(canonical_normalize)

    # Convert full team names to short codes to match Rotowire format
    # E.g., "Fulham" -> "FUL", "Man City" -> "MCI"
    fpl_stats_for_merge["Team"] = fpl_stats_for_merge["Team"].replace(TEAM_FULL_TO_SHORT)

    # Add normalized name to owned names for the merge
    owned_fpl = pd.DataFrame({"Player": owned_names})
    owned_fpl["__norm_name"] = owned_fpl["Player"].apply(canonical_normalize)

    # Merge on normalized name instead of raw name to handle accents
    owned_fpl = owned_fpl.merge(
        fpl_stats_for_merge[["__norm_name", "Team", "Position"]],
        on="__norm_name", how="left"
    )
    # Drop the helper column
    owned_fpl = owned_fpl.drop(columns=["__norm_name"], errors="ignore")

    # 3) Fuzzy map FPL -> RotoWire canonical names
    try:
        mapped_owned = merge_fpl_players_and_projections(owned_fpl, proj_for_merge)
    except Exception as e:
        raise RuntimeError(
            "merge_fpl_players_and_projections failed during ownership mapping. "
            f"owned_fpl cols={list(owned_fpl.columns)} projections_df cols={list(proj_for_merge.columns)} "
            f"Error: {e}"
        )

    # 4) Anti-join, by name *and* by player code.
    #
    # The name leg is a three-step chain -- league_ownership publishes a name,
    # which is matched to FPL stats for Team/Position, which is then fuzzy
    # matched to Rotowire -- and a miss anywhere leaves an owned player in the
    # available pool, where he is offered as a waiver target. Measured live at
    # GW9 it lost two of 125: FPL spells one "Oli McBurnie" against Rotowire's
    # "Oliver McBurnie", and the other "Đorđe Petrović" against "Djordje
    # Petrovic". McBurnie was on *this* manager's own roster and projected 5.13
    # points, so he ranked as a target worth claiming.
    #
    # `element-status` knows ownership by element id and needs no names at all.
    # It is unioned in rather than replacing the name leg, so a code that fails
    # to resolve cannot re-admit a player the names correctly removed.
    owned_rw_names = set(mapped_owned["Player"].dropna().unique().tolist())
    owned_rw_names |= _names_for_codes(
        proj_for_merge, _codes_in_state(element_states, ELEMENT_STATE_OWNED))
    avail = proj_for_merge[~proj_for_merge["Player"].isin(owned_rw_names)].copy()
    return _stamp_state(avail)

def _ensure_team_col(df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a single 'Team' column exists on df.
    - Prefer existing 'Team' if present.
    - If only Team_x/Team_y exist, coalesce.
    - Else, pull 'Team' from fpl_stats by ['Player','Position'].
    """
    base = df.copy()

    # If already present, try to coalesce with any *_y leftovers
    if "Team" in base.columns:
        if "Team_y" in base.columns:
            base["Team"] = base["Team"].fillna(base["Team_y"])
        if "Team_x" in base.columns:
            base["Team"] = base["Team"].fillna(base["Team_x"])
    else:
        # No 'Team' column — try to derive
        src = None
        if "Team_x" in base.columns or "Team_y" in base.columns:
            base["Team"] = base.get("Team_x")
            base["Team"] = base["Team"].fillna(base.get("Team_y"))
        else:
            # Join from fpl_stats by Player+Position
            try:
                add = (fpl_stats[["Player", "Position", "Team"]]
                       .drop_duplicates())
                base = base.merge(add, on=["Player", "Position"], how="left", suffixes=("", "_from_stats"))
                if "Team_from_stats" in base.columns and "Team" not in base.columns:
                    base.rename(columns={"Team_from_stats":"Team"}, inplace=True)
            except Exception:
                base["Team"] = np.nan

    # Clean up any leftovers
    for col in ("Team_x", "Team_y", "Team_from_stats"):
        if col in base.columns:
            base.drop(columns=[col], inplace=True)

    # Ensure it's stringy (FDR call expects a short code string)
    base["Team"] = base["Team"].astype(str)
    return base

def _align_roster_player_names_to_projections(
    roster_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    notes=None,
) -> pd.DataFrame:
    """
    Replace FPL-style names on the roster with the RotoWire canonical names,
    preserving row order, using your merge_fpl_players_and_projections().
    """
    try:
        # prepare inputs for the fuzzy function
        for_merge = roster_df[["Player", "Team", "Position"]].copy()
        norm_proj = _prepare_proj_for_merge(projections_df)  # already in this file

        mapped = merge_fpl_players_and_projections(for_merge, norm_proj)
        # mapped['Player'] is the canonical (RotoWire) name in the same order as input
        canon = mapped["Player"].reset_index(drop=True)

        out = roster_df.copy()
        out["Player"] = canon
        return out
    except Exception as e:
        _note(notes, f"⚠️ Name alignment to projections failed; keeping FPL "
                     f"names. Reason: {e}")
        return roster_df

def _add_fdr_and_form(
    df: pd.DataFrame,
    fpl_player_statistics_df: pd.DataFrame,
    current_gw: int,
    weeks: int,
    form_weeks: int = None,
) -> pd.DataFrame:
    """
    Join AvgFDRNextN and Form onto df.

    - Requires df to have Player/Team/Position (will create if missing).
    - Uses normalized names for matching to handle accent differences.
    - Prefers Player_ID join when available, falls back to normalized name merge.
    - Fallback chain for Form: element-summary -> FPL 'form' -> 'points_per_game' -> 0.
    """
    base = df.copy()

    # Clean up any duplicate columns from prior merges (e.g., Team_x, Team_y)
    for col in ("Team", "Position", "Player"):
        if f"{col}_x" in base.columns or f"{col}_y" in base.columns:
            # Coalesce: prefer _x, then _y, then original
            if col in base.columns:
                base[col] = base[col].fillna(base.get(f"{col}_x", pd.NA))
                base[col] = base[col].fillna(base.get(f"{col}_y", pd.NA))
            elif f"{col}_x" in base.columns:
                base[col] = base[f"{col}_x"].fillna(base.get(f"{col}_y", pd.NA))
            elif f"{col}_y" in base.columns:
                base[col] = base[f"{col}_y"]
            base.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True, errors="ignore")

    # Ensure join keys exist
    for col in ("Player", "Team", "Position"):
        if col not in base.columns:
            base[col] = np.nan

    # Safely select merge cols from stats
    stats = fpl_player_statistics_df.copy()
    for col in ("Player", "Team", "Position", "Player_ID", "form", "points_per_game"):
        if col not in stats.columns:
            stats[col] = np.nan

    # Check if we can use Player_ID for the join (preferred, more reliable)
    has_player_id = "Player_ID" in base.columns and base["Player_ID"].notna().any()

    if has_player_id:
        # Use Player_ID join (most reliable)
        base = base.merge(
            stats[["Player_ID", "form", "points_per_game"]].drop_duplicates(subset=["Player_ID"]),
            on="Player_ID",
            how="left",
            suffixes=("", "_stats")
        )
        # Coalesce form columns if needed
        if "form_stats" in base.columns:
            base["form"] = base.get("form", pd.NA)
            base["form"] = base["form"].fillna(base["form_stats"])
            base.drop(columns=["form_stats"], inplace=True, errors="ignore")
        if "points_per_game_stats" in base.columns:
            base["points_per_game"] = base.get("points_per_game", pd.NA)
            base["points_per_game"] = base["points_per_game"].fillna(base["points_per_game_stats"])
            base.drop(columns=["points_per_game_stats"], inplace=True, errors="ignore")
    else:
        # Fallback: Use normalized name + team + position for the merge
        # Add normalized name columns for matching
        base["__norm_name"] = base["Player"].apply(canonical_normalize)
        stats["__norm_name"] = stats["Player"].apply(canonical_normalize)

        # Ensure both Team columns use short codes for matching
        base["Team"] = base["Team"].replace(TEAM_FULL_TO_SHORT)
        stats["Team"] = stats["Team"].replace(TEAM_FULL_TO_SHORT)

        # Prepare stats for merge
        stats_for_merge = stats[["__norm_name", "Team", "Position", "Player_ID", "form", "points_per_game"]].copy()
        # Drop duplicates to avoid row multiplication
        stats_for_merge = stats_for_merge.drop_duplicates(subset=["__norm_name", "Team", "Position"])

        # First try: merge on normalized name + team + position (most precise)
        base = base.merge(
            stats_for_merge,
            on=["__norm_name", "Team", "Position"],
            how="left",
            suffixes=("", "_stats")
        )

        # For rows that didn't match, try a less strict merge (just name + team)
        # Determine which rows didn't get form data from the first merge
        check_col = "form_stats" if "form_stats" in base.columns else "form"
        if check_col not in base.columns:
            base[check_col] = np.nan
        unmatched_mask = base[check_col].isna()

        if unmatched_mask.any():
            stats_name_team = stats[["__norm_name", "Team", "Player_ID", "form", "points_per_game"]].copy()
            stats_name_team = stats_name_team.drop_duplicates(subset=["__norm_name", "Team"])
            stats_name_team = stats_name_team.rename(columns={
                "Player_ID": "Player_ID_fb",
                "form": "form_fb",
                "points_per_game": "ppg_fb"
            })

            # Merge unmatched rows on name + team only, preserving index
            unmatched_idx = base.loc[unmatched_mask].index
            unmatched = base.loc[unmatched_mask, ["__norm_name", "Team"]].copy()
            unmatched = unmatched.reset_index(drop=False)  # Keep original index as column
            fallback = unmatched.merge(
                stats_name_team,
                on=["__norm_name", "Team"],
                how="left"
            )
            fallback = fallback.set_index("index")  # Restore original index

            # Fill in the missing values from fallback for matching rows
            for orig_col, fb_col in [("Player_ID", "Player_ID_fb"), ("form", "form_fb"), ("points_per_game", "ppg_fb")]:
                stats_col = f"{orig_col}_stats" if f"{orig_col}_stats" in base.columns else orig_col
                if fb_col in fallback.columns:
                    # Update only rows that got a match in fallback
                    matched_in_fallback = fallback[fb_col].notna()
                    if matched_in_fallback.any():
                        update_idx = fallback.loc[matched_in_fallback].index
                        if stats_col not in base.columns:
                            base[stats_col] = np.nan
                        base.loc[update_idx, stats_col] = fallback.loc[update_idx, fb_col]

        # Coalesce Player_ID if needed
        if "Player_ID_stats" in base.columns:
            base["Player_ID"] = base.get("Player_ID", pd.NA)
            if isinstance(base["Player_ID"], pd.Series):
                base["Player_ID"] = base["Player_ID"].fillna(base["Player_ID_stats"])
            base.drop(columns=["Player_ID_stats"], inplace=True, errors="ignore")

        # Coalesce form columns
        if "form_stats" in base.columns:
            base["form"] = base.get("form", pd.NA)
            base["form"] = base["form"].fillna(base["form_stats"])
            base.drop(columns=["form_stats"], inplace=True, errors="ignore")
        if "points_per_game_stats" in base.columns:
            base["points_per_game"] = base.get("points_per_game", pd.NA)
            base["points_per_game"] = base["points_per_game"].fillna(base["points_per_game_stats"])
            base.drop(columns=["points_per_game_stats"], inplace=True, errors="ignore")

        # Clean up helper column
        base.drop(columns=["__norm_name"], inplace=True, errors="ignore")

    # Ensure Player_ID exists (for downstream)
    if "Player_ID" not in base.columns:
        base["Player_ID"] = np.nan

    # --- Tier 3: the tiered matcher, for what exact equality cannot reach ----
    #
    # Both merges above are **exact** on the normalised name. Projection sources
    # publish the common name and the FPL bootstrap publishes the full legal
    # one, so every player whose legal name is longer than his common one falls
    # straight through -- and the miss is silent, because the caller just sees
    # NaN.
    #
    # Reported from the app as "James Hill recommended over António João
    # Silva". Measured on the live 101-row available pool, 12 players lost
    # their id here, including the 4th-best available defender:
    #
    #     James Hill      BOU  D  5.03 pts   Player_ID  60.0
    #     Antonio Silva   BOU  D  5.01 pts   Player_ID   NaN
    #
    # Same club, same position, projections two hundredths apart. Everything
    # downstream is keyed on this id -- `Season_Points` and `starts` merge on
    # it and then `fillna(0)`, injury status never lands, the projected-XI
    # override is skipped -- so `season_quality` and `start_consistency` read
    # zero, ROS collapses, Transfer Score collapses, and he drops out of the
    # candidate window entirely. He generates no debug row either, so it reads
    # exactly like "evaluated and rejected". Hill matched exactly, kept his
    # full ROS, and won.
    #
    # Tier 4 of `ReferenceMatcher` is a token subset in either direction scoped
    # to team *and* position, which is precisely this shape. Note the roster
    # path uses `_backfill_player_ids`, which scores with `fuzz.WRatio` -- the
    # scorer this codebase has banned, whose failure mode is documented
    # directly beneath it -- so it is deliberately not reused here.
    if not base.empty and stats is not None and not stats.empty:
        _before = int(base["Player_ID"].isna().sum())
        if _before:
            base = attach_reference_ids(
                base, stats,
                fill_cols=("Player_ID", "form", "points_per_game"),
                ref_web_col="web_name",
                source_name="waiver available pool",
            )
            _after = int(base["Player_ID"].isna().sum())
            if _before != _after:
                _logger.info(
                    "Available pool: resolved %d of %d element ids the exact "
                    "name joins missed.", _before - _after, _before)

    # Compute Avg FDR next N GWs
    # Twenty clubs, not seven hundred players -- and the same Series is mapped
    # onto the reference pool below, which is the half that was missing.
    _club_fdr = _avg_fdr_by_short_name(current_gw, weeks)
    base["AvgFDRNextN"] = pd.to_numeric(
        base["Team"].astype(str).map(_club_fdr), errors="coerce")

    # Robust Form calculation with fallbacks (uses healthy form — filters out 0-minute GWs)
    _form_n = form_weeks if form_weeks is not None else config.FORM_LOOKBACK_WEEKS
    def _safe_form(pid, fallback_form, fallback_ppg):
        # Healthy form: average of last N GWs where player actually played
        val = None
        if pd.notna(pid):
            try:
                val = compute_healthy_form(
                    int(pid), _form_n,
                    element_history_fn=_element_history_df
                )
            except Exception:
                val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_form, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_ppg, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        return float(val)

    base["Form"] = base.apply(
        lambda r: _safe_form(r.get("Player_ID"), r.get("form"), r.get("points_per_game")),
        axis=1
    )
    base["Form"] = pd.to_numeric(base["Form"], errors="coerce").fillna(0.0)

    return base

def _availability_multiplier(chance, status) -> float:
    """Simple [0,1] multiplier for available players (adds).
    Fit players get 1.0; injured/suspended/unavailable get 0.0."""
    if not pd.isna(status) and str(status).lower() in ('i', 's', 'u'):
        return 0.0
    if not pd.isna(chance):
        try:
            return max(0.0, min(1.0, float(chance) / 100.0))
        except (ValueError, TypeError):
            pass
    # Default: if status is 'a' or unknown with no chance info, assume available
    return 1.0

# Shared with team-strength scoring; see scripts/common/injury_helpers.py
_estimate_games_to_miss = estimate_games_to_miss

def _roster_injury_factor(chance, status, news, season_pts_pctile,
                         deadlines=None) -> float:
    """Context-aware [0,1] factor for roster players (drops).
    Star players with short injuries retain more value (hold logic).

    ``deadlines`` turns a stated return date into a number of *gameweeks*
    rather than into weeks. Without it, `days // 7` overstates an absence across
    an international break -- measured at 26 of 26 players by a mean of 2.0
    gameweeks -- and the duration buckets below turn that straight into a lower
    hold factor: a player back next gameweek drops from 0.70 to 0.40, which is
    the page recommending you drop someone it should be telling you to keep.

    It accepts a **callable** as well as a sequence, and is resolved only past
    the early return above. Reading the calendar is a cached HTTP fetch, and a
    fully fit roster -- the common case, and every offline test -- must not pay
    for one to be told nobody is injured.
    """
    avail = _availability_multiplier(chance, status)

    # Fully available — no penalty
    if avail >= 1.0:
        return 1.0

    if callable(deadlines):
        deadlines = deadlines()

    gws_to_miss = _estimate_games_to_miss(news, chance, status, deadlines=deadlines)

    # Duration factor
    if gws_to_miss == 0:
        duration_factor = 1.0
    elif gws_to_miss <= 2:
        duration_factor = 0.70
    elif gws_to_miss <= 4:
        duration_factor = 0.40
    elif gws_to_miss <= 7:
        duration_factor = 0.15
    else:
        duration_factor = 0.00

    # Quality boost (up to 0.25 for top players)
    try:
        quality_boost = float(season_pts_pctile) * 0.25
    except (ValueError, TypeError):
        quality_boost = 0.0

    hold_factor = duration_factor + quality_boost

    # Effective value = max(immediate, hold), and **never above 1.0**.
    #
    # This is a multiplier on a percentile Keep Score, so a value above 1 does
    # not protect a player -- it invents value he does not have, exactly the
    # failure `check_transfer_risk()` errors on for the transfer multiplier.
    # It became reachable the moment return dates started being counted against
    # the real calendar: a player back before the next deadline now scores
    # `gws_to_miss == 0`, which pairs `duration_factor` 1.0 with a quality boost
    # of up to 0.25. Twelve live players were sitting at 1.25 and 1.125.
    return min(1.0, max(avail, hold_factor))

def _format_availability(chance, status, news) -> str:
    """Human-readable availability string."""
    # Fully fit
    if (pd.isna(status) or str(status).lower() == 'a') and \
       (pd.isna(chance) or float(chance if not pd.isna(chance) else 100) >= 100):
        return "Fit"

    parts = []
    status_map = {
        'a': 'Available', 'd': 'Doubtful', 'i': 'Injured',
        's': 'Suspended', 'u': 'Unavailable', 'n': 'Not available'
    }
    if not pd.isna(status):
        s = str(status).lower()
        parts.append(status_map.get(s, s.upper()))

    if not pd.isna(news) and str(news).strip():
        parts.append(str(news).strip())
    elif not pd.isna(chance):
        try:
            parts.append(f"{int(float(chance))}% chance")
        except (ValueError, TypeError):
            pass

    return " - ".join(parts) if parts else "Fit"

def _add_injury_data(df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge injury columns (chance_of_playing_next_round, status, news) onto player DataFrame via Player_ID."""
    result = df.copy()

    injury_cols_needed = ['chance_of_playing_next_round', 'status', 'news']
    available_injury_cols = [c for c in injury_cols_needed if c in fpl_stats.columns]

    if not available_injury_cols or 'Player_ID' not in result.columns:
        for col in injury_cols_needed:
            if col not in result.columns:
                result[col] = np.nan
        return result

    merge_cols = ['Player_ID'] + available_injury_cols
    injury_data = fpl_stats[merge_cols].drop_duplicates(subset=['Player_ID'])

    result = result.merge(injury_data, on='Player_ID', how='left', suffixes=('', '_inj'))

    # Coalesce duplicates
    for col in injury_cols_needed:
        if f'{col}_inj' in result.columns:
            if col in result.columns:
                result[col] = result[col].fillna(result[f'{col}_inj'])
            else:
                result[col] = result[f'{col}_inj']
            result.drop(columns=[f'{col}_inj'], inplace=True, errors='ignore')
        elif col not in result.columns:
            result[col] = np.nan

    return result

def _attach_minutes_dependency(df: pd.DataFrame, fpl_stats: pd.DataFrame,
                               current_gw: int) -> pd.DataFrame:
    """Whose absence is this player starting because of, and for how long.

    **Reported from the app.** Chelsea's deputy goalkeeper was offered as a
    rest-of-season upgrade worth +0.26 points a gameweek. He starts only because
    the first choice is out with a "Head injury - Unknown return date", and the
    same move is worth **-6.16** over the window if that man is back next week.
    Nothing on the card said so: `add_injury` reports the *add's* own fitness,
    and he is perfectly fit.

    The engine caps a deputy's horizon where the incumbent has a **stated**
    return (`projection_engine._deputy_start_weeks`). For an unstated one it
    deliberately does not -- "unknown" means unknown, and a decay built on the
    status default would invent the fact the text declines to give. That is
    exactly the case where the card has to carry the dependency instead, so this
    reports on **both**, with the provenance of the estimate attached: "est. 4
    GW (status)" is a very different claim from "back 22 Nov".

    Writes `Dep_Blocker` / `Dep_News` / `Dep_Weeks` / `Dep_Source`, all empty
    where the player's minutes depend on nobody.
    """
    out = df.copy()
    for col in ("Dep_Blocker", "Dep_News", "Dep_Source"):
        out[col] = ""
    out["Dep_Weeks"] = np.nan
    if out.empty or fpl_stats is None or fpl_stats.empty:
        return out
    # `news` is read as well as the four below -- a stated return date is the
    # whole gate -- so it is required rather than assumed. The available frame
    # needs only Team and Position: `starts`, `status` and `news` are what the
    # *reference* supplies.
    need_ref = {"Team", "Position", "status", "starts", "news"}
    if need_ref - set(fpl_stats.columns) or {"Team", "Position"} - set(out.columns):
        return out

    # The blocker's name is rendered on a card, so resolve it the way every
    # other surface does. `fpl_stats` carries the legal name, which is what
    # nobody says out loud ("Emiliano Martínez Romero").
    ref = fpl_stats.copy()
    if "Display_Name" not in ref.columns:
        ref = _attach_display_names(ref)
    ref["_starts"] = numeric_col(ref, "starts", 0.0)
    absent = ref[ref["status"].astype(str).str.lower().isin(OUT_OF_SQUAD_STATUSES)]
    if absent.empty:
        return out

    # A loan or a permanent exit is not a return anyone is waiting on, so the
    # deputy's place is not temporary -- `u` players are excluded by news rather
    # than by status, since the status alone cannot tell the two apart.
    absent = absent[~departed_mask(absent["news"]).to_numpy()]
    if absent.empty:
        return out

    deadlines = gameweek_deadlines()
    by_slot: Dict[Any, List[Dict[str, Any]]] = {}
    for _, r in absent.iterrows():
        weeks, source = absence_estimate(r.get("news"),
                                         r.get("chance_of_playing_next_round"),
                                         r.get("status"), deadlines=deadlines)
        by_slot.setdefault((r.get("Team"), r.get("Position")), []).append(
            # `display_of` owns the Display_Name -> Player -> Web_Name order and
            # the NaN handling; re-deriving it here is what the guard in
            # tests/common/test_display_names.py forbids, and rightly.
            {"name": _display_of(r),
             "news": r.get("news") or "", "weeks": weeks, "source": source,
             "starts": float(r["_starts"])})

    # `numeric_col`, not `pd.to_numeric(out.get(...)).fillna(...)`. A missing
    # column makes `DataFrame.get` return **None**, so the `.fillna` raises
    # `AttributeError: 'numpy.float64' object has no attribute 'fillna'` -- the
    # latent crash CLAUDE.md documents under "`DataFrame.get()` is not a safe
    # accessor", and it took this page down the moment the column was absent.
    own_starts = numeric_col(out, "starts", 0.0)
    elapsed = max(1, int(current_gw or 1) - 1)
    for idx in out.index:
        ahead = by_slot.get((out.at[idx, "Team"], out.at[idx, "Position"]))
        if not ahead:
            continue
        mine = float(own_starts.get(idx, 0))
        # Same depth rule as the engine's cap: only a *more established*
        # club-mate at the position makes this player a stand-in, and a player
        # who already starts most weeks is a regular rather than a deputy.
        if mine / elapsed > DEPUTY_MAX_START_RATE:
            continue
        blockers = [b for b in ahead if b["starts"] > mine]
        if not blockers:
            continue
        # The one back soonest is the one who ends the run.
        first = min(blockers, key=lambda b: b["weeks"])
        out.at[idx, "Dep_Blocker"] = str(first["name"])
        out.at[idx, "Dep_News"] = str(first["news"])
        out.at[idx, "Dep_Weeks"] = float(first["weeks"])
        out.at[idx, "Dep_Source"] = str(first["source"])
    return out

def _build_rationale(drop: pd.Series, add: pd.Series) -> str:
    """Generate a concise human-readable explanation for a swap."""
    parts = []

    drop_form = float(drop.get('Form', 0) or 0)
    add_form = float(add.get('Form', 0) or 0)
    if add_form > drop_form and drop_form >= 0:
        parts.append(f"better form ({add_form:.1f} vs {drop_form:.1f})")

    drop_fdr = float(drop.get('AvgFDRNextN', 3) or 3)
    add_fdr = float(add.get('AvgFDRNextN', 3) or 3)
    if add_fdr < drop_fdr:
        parts.append(f"easier fixtures ({add_fdr:.1f} vs {drop_fdr:.1f} FDR)")

    add_proj = float(add.get('Points', 0) or 0)
    if add_proj > 0:
        parts.append(f"projected {add_proj:.1f} pts")

    # Compared only when both sides resolve to the *same* column: `Proj_Next3`
    # is expected points and `MultiGW_Proj` can be a conditional "if he starts"
    # total, so mixing them charges a rotation risk to one player only.
    add_horizon_col, drop_horizon_col = horizon_column(add), horizon_column(drop)
    if add_horizon_col is not None and add_horizon_col == drop_horizon_col:
        add_multi = horizon_points(add)
        drop_multi = horizon_points(drop)
        if add_multi > drop_multi and add_multi > 0 and drop_multi > 0:
            parts.append(f"multi-GW outlook: {add_multi:.1f} vs {drop_multi:.1f} pts")
        elif add_multi > 0 and drop_multi == 0:
            parts.append(f"multi-GW outlook: {add_multi:.1f} pts")

    drop_avail = _format_availability(
        drop.get('chance_of_playing_next_round'),
        drop.get('status'),
        drop.get('news')
    )
    if drop_avail != 'Fit':
        parts.append(f"drop is {drop_avail.lower()}")

    return "; ".join(parts) if parts else "higher overall value"

def _sanity_check_suggestion(drop_row: pd.Series, add_row: pd.Series) -> Tuple[bool, str]:
    """Veto a swap where ADD is clearly worse than DROP on raw observables.

    Thin delegate to :mod:`scripts.common.transfer_sanity`, which is now shared
    with Classic Transfers -- the veto guarded only Draft for as long as it
    lived here, and a Classic transfer is the one that can cost a -4 hit.
    """
    return sanity_check_suggestion(drop_row, add_row)

#: Gameweeks in the engine's `Proj_Next3` window, so it can be read as a rate.
_HORIZON_GWS = 3

def _claim_points_rate(row) -> float:
    """Expected points per gameweek, for ranking claims against each other.

    **The scores this page decides with are positional percentiles, and a
    percentile difference does not mean the same thing at two positions.**
    Measured on the live GW6 pool, a 0.10 gain is worth 0.091 expected points at
    goalkeeper and 0.405 at forward -- a 4.4x spread -- because the positions
    differ in depth and in scoring range. Ranking a GK swap against a FWD swap
    on that difference is the saturate-and-invert failure the Initial Squad
    Optimizer and the Multi-Transfer Planner were both fixed for; this page
    still had it in the one place it decides submission order.

    Percentiles stay in charge of *whether* a swap is worth suggesting -- the
    per-position thresholds and the sanity veto are calibrated on them. Points
    decide only the order.

    The window is three gameweeks rather than one because a Draft claim is a
    lasting roster change: the player is yours until you drop him, and a
    one-week number would rank a good fixture above a better player.
    `Proj_Next3` includes the current gameweek (see "Fantasy Football Pundit
    feed"), so dividing by three gives a rate on the same basis as `Proj`.

    No injury multiplier is applied here. `Proj` is *expected* points -- start
    probability and availability are already inside it -- so discounting again
    is the double-charge this codebase has paid for repeatedly. The injury
    factors stay where they belong, on the percentile side that decides whether
    to suggest the swap at all.
    """
    next3 = pd.to_numeric(row.get("Proj_Next3"), errors="coerce")
    if pd.notna(next3) and next3 > 0:
        return float(next3) / _HORIZON_GWS
    for col in ("_effective_proj", "Proj"):
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val):
            return float(val)
    return 0.0

def _build_suggestion(worst_roster, best_avail, pos, txn_score, depth_map, _ef,
                      outlook=None, points_gain=0.0, lane=LANE_HOLD,
                      release_cost=None, floor=None, thin_wire=False):
    """One suggestion card's payload.

    Extracted from the middle of the search loop so that loop can be read as
    what it is -- collect pairs, then assign them -- rather than as forty lines
    of dictionary construction with the algorithm hidden inside it.
    """
    _drop_proj = _ef(worst_roster.get('_effective_proj', 0))
    _add_proj = _ef(best_avail.get('_effective_proj', 0))
    # Fall back to raw Points column if effective_proj unavailable for ADD
    if _add_proj == 0:
        _add_proj = _ef(best_avail.get('Points', 0))

    # Reachability is a property of the *add*, not of the swap: whether you can
    # win this player at your waiver priority has nothing to do with who you drop.
    outlook = outlook or {}

    return {
        'outlook_band': outlook.get('band', ''),
        'outlook_reason': outlook.get('reason', ''),
        'drop_player': _display_of(worst_roster),
        'drop_team': str(worst_roster.get('Team', '')),
        'drop_position': pos,
        'drop_value': round(float(worst_roster.get('Keep Score', 0)), 3),
        'drop_form': round(float(worst_roster.get('Form', 0) or 0), 1),
        'drop_season_pts': int(float(worst_roster.get('Season_Points', 0) or 0)),
        'drop_proj_pts': round(_drop_proj, 1),
        'drop_3gw_proj': round(horizon_points(worst_roster), 1),
        'drop_has_data': _drop_proj > 0,
        'drop_injury': _format_availability(
            worst_roster.get('chance_of_playing_next_round'),
            worst_roster.get('status'),
            worst_roster.get('news')
        ),
        'add_player': _display_of(best_avail),
        'add_team': str(best_avail.get('Team', '')),
        'add_position': pos,
        'add_value': round(float(best_avail.get('Transfer Score', 0)), 3),
        'add_proj_pts': round(_add_proj, 1),
        'add_3gw_proj': round(horizon_points(best_avail), 1),
        'add_has_data': _add_proj > 0,
        'add_season_pts': int(float(best_avail.get('Season_Points', 0) or 0)),
        'add_form': round(float(best_avail.get('Form', 0) or 0), 1),
        'add_injury': _format_availability(
            best_avail.get('chance_of_playing_next_round'),
            best_avail.get('status'),
            best_avail.get('news')
        ),
        # Which question this card answers. A rental is justified by the drop
        # being *replaceable* rather than by the add having a future -- the add
        # is going back on the wire next week -- so the card carries the release
        # cost as its reason rather than a percentile gap.
        'lane': lane,
        # Whose absence the add's minutes depend on. `add_injury` reports his
        # *own* fitness and he is typically perfectly fit, which is exactly why
        # the Penders card read as safe.
        'add_blocker': str(best_avail.get('Dep_Blocker', '') or ''),
        'add_blocker_news': str(best_avail.get('Dep_News', '') or ''),
        'add_blocker_weeks': (None if pd.isna(best_avail.get('Dep_Weeks', np.nan))
                              else float(best_avail.get('Dep_Weeks'))),
        'add_blocker_source': str(best_avail.get('Dep_Source', '') or ''),
        'thin_wire': thin_wire,
        # Set-piece duty, carried so a card can say it. A first-choice penalty
        # taker has a points source his replacement does not, and the app
        # scored that at zero -- it proposed dropping two of them in one board.
        'drop_penalties': is_first_choice_penalty_taker(worst_roster),
        'add_penalties': is_first_choice_penalty_taker(best_avail),
        'release_cost': (None if release_cost is None or pd.isna(release_cost)
                         else round(float(release_cost), 2)),
        'position_floor': (None if floor is None else round(float(floor), 2)),
        'transaction_score': round(float(txn_score), 3),
        # Expected points per gameweek this swap adds. The ranking currency --
        # `transaction_score` is a percentile gap and is not comparable across
        # positions. See `_claim_points_rate`.
        'points_gain': round(float(points_gain), 2),
        'rationale': _build_rationale(worst_roster, best_avail),
        'urgency': compute_transfer_urgency(pos, depth_map) if depth_map else "",
    }



#: Set-piece duties FPL publishes and the app has never scored.
#:
#: `penalties_order == 1` is a first-choice penalty taker, and a penalty is
#: worth roughly half a goal in expectation -- 2 to 3 points a time for the
#: positions that take them. The bootstrap has carried it all along and only the
#: Player Statistics page reads it.
#:
#: Reported from the app, twice in one breath: it proposed dropping **Enzo Le
#: Fee** (Sunderland's first-choice penalty *and* corner taker) and **Igor
#: Thiago** (Brentford's first-choice penalty taker), in both cases for a
#: replacement who takes neither. It is surfaced here rather than scored: a
#: weight belongs with a measurement, and the accuracy harness has no opinion on
#: this yet. Carrying the column is what makes that measurement possible later.
_SET_PIECE_COLS = ("penalties_order", "direct_freekicks_order",
                   "corners_and_indirect_freekicks_order")


def _attach_set_pieces(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Carry set-piece duties from the reference pool onto a scored frame.

    `pull_fpl_player_stats` has always published these and nothing downstream
    ever received them -- they reached `board.reference` and stopped there, so
    no suggestion, card or score could see them.
    """
    if df is None or df.empty or reference is None or reference.empty:
        return df
    if 'Player_ID' not in df.columns or 'Player_ID' not in reference.columns:
        return df
    cols = [c for c in _SET_PIECE_COLS if c in reference.columns]
    if not cols:
        return df
    have = [c for c in cols if c in df.columns]
    if len(have) == len(cols):
        return df
    src = reference[['Player_ID'] + cols].drop_duplicates(subset=['Player_ID'])
    return df.merge(src, on='Player_ID', how='left',
                    suffixes=('', '_ref'))


def is_first_choice_penalty_taker(row) -> bool:
    """Is he his club's number-one penalty taker?"""
    try:
        order = row.get('penalties_order')
    except AttributeError:
        return False
    try:
        return int(order) == 1
    except (TypeError, ValueError):
        return False


def _lane_num(row, *cols):
    """First numeric value among ``cols`` on a Series row, or None."""
    for col in cols:
        try:
            if col not in row.index:
                continue
            val = float(row.get(col))
        except (TypeError, ValueError):
            continue
        if not pd.isna(val):
            return val
    return None


def _hold_lane_improves_ros(drop_row, add_row) -> bool:
    """Does this swap actually improve the squad for the rest of the season?

    The hold lane's whole premise is a lasting upgrade, and it ranks on
    ``Transfer Score`` -- a blend that is mostly ROS but still carries this
    gameweek. For a player who is temporarily unavailable that blend collapses,
    because the engine correctly writes his gameweek projection to zero.

    Reported from the app: it offered to drop **Igor Thiago** -- an elite
    forward, suspended until 5 Dec, and his club's first-choice penalty taker --
    for Callum Wilson, who projected **1.3 points across the next three
    gameweeks** against Thiago's 30 season points. Every observable the sanity
    veto compares favoured Wilson, because the suspension had zeroed both of
    Thiago's projections; only the rest-of-season view can see that Wilson is
    replacement level (ROS 0.19) and Thiago is not (0.38).

    So the lane now asserts the thing it is named for. A **veto**, never a
    ranking input: it can only remove a suggestion, and no comparable data on
    either side passes.
    """
    drop_ros = _lane_num(drop_row, "ROS")
    add_ros = _lane_num(add_row, "ROS")
    if drop_ros is None or add_ros is None:
        return True
    return add_ros > drop_ros


def _rent_lane_keeps_the_better_asset(drop_row, add_row,
                                      horizon_col="Proj_Next3") -> bool:
    """A rental lasts a week; the drop is permanent.

    ``release_cost`` is meant to price that asymmetry, and it is computed
    against the **waiver floor** -- what you could re-claim at this position
    next round. Where the floor exceeds the roster player's own forward rate the
    cost comes out at zero, i.e. "the wire replaces him for free", and the lane
    will then trade away a better medium-term asset for a one-week bump.

    Reported from the app: drop **James Tarkowski** (56 season points, 7.1 over
    the next three) for Abdukodir Khusanov (18, **6.1**) to gain 0.8 points this
    week. The percentile gap on that pair was already negative. The same shape
    proposed dropping Alex Iwobi for a player worth less over the window.

    The floor is not so much wrong as over-claimed: it assumes you re-claim the
    replacement next round, while the rental has already spent this round's
    claim on somebody else. Rather than re-model that, this refuses the case
    where it bites -- permanently losing the better three-gameweek asset. A
    rental that genuinely upgrades the window is untouched.
    """
    drop_h = _lane_num(drop_row, horizon_col)
    add_h = _lane_num(add_row, horizon_col)
    if drop_h is None or add_h is None:
        return True
    return add_h >= drop_h


def _dedupe_across_lanes(suggestions):
    """One card per (drop, add) pair, keeping the Hold one.

    Both lanes search the same roster against the same pool, so a swap that is
    good this week *and* for the rest of the season legitimately qualifies for
    both -- and then renders twice. Reported from the app for Hume -> Collins.

    Hold wins because it is the stronger claim: "worth keeping" subsumes "worth
    renting", and the rent card would otherwise invite the manager to make a
    lasting upgrade and then hand the player back. Relies on the caller having
    already sorted Hold ahead of Rent, which the sort above guarantees.
    """
    seen, kept = set(), []
    for item in suggestions:
        key = (item.get("drop_player"), item.get("add_player"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def _compute_transfer_suggestions(
    avail_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    top_n: Optional[int] = 3,
    depth_map: Optional[Dict] = None,
    positions: Optional[List[str]] = None,
    roster_candidates: Optional[int] = 2,
    avail_candidates: Optional[int] = 5,
    one_per_position: bool = True,
    waiver_context: Optional[Dict] = None,
    floor_context: Optional[Dict] = None,
    horizon: int = HORIZON_GWS,
    lanes: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Core suggestion logic: position-locked swaps using pre-computed Transfer/Keep Scores.

    Uses Transfer Score (available players) and Keep Score (roster players) from
    compute_player_scores() with dynamic alpha blending. Injury adjustments are
    applied on top of these pre-computed scores.

    Args:
        top_n: cap on the returned list; ``None`` returns every move found.
        positions: restrict the search to these position codes (``G/D/M/F``).
        roster_candidates: how many of the *weakest* roster players to consider
            per position. ``None`` scans the whole roster.
        avail_candidates: how many of the *best* available players to consider
            per position. ``None`` scans the whole pool.
        one_per_position: stop at the first move found per position. The default
            (2 x 5 window, one move per position) is the compact view; passing
            ``None`` for both candidate limits with this off enumerates every
            upgrade on the board — at most one per droppable roster player.
        waiver_context: ``{n_ahead, claim_rates, expected_gone, rival_needs}`` —
            how many managers claim before this one, how many players this
            league's log says go at each position and overall before its turn,
            and (as prose only) which positions those managers are short at.
            Used to stamp each suggestion with how reachable the add actually
            is. Computed **here** rather than at the callsite, for the same
            reason the locked-player filter below is: a caller that forgets it
            produces a page that looks identical and quietly recommends players
            six other managers see first. An empty context — free agency, an
            unknown queue, or a league with no waiver history — leaves every
            band blank, which renders no badge at all.

        floor_context: ``{n_ahead, claim_rates}`` — what the *next* waiver round
            looks like, used to price what you could re-claim. Deliberately a
            separate argument from ``waiver_context`` even though both read the
            same two values: ``waiver_context`` is suppressed during free agency
            because reachability bands describe a queue you are not in, whereas
            the **floor describes next round's queue, which exists either way**.
            Folding them together would switch the rental lane off for half of
            every gameweek.
        horizon: gameweeks the release cost is priced over. See
            ``claim_value.HORIZON_GWS``.
        lanes: which lanes to search, default both.

    Returns:
        (suggestions, debug_rows) — suggestions is the ranked list of transfers;
        debug_rows is a per-position breakdown for the transparency expander.
        Each suggestion carries ``lane``: ``LANE_HOLD`` for a rest-of-season
        upgrade and ``LANE_RENT`` for a one-week play funded by releasing a
        player the wire can replace.
    """
    # Debug pairs are for eyeballing near-misses, not an audit log. An exhaustive
    # scan can pit 15 roster players against 150 available ones; capping keeps the
    # transparency expander readable (and the page from rendering a wall of rows).
    DEBUG_PAIR_CAP = 40

    def _ef(val) -> float:
        """Float conversion that collapses NaN/None to 0."""
        try:
            v = float(val)
            return 0.0 if np.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    suggestions = []
    debug_rows: List[Dict] = []

    # Locked players cannot be acquired — they are on nobody's roster but only
    # become claimable at the next waiver deadline. Filtering here rather than at
    # the callsite means no future caller can forget: the same class of bug as the
    # Fixtures Overview omitting ffp_df from one of two callsites.
    locked_excluded = 0
    if "Draft_State" in avail_df.columns:
        _locked_mask = avail_df["Draft_State"] == ELEMENT_STATE_LOCKED
        locked_excluded = int(_locked_mask.sum())
        if locked_excluded:
            avail_df = avail_df.loc[~_locked_mask].copy()

    # **A player who cannot play is ineligible, not merely worth less.** This sat
    # in the score as `Transfer Score x _availability_multiplier`, which was a
    # *second* charge for availability: `Transfer Score` blends 1GW, a percentile
    # of the engine's `Proj` = `Proj_Start x Start_Pct`, and `Start_Pct` already
    # carries FPL's availability ceiling. A 75%-doubtful player was discounted
    # once inside the score and again on top of it.
    #
    # For a player the multiplier zeroed the effect was not a discount at all but
    # an exclusion -- `_adj_value` 0 makes `txn_score` negative, so he could never
    # clear a threshold -- so saying it here changes no recommendation and states
    # the rule where the locked filter already states its own. Inside the
    # function, for the same reason: no future caller can forget it.
    unavailable_excluded = 0
    if not avail_df.empty:
        _cannot_play = avail_df.apply(
            lambda r: _availability_multiplier(
                r.get('chance_of_playing_next_round'), r.get('status')) <= 0.0,
            axis=1)
        unavailable_excluded = int(_cannot_play.sum())
        if unavailable_excluded:
            avail_df = avail_df.loc[~_cannot_play].copy()

    # What you could re-claim at each position next round. Computed **here**,
    # after the locked and unavailable filters, for two reasons: a locked player
    # is not claimable so he cannot be the floor, and -- as with those filters --
    # computing it inside means no future caller can forget it and silently get
    # a lane whose every release reads as free.
    #
    # `waiver_floors` returns nothing for a position it cannot measure (no claim
    # history, or fewer than MIN_FLOOR_POOL available), and `rental_gain` returns
    # None without a floor, so the rental lane switches itself off per position
    # rather than falling back to zero. Live, that correctly refused a goalkeeper
    # "floor" that rested on the single available keeper.
    _fc = floor_context or {}
    floors = waiver_floors(avail_df, _fc.get("n_ahead"), _fc.get("claim_rates"),
                           horizon=horizon)
    _lanes = tuple(lanes) if lanes else (LANE_HOLD, LANE_RENT)

    # Board rank across the whole available pool, for the claim outlook. Computed
    # after the locked filter, because a locked player is not on the board anyone
    # is claiming from this round.
    ctx = waiver_context or {}
    _overall_rank: Dict[Any, int] = {}
    if ctx and "Transfer Score" in avail_df.columns:
        _ordered = avail_df["Transfer Score"].astype(float).sort_values(ascending=False)
        _overall_rank = {idx: rank for rank, idx in enumerate(_ordered.index, start=1)}

    # Season points for injury hold logic
    roster_season_pts = pd.to_numeric(
        roster_df.get('Season_Points', pd.Series(dtype=float)), errors='coerce'
    ).dropna()

    for pos in (positions if positions is not None else ['G', 'D', 'M', 'F']):
        roster_pos = roster_df[roster_df['Position'] == pos].copy()
        avail_pos = avail_df[avail_df['Position'] == pos].copy()

        # Skip if can't drop (<=1 at position) or nothing to add
        if len(roster_pos) <= 1 or len(avail_pos) == 0:
            debug_rows.append({
                'pos': pos,
                'skipped': True,
                'reason': 'Not enough roster players or no available players',
                'pairs': [],
                'locked_excluded': locked_excluded,
                'unavailable_excluded': unavailable_excluded,
            })
            continue

        # The score as the engine produced it. Availability is already inside it
        # (see the eligibility filter above) and the players it would have zeroed
        # are gone.
        avail_pos['_adj_value'] = pd.to_numeric(
            avail_pos.get('Transfer Score'), errors='coerce').fillna(0.0)

        for idx in roster_pos.index:
            row = roster_pos.loc[idx]
            sp = pd.to_numeric(row.get('Season_Points'), errors='coerce')
            if not pd.isna(sp) and not roster_season_pts.empty:
                pctile = float((roster_season_pts < sp).sum()) / len(roster_season_pts)
            else:
                pctile = 0.5
            factor = _roster_injury_factor(
                row.get('chance_of_playing_next_round'),
                row.get('status'),
                row.get('news'),
                pctile,
                # Passed unresolved: `_roster_injury_factor` reads the calendar
                # only for a player who is actually carrying a doubt.
                deadlines=gameweek_deadlines,
            )
            base_score = float(row.get('Keep Score', 0) or 0)
            roster_pos.loc[idx, '_adj_value'] = base_score * factor

        roster_sorted = roster_pos.sort_values('_adj_value')
        avail_sorted = avail_pos.sort_values('_adj_value', ascending=False)

        # Where this add sits among everyone available at his position — the same
        # ordering the search itself walks, so the outlook cannot disagree with the
        # list it annotates.
        _pos_rank = {idx: rank for rank, idx in enumerate(avail_sorted.index, start=1)}

        # How many players this league's log says go at *this* position before
        # your pick. One figure per position, so it is computed out here rather
        # than per candidate. None where the league has no history to measure,
        # which `claim_outlook` reads as "no opinion" rather than as zero.
        _expected_at_pos = expected_taken_at_position(
            pos, ctx.get('n_ahead'), ctx.get('claim_rates')) if ctx else None

        # Points per gameweek permanently given up by releasing each roster
        # player, net of what the wire would hand back at his position. This is
        # the quantity `Keep Score` cannot express: it is a percentile against
        # ~700 players and has no idea what is *available*. Measured live, three
        # of five roster midfielders sat at or below their position's floor --
        # the wire offered their equal -- while one of them carried a Keep Score
        # of 0.80 and so sat behind the 0.12 "elite" drop threshold.
        _rel = _release_cost(roster_pos, floors, horizon=horizon)
        _drop_release = {idx: _rel.get(idx) for idx in roster_pos.index}

        # **No floor means nothing to fall back on if the move reverses.** It is
        # the same fact `waiver_floors` refused to quote a number for, seen from
        # the manager's side: live there was exactly one available goalkeeper, so
        # a GK claim that goes wrong cannot be undone. The hold lane has no floor
        # gate of its own -- it recommended the deputy keeper anyway -- so the
        # warning has to reach the card.
        _thin_wire = floors.get(pos) is None and len(avail_pos) < MIN_FLOOR_POOL

        pos_debug_pairs = []
        roster_iter = (roster_sorted if roster_candidates is None
                       else roster_sorted.head(roster_candidates))
        avail_iter = (avail_sorted if avail_candidates is None
                      else avail_sorted.head(avail_candidates))

        # Pass 1 -- collect every viable (drop, add) pair.
        #
        # This deliberately does *not* stop at the first add that clears a drop's
        # threshold, the way it used to. An add claimed by a better pairing has
        # to leave its runners-up behind for the drops that lose it, or those
        # drops vanish from the list entirely.
        candidates = []
        for drop_idx, worst_roster in (roster_iter.iterrows() if LANE_HOLD in _lanes
                                       else iter(())):
            keep_val = float(worst_roster.get('Keep Score', 0.5) or 0.5)
            if keep_val > 0.7:
                min_threshold = 0.12   # elite: 12%+ improvement required
            elif keep_val > 0.5:
                min_threshold = 0.07   # above-avg: 7%+ required
            else:
                min_threshold = 0.05   # weak: 5% — raised from 2% to cut noise

            drop_rate = _claim_points_rate(worst_roster)
            for add_idx, best_avail in avail_iter.iterrows():
                txn_score = best_avail['_adj_value'] - worst_roster['_adj_value']
                points_gain = _claim_points_rate(best_avail) - drop_rate

                pair_info = {
                    'drop': _display_of(worst_roster),
                    'drop_keep': round(float(worst_roster.get('Keep Score', 0)), 3),
                    'drop_adj': round(float(worst_roster.get('_adj_value', 0)), 3),
                    'add': _display_of(best_avail),
                    'add_transfer': round(float(best_avail.get('Transfer Score', 0)), 3),
                    'add_adj': round(float(best_avail.get('_adj_value', 0)), 3),
                    'add_chance': best_avail.get('chance_of_playing_next_round'),
                    'gap': round(float(txn_score), 3),
                    'points_gain': round(float(points_gain), 2),
                    'threshold': min_threshold,
                    'passed': txn_score > min_threshold,
                    'sanity': 'n/a',
                    # Clearing the threshold is no longer the same thing as being
                    # recommended: an add can be claimed by a stronger pairing.
                    'assigned': False,
                }
                if len(pos_debug_pairs) < DEBUG_PAIR_CAP:
                    pos_debug_pairs.append(pair_info)

                if txn_score > min_threshold:
                    if not _hold_lane_improves_ros(worst_roster, best_avail):
                        pair_info['passed'] = False
                        pair_info['sanity'] = 'add is worse rest-of-season'
                        continue
                    sanity_ok, sanity_reason = _sanity_check_suggestion(worst_roster, best_avail)
                    pair_info['sanity'] = sanity_reason
                    if not sanity_ok:
                        pair_info['passed'] = False
                        continue
                    candidates.append({
                        'score': float(txn_score),
                        'points': float(points_gain),
                        'drop_idx': drop_idx,
                        'add_idx': add_idx,
                        'drop': worst_roster,
                        'add': best_avail,
                        'debug': pair_info,
                    })
                elif avail_candidates is None:
                    # Available players are sorted by value descending, so once one
                    # misses this drop's threshold no lower-ranked player can clear
                    # it either. Only worth short-circuiting on a full-pool scan.
                    break

        # Pass 2 -- assign, greedily, best gain first.
        #
        # A player can only be added once and dropped once, so the list has to be
        # a *set of moves you can actually make*, not a set of alternatives that
        # all want the same target. It used to offer the single best available
        # player against every drop that cleared a threshold.
        #
        # Greedy on gain gives the intuitively right answer for free: gain is
        # add minus drop, so for a fixed add it is largest against the weakest
        # drop. The best target lands on the player you most want to replace,
        # and the next drop takes the next-best target.
        #
        # This is the assignment problem, and greedy is not provably optimal --
        # but scipy is not a dependency here (Spearman is hand-rolled in
        # projection_accuracy for the same reason) and at 4 positions x a handful
        # of droppable players the difference is not worth the dependency.
        #
        # **This stays on the percentile gap, while the cross-position sort below
        # moves to points.** The two currencies answer different questions here.
        # Assignment is a within-position matching problem, and everything it
        # depends on is calibrated on percentiles: the per-position thresholds,
        # the `break` short-circuit -- which is only sound because `avail_sorted`
        # is ordered by the same `_adj_value` the threshold tests -- and
        # `_pos_rank`, which the reachability badge reads. Re-ordering only this
        # loop on points desynchronises it from all three: a card could then be
        # captioned "#1 available" while sitting third in the list it annotates,
        # which is exactly the disagreement `_pos_rank` was introduced to end.
        #
        # Cross-position ordering has no such entanglement. It is pure
        # presentation and submission order, and that is where the percentile
        # gap is not a comparable quantity at all.
        used_drops, used_adds = set(), set()
        for cand in sorted(candidates, key=lambda c: c['score'], reverse=True):
            if cand['drop_idx'] in used_drops or cand['add_idx'] in used_adds:
                continue
            cand['debug']['assigned'] = True
            outlook = None
            if ctx:
                outlook = claim_outlook(
                    position=pos,
                    pos_rank=_pos_rank.get(cand['add_idx']),
                    n_ahead=ctx.get('n_ahead'),
                    expected_at_position=_expected_at_pos,
                    rivals_needing=len((ctx.get('rival_needs') or {}).get(pos, [])),
                    expected_gone=ctx.get('expected_gone'),
                    overall_rank=_overall_rank.get(cand['add_idx']),
                )
            suggestions.append(_build_suggestion(
                cand['drop'], cand['add'], pos, cand['score'], depth_map, _ef,
                outlook, points_gain=cand['points'], lane=LANE_HOLD,
                release_cost=_drop_release.get(cand['drop_idx']),
                floor=floors.get(pos), thin_wire=_thin_wire,
            ))
            used_drops.add(cand['drop_idx'])
            used_adds.add(cand['add_idx'])
            if one_per_position:
                break
        # ------------------------------------------------------------------
        # The rental lane -- "I only need him this week."
        #
        # **Ranked on `Proj`, not on `Transfer Score`.** That is the whole
        # reason this pass exists separately rather than reusing `avail_iter`:
        # `avail_iter` is the top few by `Transfer Score`, which is ~60% ROS in
        # Draft, and the player who is worth renting is by definition the one
        # with a big this-week projection and a thin future. Live at GW10 the
        # highest-projected available player in the game ranked 4th at his
        # position on `Transfer Score` and was suggested nowhere.
        #
        # The gain is priced by `claim_value.rental_gain`: this week's points
        # difference, less the drop's release cost over the rest of the window.
        # The add's own horizon never enters -- he goes back on the wire -- so
        # what makes a rental cheap is a *replaceable drop*, not a good add.
        if LANE_RENT in _lanes and floors.get(pos) is not None:
            rent_avail = avail_pos.sort_values('Proj', ascending=False)
            if avail_candidates is not None:
                rent_avail = rent_avail.head(avail_candidates)

            # **The same drop window as the hold lane**, and for the same
            # reason: these are the players you would actually consider
            # releasing, weakest first (`roster_sorted` is ordered by
            # `_adj_value`). The rent lane used to iterate the *whole* position,
            # which is why a player could be offered up here who was never a
            # drop candidate in Hold at all.
            #
            # Reported from the app: "Enzo Le Fee is not the person that should
            # be dropped for Rayan Vitor. I have better options that could be
            # dropped instead." Le Fee was the *fourth* weakest of five
            # midfielders and still won the pairing.
            rent_roster = (roster_sorted if roster_candidates is None
                           else roster_sorted.head(roster_candidates))

            rent_cands = []
            for drop_idx, drop_row in rent_roster.iterrows():
                for add_idx, add_row in rent_avail.iterrows():
                    gain = rental_gain(drop_row, add_row, floors, horizon=horizon)
                    if gain is None or gain <= RENTAL_MIN_GAIN:
                        continue
                    if not _rent_lane_keeps_the_better_asset(drop_row, add_row):
                        continue
                    # The sanity veto applies here too: it compares raw
                    # observables and is just as able to catch a nonsense
                    # one-week swap as a season-long one.
                    sane, why = _sanity_check_suggestion(drop_row, add_row)
                    if not sane:
                        continue
                    rent_cands.append({
                        'gain': float(gain), 'drop_idx': drop_idx, 'add_idx': add_idx,
                        'drop': drop_row, 'add': add_row,
                    })

            # **Among drops competing for the same add, release the one you
            # value least** -- not the one that yields the biggest number.
            #
            # `rental_gain` is `(Proj(add) - Proj(drop)) - (H-1) x release_cost`,
            # and `release_cost` floors at zero, so every roster player whose
            # forward rate sits below the wire's floor looks *equally* free to
            # release. Once it saturates, the ranking is driven by
            # `-Proj(drop)`: whoever contributes least **this week** wins the
            # pairing, which is a different question from who you would rather
            # keep. Measured on the live board, the gain for adding Rayan Vitor
            # rose monotonically with Keep Score -- Madueke (0.668, ROS 0.58)
            # yielded +0.58 while Le Fee (0.773, ROS 0.82) yielded +1.46, so the
            # lane proposed releasing the better midfielder.
            #
            # Keep Score first, gain as the tie-break. Every pair here has
            # already cleared `RENTAL_MIN_GAIN`, so each is worth doing on its
            # own terms; the only question left is which player to give up.
            r_drops, r_adds = set(), set()
            for cand in sorted(rent_cands,
                               key=lambda c: (float(c['drop'].get('Keep Score', 0) or 0),
                                              -c['gain'])):
                if cand['drop_idx'] in r_drops or cand['add_idx'] in r_adds:
                    continue
                r_outlook = None
                if ctx:
                    # `_pos_rank` is the **Transfer Score** order, while this lane
                    # is ordered by rental gain -- deliberately, and not the
                    # desync the hold lane's comment warns about. Contention is a
                    # statement about how desirable this player is to *other*
                    # managers, whose ordering is their own; the available board
                    # is the app's best proxy for it and is what the Available
                    # Players table below renders. Nothing here keys a threshold
                    # or a short-circuit on the rank, which is what made the
                    # coupling load-bearing in the hold pass.
                    r_outlook = claim_outlook(
                        position=pos,
                        pos_rank=_pos_rank.get(cand['add_idx']),
                        n_ahead=ctx.get('n_ahead'),
                        expected_at_position=_expected_at_pos,
                        rivals_needing=len((ctx.get('rival_needs') or {}).get(pos, [])),
                        expected_gone=ctx.get('expected_gone'),
                        overall_rank=_overall_rank.get(cand['add_idx']),
                    )
                suggestions.append(_build_suggestion(
                    cand['drop'], cand['add'], pos,
                    # The percentile gap is still carried so the card can show it
                    # and so the cross-lane tie-break has something to fall back
                    # on, but it is not what ranked this pair.
                    float(cand['add'].get('Transfer Score', 0) or 0)
                    - float(cand['drop'].get('Keep Score', 0) or 0),
                    depth_map, _ef, r_outlook,
                    # `points_gain` is the lane's own currency: total points over
                    # the window, not a per-gameweek rate like the hold lane's.
                    # `_render_transfer_suggestions` labels them differently for
                    # that reason.
                    points_gain=cand['gain'], lane=LANE_RENT,
                    release_cost=_drop_release.get(cand['drop_idx']),
                    floor=floors.get(pos), thin_wire=_thin_wire,
                ))
                r_drops.add(cand['drop_idx'])
                r_adds.add(cand['add_idx'])
                if one_per_position:
                    break

        debug_rows.append({
            'pos': pos,
            'skipped': False,
            'reason': '',
            'pairs': pos_debug_pairs,
            'locked_excluded': locked_excluded,
            'unavailable_excluded': unavailable_excluded,
            'floor': floors.get(pos),
        })

    # **Ordered by expected points, not by the percentile gap.** This is the
    # cross-position comparison -- a GK swap against a FWD swap -- and a
    # percentile difference means something different at each position: 0.10 is
    # worth 0.091 points at GK and 0.405 at FWD on the live GW6 pool. The gap
    # still decides which swaps qualify; it just no longer decides their order.
    #
    # The percentile gap is the tie-break, so a gameweek with no projections
    # published degrades to exactly the old ordering rather than to an arbitrary
    # one.
    # Sorted **within** lane, not across it. The two lanes' `points_gain` are
    # different quantities -- the hold lane's is points per gameweek, the rental
    # lane's is total points over the window -- so one sort over both would be
    # the cross-position percentile mistake again, one level up. The renderer
    # draws them as separate sections for the same reason.
    suggestions.sort(key=lambda x: (x['lane'] != LANE_HOLD,
                                    -x['points_gain'], -x['transaction_score']))
    # After the sort, so Hold already leads, and before the per-lane truncation,
    # so a deduped rent card frees its slot for the next one.
    suggestions = _dedupe_across_lanes(suggestions)
    if top_n is not None:
        kept = []
        for lane in (LANE_HOLD, LANE_RENT):
            kept.extend([x for x in suggestions if x['lane'] == lane][:top_n])
        suggestions = kept
    return suggestions, debug_rows

def _build_waiver_context(
    waiver_order: List[Dict],
    my_entry_id,
    current_gw: int,
    transaction_mode: Optional[str] = None,
) -> Dict:
    """What the suggestion engine needs to judge whether a claim is reachable.

    Four things: how many managers claim before you, how many players this
    league's log says go at **each position** before your turn, how many go
    overall, and — as prose only — which positions those managers are short at.

    **Nothing is reachable-or-not during free agency, so nothing is computed.**
    The bands describe a queue, and in free agency an available player is simply
    added: there is no priority to lose and no one to be beaten to him by. An
    empty context disables the outlook end to end — `_build_suggestion()` stamps
    an empty band, `_outlook_pill('')` renders nothing, and the claim plan drops
    its parenthetical — so this one early return is the whole suppression.
    It also means free agency no longer pays for `build_league_strength()`.

    An **unreadable** mode keeps the bands rather than suppressing them: a
    transient `/league/{id}/details` failure must not be indistinguishable from
    free agency, and failing to the behaviour that already exists is the rule
    everywhere else in this codebase.

    Rival need comes from the Power Rankings model rather than a second notion of
    squad strength — `aggregate_team_strength()` already scores every roster by
    position against the full FPL pool, and its `Team_ID` is the `entry_id` space
    the waiver order is keyed on, so the join is exact rather than by name. It no
    longer sets a band; see `claim_outlook()`.

    Returns `{}` whenever free agency is open, the order is unknown, or the league
    has no waiver history to calibrate on. A neutral badge on every card is noise,
    and "we could not tell" is not a finding about a player.
    """
    if transaction_mode == TRANSACTION_MODE_FREE_AGENCY:
        return {}
    if not waiver_order or my_entry_id is None:
        return {}
    pick = my_waiver_pick(waiver_order, my_entry_id)
    if pick is None:
        return {}

    n_ahead = len(managers_ahead(waiver_order, my_entry_id))
    if n_ahead == 0:
        # You claim first: nothing ahead of you can take anyone.
        return {"n_ahead": 0, "rival_needs": {}, "claim_rates": {},
                "expected_gone": 0.0}

    needs = {}
    try:
        from scripts.common.team_strength import build_league_strength
        from scripts.common.waiver_priority import rival_needs as _rival_needs

        team_df, _ = build_league_strength(config.FPL_DRAFT_LEAGUE_ID, current_gw)
        needs = _rival_needs(team_df, waiver_order, my_entry_id)
    except Exception:
        # Power Rankings failing must never take this page down; the reason string
        # simply loses its trailing clause.
        _logger.warning("Unable to derive rival positional needs", exc_info=True)

    stats = _waiver_claim_stats(config.FPL_DRAFT_LEAGUE_ID, len(waiver_order)) or {}
    return {
        "n_ahead": n_ahead,
        "rival_needs": needs,
        "claim_rates": stats.get("position_claim_rates") or {},
        # The recent window, not the season: the post-draft rounds are churn and
        # report a league as more active than it now is.
        "expected_gone": expected_gone_before(
            pick, stats.get("participation_rate_recent")),
    }

def _build_floor_context(waiver_order: List[Dict], my_entry_id, league_id) -> Dict:
    """What the *next* waiver round looks like, for pricing a re-claim.

    Deliberately **not** `_build_waiver_context`, even though it reads the same
    two values. That function returns `{}` during free agency, correctly, because
    the reachability bands describe a queue you are not currently standing in.
    The floor is a different question: it asks what you could claim *next* round,
    and next round is a waiver round whether or not free agency is open now.
    `waiver_pick` is published for the next round throughout, so folding the two
    together would switch the rental lane off for half of every gameweek.

    Returns `{}` when the order or the league's claim history is unreadable, which
    `waiver_floors()` reads as "no floor" and so disables the rental lane rather
    than pricing every release as free.
    """
    if not waiver_order or my_entry_id is None:
        return {}
    pick = my_waiver_pick(waiver_order, my_entry_id)
    if pick is None:
        return {}
    n_ahead = len(managers_ahead(waiver_order, my_entry_id))
    stats = _waiver_claim_stats(league_id, len(waiver_order)) or {}
    return {"n_ahead": n_ahead,
            "claim_rates": stats.get("position_claim_rates") or {}}

def _compute_waiver_score(df: pd.DataFrame,
                          all_players_df: Optional[pd.DataFrame],
                          current_gw: int = 19) -> pd.DataFrame:
    """Compute waiver scores using positional percentile scoring.

    Delegates to shared compute_player_scores() with format_context="draft".
    Returns 1GW, ROS, Transfer Score, and Keep Score columns.
    """
    return compute_player_scores(df, all_players_df, current_gw, format_context="draft")

def _compute_keep_score(roster_df: pd.DataFrame,
                        all_players_df: Optional[pd.DataFrame],
                        current_gw: int = 19,
                        depth_map: Optional[Dict] = None) -> pd.DataFrame:
    """Compute keep scores using positional percentile scoring.

    Delegates to shared compute_player_scores() with format_context="draft"
    and depth_map for squad depth awareness.
    """
    return compute_player_scores(roster_df, all_players_df, current_gw,
                                 format_context="draft", depth_map=depth_map)

@st.cache_data(show_spinner=False)
def _fetch_all_transactions(league_id: int) -> pd.DataFrame:
    """
    Fetch all transactions for a league and return as a DataFrame.

    Transaction kinds:
        'f' = Free transfer (free agent pickup)
        'w' = Waiver claim

    Result values (verified against 160 live rows on 2026-09-24):
        'a'  = Accepted
        'di' = Declined — beaten to the player by a higher waiver priority. All
               58 of these had that same element_in accepted by another manager
               in the same gameweek, which is what pins the meaning.
        'do' = Declined for some other reason. Only 9 of 15 had the player taken,
               so this is *not* a contention signal and must not be counted as one.
    """
    url = f"https://draft.premierleague.com/api/draft/league/{league_id}/transactions"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        _logger.warning("Failed to fetch transactions for league %s: %s", league_id, e)
        return pd.DataFrame()

    transactions = data.get('transactions', [])
    if not transactions:
        return pd.DataFrame()

    return pd.DataFrame(transactions)

@st.cache_data(ttl=600, show_spinner=False)
def _waiver_claim_stats(league_id: int, n_managers: int) -> Dict[str, Any]:
    """What this league's own past waiver rounds looked like.

    Two figures, both measurable from nothing but this league's own log. How many
    managers actually claim in a given round — the number of players gone before
    your turn is not your pick number, it is the number of managers ahead of you
    who bother to submit anything. And **where those claims land**, which is what
    the reachability bands are calibrated on: a position the league rarely shops
    at is not contested however badly a rival is ranked there.

    The element to position map is built here from ``get_fpl_player_mapping()``,
    which is itself cached, rather than taken as an argument — so this function's
    cache key stays ``(league_id, n_managers)`` and no caller has to assemble a
    667-entry dict for Streamlit to hash.

    Returns ``{}`` on any failure — the queue expander then states the rule
    without the calibrated line, and the bands abstain, rather than either
    quoting a fabricated rate.
    """
    from scripts.common.waiver_priority import claim_history_stats, position_claim_rates

    try:
        df = _fetch_all_transactions(league_id)
        if df.empty:
            return {}
        rows = df.to_dict("records")
        stats = claim_history_stats(rows, n_managers=n_managers or None)
        try:
            positions = {
                pid: info.get("Position")
                for pid, info in (get_fpl_player_mapping() or {}).items()
            }
            stats["position_claim_rates"] = position_claim_rates(
                rows, positions, n_managers=n_managers or None)
        except Exception:
            # The rates are the band's only input, so losing them disables the
            # outlook. That is the intended degradation, and far better than
            # falling back to the rival-need model this replaced.
            _logger.warning("Unable to measure per-position claim rates", exc_info=True)
            stats["position_claim_rates"] = {}
        return stats
    except Exception:
        _logger.warning("Unable to summarise waiver claim history", exc_info=True)
        return {}

@st.cache_data(ttl=600, show_spinner=False)
def _next_waiver_deadline(gw: Optional[int]) -> Optional[datetime]:
    """Next transaction deadline in ET, or None if it can't be resolved.

    Thin cached wrapper over waiver_alerts.get_next_transaction_deadline(), which
    hits the fixtures endpoint and is not Streamlit-aware. Purely decorative — any
    failure means the banner drops the line, never that the page breaks.
    """
    try:
        from scripts.common.waiver_alerts import get_next_transaction_deadline
        deadline_et, _kickoff_et, _gw = get_next_transaction_deadline(
            offset_hours=config.TRANSACTION_DEADLINE_HOURS_BEFORE_KICKOFF,
            gw=gw,
        )
        return deadline_et
    except Exception:
        _logger.warning("Could not resolve next transaction deadline", exc_info=True)
        return None


# ===========================================================================
# The board: one assembly of the claim engine's inputs, shared by every caller
# ===========================================================================
@dataclass(frozen=True)
class WaiverBoard:
    """Everything needed to answer "which claim is worth making", assembled once.

    `frozen=True` is shallow -- the frames inside are still mutable pandas
    objects. What protects callers from each other is that `st.cache_data`
    returns a fresh object per call; what protects a caller from *itself* is
    not mutating `available` or `roster` in place, which
    `test_waiver_board.py` pins.

    The search controls are deliberately *not* fields here. They belong to
    `search_board()`, and that separation is the whole reason a widget
    interaction on the Waiver Wire is free: changing the position filter
    re-searches a cached board instead of rebuilding it.
    """

    available: pd.DataFrame
    roster: pd.DataFrame
    reference: pd.DataFrame
    projections: pd.DataFrame
    ownership: Dict[int, Dict[str, Any]]
    element_states: Dict[int, Dict[str, Any]]
    transaction_window: Dict[str, Any]
    waiver_order: List[Dict[str, Any]]
    waiver_context: Dict[str, Any]
    floor_context: Dict[str, Any]
    depth_map: Dict[str, Any]
    locked_count: int
    league_id: Any
    team_id: Any
    entry_id: Any
    current_gw: int
    lookahead: int
    form_weeks: int
    ffp: Any = None
    status: str = BOARD_OK
    notes: Tuple[Tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == BOARD_OK

    @property
    def free_agency(self) -> bool:
        """An unreadable mode is *not* free agency -- see `_build_waiver_context`."""
        return self.transaction_window.get("mode") == TRANSACTION_MODE_FREE_AGENCY

    @property
    def waiver_gw(self):
        return (self.transaction_window.get("next_event")
                or self.transaction_window.get("current_event"))

    @property
    def team_name(self) -> str:
        return (self.ownership.get(self.team_id, {}) or {}).get(
            "team_name", "Team %s" % self.team_id)


@st.cache_data(ttl=120, show_spinner=False)
def build_waiver_board(league_id, team_id, current_gw=None, *, entry_id=None,
                       lookahead=None, form_weeks=None,
                       rotowire_url=None) -> WaiverBoard:
    """Assemble the scored pools, the league's transaction state and the contexts.

    Every argument is a primitive, so the whole board is cacheable -- the
    `my_week._assemble` rule that no frame enters a cache key.

    `entry_id` is separate from `team_id` because the page already
    distinguishes them: the waiver pick and the reachability bands belong to
    *your* entry, while the roster being scored is whichever team the picker
    selects. Inspect a rival's squad and you get your own claim bands against
    his players, which is how the page has always behaved.

    **A failure is cached as a failure, never as a success** -- the
    `get_ffp_feed` rule. A board that cannot be built comes back with a
    non-`ok` status and the reason in `notes`, so the Waiver Wire can render
    it and the My Week strip can turn it into an `unknown` row rather than a
    reassuring one.
    """
    notes: List[Tuple[str, str]] = []
    lookahead = int(lookahead if lookahead is not None
                    else config.UPCOMING_WEEKS_DEFAULT)
    form_weeks = int(form_weeks if form_weeks is not None
                     else config.FORM_LOOKBACK_WEEKS)
    if current_gw is None:
        try:
            current_gw = int(get_current_gameweek() or 1)
        except Exception:
            current_gw = 1
    current_gw = int(current_gw)

    def _empty(status: str) -> WaiverBoard:
        return WaiverBoard(
            available=pd.DataFrame(), roster=pd.DataFrame(),
            reference=pd.DataFrame(), projections=pd.DataFrame(),
            ownership={}, element_states={}, transaction_window={},
            waiver_order=[], waiver_context={}, floor_context={}, depth_map={},
            locked_count=0, league_id=league_id, team_id=team_id,
            entry_id=entry_id, current_gw=current_gw, lookahead=lookahead,
            form_weeks=form_weeks, status=status, notes=tuple(notes))

    # --- League transaction state. Every reader fails open: an empty result
    # means the engine behaves as it did before that signal existed. ---
    try:
        ownership = get_league_player_ownership(league_id)
    except Exception as e:
        _note(notes, "Unable to load league ownership: %s" % e)
        ownership = {}
    try:
        element_states = get_league_element_states(league_id)
    except Exception:
        _logger.warning("Unable to load Draft element states", exc_info=True)
        element_states = {}
    try:
        transaction_window = get_draft_transaction_window(league_id)
    except Exception:
        _logger.warning("Unable to load Draft transaction window", exc_info=True)
        transaction_window = {}
    try:
        waiver_order = get_league_waiver_order(league_id)
    except Exception:
        _logger.warning("Unable to load Draft waiver order", exc_info=True)
        waiver_order = []

    locked_count = sum(1 for s in element_states.values()
                       if s.get("status") == ELEMENT_STATE_LOCKED)

    # --- Projections and the reference pool ---
    if rotowire_url is None:
        rotowire_url = config.ROTOWIRE_URL
    if not rotowire_url:
        _note(notes,
              "\u26a0\ufe0f Rotowire player projections are unavailable \u2014 the app could not "
              "discover the current rankings article URL. Rotowire may have changed "
              "their URL format. To fix immediately, add `ROTOWIRE_URL=<article URL>` "
              "to your `.env` file and restart the app.")
    projections_raw = get_rotowire_player_projections(rotowire_url)
    fpl_stats_raw = pull_fpl_player_stats()

    teams_df = _bootstrap_teams_df()
    try:
        fpl_stats_norm = normalize_fpl_players_to_rotowire_schema(fpl_stats_raw, teams_df=teams_df)
    except Exception as e:
        _logger.warning("FPL stats normalization failed, using raw data: %s", e)
        fpl_stats_norm = fpl_stats_raw
    try:
        projections_norm = normalize_rotowire_players(projections_raw)
    except Exception as e:
        _logger.warning("Projections normalization failed, using raw data: %s", e)
        projections_norm = projections_raw

    fpl_stats = _enforce_rw_schema_fpl(fpl_stats_norm, teams_df)
    proj = _enforce_rw_schema_proj(projections_norm, teams_df)

    req = {"Player", "Team", "Position"}
    miss_stats = req - set(fpl_stats.columns)
    if miss_stats:
        _note(notes, "Failed to compute available players: "
                     "fpl_player_statistics_df missing %s after normalization."
                     % sorted(miss_stats), "error")
        return _empty(BOARD_BAD_SCHEMA)
    miss_proj = req - set(proj.columns)
    if miss_proj:
        _note(notes, "Projections table missing %s after normalization."
                     % sorted(miss_proj), "error")
        return _empty(BOARD_BAD_SCHEMA)

    if "Season_Points" not in fpl_stats.columns:
        fpl_stats["Season_Points"] = 0

    # --- The available pool ---
    try:
        avail_all = _available_from_projections(proj, fpl_stats, ownership,
                                                element_states, notes=notes)
    except WaiverDataUnavailable as unavailable:
        _note(notes, unavailable.message, "info")
        board = _empty(unavailable.status)
        return replace(board, projections=proj, ownership=ownership,
                       element_states=element_states,
                       transaction_window=transaction_window,
                       waiver_order=waiver_order, locked_count=locked_count,
                       notes=tuple(notes))
    except Exception as e:
        _note(notes, "Failed to compute available players: %s" % e, "error")
        board = _empty(BOARD_POOL_FAILED)
        return replace(board, projections=proj, ownership=ownership,
                       element_states=element_states,
                       transaction_window=transaction_window,
                       waiver_order=waiver_order, locked_count=locked_count,
                       notes=tuple(notes))

    avail_all = _add_fdr_and_form(avail_all, fpl_stats, current_gw, lookahead,
                                  form_weeks=form_weeks)
    avail_all = _add_injury_data(avail_all, fpl_stats)

    # Season_Points and starts via Player_ID
    if 'Player_ID' in avail_all.columns and 'Season_Points' not in avail_all.columns:
        sp_cols = ['Player_ID', 'Season_Points']
        if 'starts' in fpl_stats.columns:
            sp_cols.append('starts')
        sp_data = fpl_stats[sp_cols].drop_duplicates(subset=['Player_ID'])
        avail_all = avail_all.merge(sp_data, on='Player_ID', how='left')
    elif 'Player_ID' in avail_all.columns and 'starts' not in avail_all.columns and 'starts' in fpl_stats.columns:
        starts_data = fpl_stats[['Player_ID', 'starts']].drop_duplicates(subset=['Player_ID'])
        avail_all = avail_all.merge(starts_data, on='Player_ID', how='left')
    avail_all['Season_Points'] = numeric_col(avail_all, 'Season_Points', 0)
    avail_all['starts'] = numeric_col(avail_all, 'starts', 0)
    avail_all = _attach_set_pieces(avail_all, fpl_stats)

    # **After the `starts` merge, never before it.** The depth rule reads season
    # starts to tell a deputy from a regular; run earlier, where the column does
    # not exist yet, it silently treats every available player as having zero
    # starts. That shipped once. The assertion is cheap and the failure is not.
    assert 'starts' in avail_all.columns or avail_all.empty, (
        "_attach_minutes_dependency needs `starts`; see the deputy rule")
    avail_all = _attach_minutes_dependency(avail_all, fpl_stats, current_gw)
    avail_all = _attach_display_names(avail_all)


    # --- Build MY ROSTER ---
    my_players = []
    for pos, names in ownership.get(team_id, {}).get("players", {}).items():
        for nm in names:
            my_players.append({"Player": nm, "Position": pos})
    my_roster = pd.DataFrame(my_players)

    if not my_roster.empty:
        # Attach Team/Player_ID/Season_Points/starts from FPL master using normalized names
        my_roster["__norm_name"] = my_roster["Player"].apply(canonical_normalize)
        roster_merge_cols = ["Player", "Team", "Position", "Player_ID", "Season_Points"]
        if "starts" in fpl_stats.columns:
            roster_merge_cols.append("starts")
        fpl_stats_for_roster = fpl_stats[roster_merge_cols].copy()
        fpl_stats_for_roster["__norm_name"] = fpl_stats_for_roster["Player"].apply(canonical_normalize)

        merge_right_cols = ["__norm_name", "Team", "Position", "Player_ID", "Season_Points"]
        if "starts" in fpl_stats_for_roster.columns:
            merge_right_cols.append("starts")
        my_roster = my_roster.merge(
            fpl_stats_for_roster[merge_right_cols],
            on=["__norm_name", "Position"], how="left"
        )
        my_roster.drop(columns=["__norm_name"], inplace=True, errors="ignore")

        # Ensure Team present and align names
        my_roster = _ensure_team_col(my_roster, fpl_stats)
        my_roster = _align_roster_player_names_to_projections(
            my_roster, proj, notes=notes)
        my_roster["Team"] = my_roster["Team"].replace(TEAM_FULL_TO_SHORT)

        # After name alignment the Player column now holds Rotowire canonical names.
        # Recover Team from proj for any player whose Team is still NaN/missing —
        # this happens when the FPL short web-name ("Eze") didn't match the full FPL
        # stats name ("Eberechi Eze") on the earlier normalized-name merge, leaving
        # Team blank even though the name was correctly aligned to Rotowire.
        _nan_team = my_roster["Team"].isna() | my_roster["Team"].isin(["nan", "None", ""])
        if _nan_team.any():
            _proj_team_lookup = (
                proj[["Player", "Team"]]
                .drop_duplicates(subset=["Player"], keep="first")
                .set_index("Player")["Team"]
            )
            my_roster.loc[_nan_team, "Team"] = (
                my_roster.loc[_nan_team, "Player"].map(_proj_team_lookup)
            )

        # Add next-GW projected points
        proj_points_df = proj[["Player", "Team", "Position", "Points"]].rename(
            columns={"Points": "Projected_Points"}
        ).copy()
        proj_points_df["__norm_name"] = proj_points_df["Player"].apply(canonical_normalize)
        my_roster["__norm_name"] = my_roster["Player"].apply(canonical_normalize)

        # Primary merge: name + team + position (exact)
        my_roster = my_roster.merge(
            proj_points_df[["__norm_name", "Team", "Position", "Projected_Points"]],
            on=["__norm_name", "Team", "Position"],
            how="left"
        )
        # Fallback 1: name + position only (catches team-code mismatches).
        _still_missing = my_roster["Projected_Points"].isna()
        if _still_missing.any():
            _proj_name_pos = (
                proj_points_df[["__norm_name", "Position", "Projected_Points"]]
                .drop_duplicates(subset=["__norm_name", "Position"], keep="first")
            )
            _fallback = my_roster.loc[_still_missing, ["__norm_name", "Position"]].merge(
                _proj_name_pos, on=["__norm_name", "Position"], how="left"
            )
            my_roster.loc[_still_missing, "Projected_Points"] = _fallback["Projected_Points"].values

        # Fallback 2: name only (catches FPL vs Rotowire position classification differences,
        # e.g. FPL MID vs Rotowire FWD for attacking players like Eze).
        _still_missing2 = my_roster["Projected_Points"].isna()
        if _still_missing2.any():
            _proj_name_only = (
                proj_points_df[["__norm_name", "Projected_Points"]]
                .drop_duplicates(subset=["__norm_name"], keep="first")
            )
            _fallback2 = my_roster.loc[_still_missing2, ["__norm_name"]].merge(
                _proj_name_only, on="__norm_name", how="left"
            )
            my_roster.loc[_still_missing2, "Projected_Points"] = _fallback2["Projected_Points"].values

        my_roster.drop(columns=["__norm_name"], inplace=True, errors="ignore")

        # `_attach_display_names` resolves by id first, so the ids must already
        # be on the frame -- they come from the normalized-name merge above,
        # which resolves 15 of 15 on the live roster. The `fuzz.WRatio`
        # backfill that used to sit here filled 0 and is deleted; see the
        # tombstone in `player_matching`.
        my_roster = _attach_display_names(my_roster)
        my_roster = _attach_set_pieces(my_roster, fpl_stats)
        # `form_weeks` as well as `lookahead`: the available pool got the widget
        # value and the roster silently kept `config.FORM_LOOKBACK_WEEKS`, so moving
        # the slider compared a 2-week pool form against a 4-week roster form -- in
        # `_build_rationale`, which sets them side by side, and in the ROS form term.
        # They agree at the default, which is why it never showed.
        my_roster = _add_fdr_and_form(my_roster, fpl_stats, current_gw, lookahead,
                                      form_weeks=form_weeks)
        my_roster = _add_injury_data(my_roster, fpl_stats)
    # --- Multi-GW (FFP), season projections, and the reference pool ---
    try:
        ffp_feed_result = get_ffp_feed()
        ffp_df = ffp_feed_result.df
    except Exception:
        ffp_feed_result, ffp_df = None, None

    _remaining_gws = max(1, 38 - current_gw)
    if not avail_all.empty:
        avail_all = blend_multi_gw_projections(avail_all, ffp_df, single_gw_col="Points", remaining_gws=_remaining_gws)
    if not my_roster.empty:
        pts_col = "Projected_Points" if "Projected_Points" in my_roster.columns else "Points"
        my_roster = blend_multi_gw_projections(my_roster, ffp_df, single_gw_col=pts_col, remaining_gws=_remaining_gws)

    try:
        season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
    except Exception:
        season_rankings_df = None

    if not avail_all.empty:
        avail_all = merge_season_projections(avail_all, season_rankings_df)
    if not my_roster.empty:
        my_roster = merge_season_projections(my_roster, season_rankings_df)

    if not avail_all.empty:
        avail_all = merge_ffp_single_gw_data(avail_all, ffp_df)
    if not my_roster.empty:
        my_roster = merge_ffp_single_gw_data(my_roster, ffp_df)

    # fpl_stats is the reference pool for positional_percentile(), so it must
    # carry the same enrichment columns as the frames being scored against it.
    fpl_stats = blend_multi_gw_projections(fpl_stats, ffp_df, single_gw_col="points_per_game", remaining_gws=_remaining_gws)
    fpl_stats = merge_season_projections(fpl_stats, season_rankings_df)
    fpl_stats = merge_ffp_single_gw_data(fpl_stats, ffp_df)
    fpl_stats = enrich_reference_with_projections(fpl_stats, proj)
    # **The fixture term's reference column, which never existed.** Written onto
    # the scored frames and not onto the reference pool, `compute_player_scores`
    # took its min-max fallback: the 15-row roster normalised against itself,
    # the pool against itself, and the suggestion gap subtracts one from the
    # other. Two populations, min-max rather than percentile, no symptom.
    fpl_stats["AvgFDRNextN"] = pd.to_numeric(
        fpl_stats["Team"].astype(str).map(_avg_fdr_by_short_name(current_gw, lookahead)),
        errors="coerce")

    depth_map = compute_positional_depth(my_roster) if not my_roster.empty else {}

    waiver_context = _build_waiver_context(waiver_order, entry_id, current_gw,
                                           transaction_mode=transaction_window.get("mode"))
    floor_context = _build_floor_context(waiver_order, entry_id, league_id)

    # Scoring happens here rather than inside the search, so that changing a
    # position filter re-searches rather than re-percentiles.
    if not my_roster.empty and not avail_all.empty:
        try:
            avail_all = _compute_waiver_score(avail_all, fpl_stats, current_gw=current_gw)
            my_roster = _compute_keep_score(my_roster, fpl_stats, current_gw=current_gw,
                                            depth_map=depth_map)
        except Exception as e:
            _note(notes, "Could not score the pools: %s" % e)

    return WaiverBoard(
        available=avail_all, roster=my_roster, reference=fpl_stats,
        projections=proj, ownership=ownership, element_states=element_states,
        transaction_window=transaction_window, waiver_order=waiver_order,
        waiver_context=waiver_context, floor_context=floor_context,
        depth_map=depth_map, locked_count=locked_count, league_id=league_id,
        team_id=team_id, entry_id=entry_id, current_gw=current_gw,
        lookahead=lookahead, form_weeks=form_weeks, ffp=ffp_feed_result,
        status=BOARD_OK, notes=tuple(notes))


def search_board(board: WaiverBoard, *, positions=None, lanes=None,
                 compact: bool = True, limit=None):
    """Search a built board. Returns `(suggestions, debug_rows, total_found)`.

    Never mutates the board's frames -- the Waiver Wire renders them after
    searching, and a board may be shared with the My Week strip inside one
    rerun.
    """
    if board is None or not board.ok:
        return [], [], 0
    if board.roster.empty or board.available.empty or not positions:
        return [], [], 0
    try:
        suggestions, debug_rows = _compute_transfer_suggestions(
            board.available.copy(), board.roster.copy(),
            top_n=None, depth_map=board.depth_map,
            positions=positions,
            roster_candidates=2 if compact else None,
            avail_candidates=5 if compact else None,
            one_per_position=compact,
            waiver_context=board.waiver_context,
            floor_context=board.floor_context,
            lanes=lanes,
        )
    except Exception as e:
        _logger.warning("Could not compute transfer suggestions: %s", e, exc_info=True)
        return [], [], 0

    total_found = len(suggestions)
    if limit is not None:
        # **Per lane, not across both.** A flat slice lets the hold lane fill
        # the quota and the rental lane vanish -- the exact failure the two
        # lanes exist to fix, re-created by a display limit.
        kept = []
        for lane in (LANE_HOLD, LANE_RENT):
            kept.extend([x for x in suggestions if x["lane"] == lane][:limit])
        suggestions = kept
    return suggestions, debug_rows, total_found


def best_claim(board: WaiverBoard, *, lane=LANE_HOLD, positions=None):
    """The single best claim, or None.

    Routed **through** `search_board` deliberately. The locked-player filter
    and the cannot-play exclusion live *inside*
    `_compute_transfer_suggestions` so that no caller can forget them, and the
    obvious shortcut for a checklist -- take the top row of `board.available`
    by Transfer Score -- bypasses both. That is how a player another manager
    dropped an hour ago comes to be recommended as a claim.
    """
    suggestions, _, _ = search_board(
        board, positions=positions or list(POSITION_LABEL_TO_CODE.values()),
        lanes=(lane,), compact=False)
    return suggestions[0] if suggestions else None


def _code_states(element_states: Optional[Dict[int, Dict[str, Any]]]) -> Dict[int, str]:
    """``{player_code: state}`` for every element the league has a state for.

    `element-status` is keyed on the **Draft** element id and every pool frame in
    the app is keyed on the **Classic** one, and 64 of 669 of those ids disagree
    -- every collision landing on a real but wrong player, so nothing raises and
    every value on the row stays plausible. `get_fpl_player_mapping` carries each
    element's permanent `Player_Code`, so this crosses to the app's join key with
    neither name matching nor an id translation. The same route `_codes_in_state`
    takes, generalised from one state to all of them.
    """
    if not element_states:
        return {}
    try:
        player_map = get_fpl_player_mapping()
    except Exception:
        _logger.warning("Could not read the Draft player map for element states",
                        exc_info=True)
        return {}
    out: Dict[int, str] = {}
    for element_id, state in (element_states or {}).items():
        status = (state or {}).get("status")
        if not status:
            continue
        code = (player_map.get(element_id) or {}).get("Player_Code")
        if code is None or pd.isna(code):
            continue
        out[int(code)] = str(status)
    return out


def return_watch(board: "WaiverBoard", *, limit=None):
    """Claimable players returning from a spell out, best pedigree first.

    ``(frame, notes)``. Deliberately **not** cached and deliberately taking the
    board rather than ids: the expensive inputs are already cached one layer
    down (``build_waiver_board`` at 120s, ``get_prior_season_rates`` at a day),
    so all this adds is a percentile pass over ~670 rows. Re-deriving the board
    from primitives in order to put a cache on it would pay 15s to save 20ms --
    the same reasoning that keeps ``search_board`` uncached.

    The pedigree reference needs the whole pool, not the available one, so this
    reads ``board.reference``: ``board.available`` is built from Rotowire's
    weekly table, and the players this board exists for are *absent from it* --
    which is why they were invisible in the first place.
    """
    notes: List[Tuple[str, str]] = []
    if board is None or not getattr(board, "ok", False):
        return return_watch_module._empty(notes, "No waiver board, so no returning "
                                                 "players can be identified."), tuple(notes)
    reference = getattr(board, "reference", None)
    if reference is None or len(reference) == 0:
        return return_watch_module._empty(notes, "The player pool is empty."), tuple(notes)

    pool = reference
    rates = None
    try:
        ids = dict(zip(
            pd.to_numeric(pool.get("Player_ID"), errors="coerce").dropna().astype(int),
            pd.to_numeric(
                pool.get("Player_Code", pool.get("code")), errors="coerce"
            ).reindex(pool.index).dropna().astype(int),
        ))
        if ids:
            rates = get_prior_season_rates(ids)
    except Exception:
        _logger.warning("Prior-season rates unavailable for the return watch",
                        exc_info=True)

    try:
        deadlines = gameweek_deadlines()
    except Exception:
        deadlines = None

    frame = return_watch_module.build_return_watch(
        pool,
        _code_states(getattr(board, "element_states", None)),
        current_gw=getattr(board, "current_gw", None),
        prior_rates=rates,
        deadlines=deadlines,
        limit=limit,
        notes=notes,
    )
    return frame, tuple(notes)
