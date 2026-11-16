"""
Previous-season player history — the app's first pedigree *rate*.

Every quality signal the app had was about the current season or a preseason
forecast, which between them cannot describe a player who has not played. A
manager returning from a long injury scores zero on all of them: zero minutes,
zero starts, no form, no projection, and absence from Rotowire's weekly table.
``grep history_past`` over this repo returned nothing before this module.

``element-summary/{id}`` has published ``history_past`` all along -- one row per
prior Premier League season with that season's points, starts and minutes. From
it, ``total_points / starts`` is a **rate in the currency the app optimises**,
measured rather than forecast, which is exactly what a returning player's own
season cannot supply. Measured live: William Saliba, 0 points this season, read
137 points from 30 starts last season.

Two properties are load-bearing.

**Keyed on ``Player_Code``, never an element id.** Element ids are *reassigned
at season rollover*, so a frame of prior-season rows keyed on one cannot be
joined to anything in a later season -- the same reason ``projection_snapshot``
writes ``player_code`` beside ``player_id``. ``code`` is FPL's permanent
identifier and the only id the two APIs agree on. ``fetch_history_past`` takes
the id-to-code mapping as its argument rather than deriving it, so there is no
way for a caller to get a frame back that is keyed on the wrong thing.

**The ids are Classic-space.** ``element-summary`` is the Classic API, so a
Draft element id passed here resolves to a *real but wrong* player for 64 of 669
players. See ``scripts/common/element_ids.py``.

Pure: no Streamlit, no app imports. The cached wrapper lives in ``scraping.py``.
"""

import concurrent.futures
import logging

import pandas as pd
import requests

_logger = logging.getLogger("fpl_app.player_history")

ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/%s/"

#: One row per (player, prior season).
HISTORY_COLUMNS = [
    "Player_Code", "Player_ID", "Season_Name",
    "total_points", "starts", "minutes",
]

#: What ``prior_season_rate`` emits, one row per player.
RATE_COLUMNS = [
    "Player_Code", "Prior_PPS", "Prior_Role",
    "Prior_Season", "Prior_Starts", "Prior_Points",
]

#: Gameweeks in a season -- the denominator for a prior season's start share.
SEASON_GWS = 38

DEFAULT_TIMEOUT = 30
DEFAULT_WORKERS = 8


def _empty(columns):
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})


def _fetch_one(session, player_id, player_code, timeout):
    """``(player_code, player_id, rows)`` for one element, or empty rows."""
    try:
        payload = session.get(ELEMENT_SUMMARY_URL % player_id, timeout=timeout).json()
    except Exception:
        _logger.debug("element-summary failed for id %r", player_id, exc_info=True)
        return player_code, player_id, []
    rows = payload.get("history_past") or []
    return player_code, player_id, rows if isinstance(rows, list) else []


def fetch_history_past(id_to_code, *, workers=DEFAULT_WORKERS,
                       timeout=DEFAULT_TIMEOUT, session=None):
    """Prior-season rows for the given players, as one long frame.

    Args:
        id_to_code: mapping of **Classic** element id -> permanent player code.
            A mapping rather than a bare list of ids so the returned frame
            cannot be keyed on the wrong identifier; see the module docstring.
        workers: pooled request concurrency. The work is pure network latency,
            so it parallelises -- measured 672 requests in 3.3s. Follows
            ``pl_content.fetch_pl_injuries``.

    Returns:
        A frame of ``HISTORY_COLUMNS``, empty (with the columns declared) if
        nothing could be read. **A failure is never an exception**: this feeds
        one rung of a three-rung ladder, and the other two must keep working.
    """
    pairs = []
    for player_id, code in dict(id_to_code or {}).items():
        if player_id is None or code is None:
            continue
        try:
            if pd.isna(player_id) or pd.isna(code):
                continue
        except (TypeError, ValueError):
            pass
        pairs.append((int(player_id), int(code)))
    if not pairs:
        return _empty(HISTORY_COLUMNS)

    own_session = session is None
    session = session or requests.Session()
    records = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_fetch_one, session, pid, code, timeout)
                       for pid, code in pairs]
            for future in concurrent.futures.as_completed(futures):
                code, pid, rows = future.result()
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    records.append({
                        "Player_Code": code,
                        "Player_ID": pid,
                        "Season_Name": row.get("season_name"),
                        "total_points": row.get("total_points"),
                        "starts": row.get("starts"),
                        "minutes": row.get("minutes"),
                    })
    except Exception:
        _logger.warning("Prior-season history fetch failed outright", exc_info=True)
        return _empty(HISTORY_COLUMNS)
    finally:
        if own_session:
            try:
                session.close()
            except Exception:
                pass

    if not records:
        _logger.info("No prior-season history for any of %d players", len(pairs))
        return _empty(HISTORY_COLUMNS)

    out = pd.DataFrame.from_records(records)
    for col in ("total_points", "starts", "minutes"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[HISTORY_COLUMNS]


def prior_season_rate(history, *, min_starts=10, season_gws=SEASON_GWS):
    """One pedigree row per player: his rate, and how regularly he started.

    ``Prior_PPS`` is points per start and ``Prior_Role`` is the share of the
    season he started. They are kept **separate on purpose**. Multiplying them
    was tried and is wrong: a prior season's share is over all 38 gameweeks
    while a current season's is over however many have elapsed, so the product
    mixes two denominators and systematically buries the prior rung. The rate is
    the pedigree; the share is a gate on whether he walks back into the side.

    The chosen season is the **most recent** one with at least ``min_starts``
    starts, not the best. A four-year-old peak is not pedigree -- Saliba's 2022/23
    through 2025/26 all qualify, and 2025/26 is the one that describes him now.

    A low ``min_starts`` is safe here precisely *because* the share is reported
    rather than folded in: a part-season shows up as a low ``Prior_Role`` and is
    excluded by the caller's gate, instead of inflating a rate that outranks
    everybody (Tomiyasu's 7.00 off 10 starts).
    """
    if history is None or len(history) == 0:
        return _empty(RATE_COLUMNS)
    need = {"Player_Code", "Season_Name", "total_points", "starts"}
    if not need.issubset(set(history.columns)):
        _logger.warning("Prior-season history is missing %s",
                        sorted(need - set(history.columns)))
        return _empty(RATE_COLUMNS)

    df = history.copy()
    df["starts"] = pd.to_numeric(df["starts"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df = df[df["starts"].notna() & df["total_points"].notna()]
    df = df[df["starts"] >= float(min_starts)]
    if df.empty:
        return _empty(RATE_COLUMNS)

    # Sort by season label rather than trusting the payload's order. "2025/26"
    # sorts after "2024/25" lexicographically, so this is stable without
    # parsing, and `keep="last"` then takes the most recent qualifying season.
    df = df.sort_values(["Player_Code", "Season_Name"], kind="mergesort")
    df = df.drop_duplicates(subset=["Player_Code"], keep="last")

    out = pd.DataFrame({
        "Player_Code": df["Player_Code"].astype("int64"),
        "Prior_PPS": df["total_points"] / df["starts"],
        "Prior_Role": (df["starts"] / float(season_gws)).clip(upper=1.0),
        "Prior_Season": df["Season_Name"],
        "Prior_Starts": df["starts"],
        "Prior_Points": df["total_points"],
    }).reset_index(drop=True)
    return out[RATE_COLUMNS]
