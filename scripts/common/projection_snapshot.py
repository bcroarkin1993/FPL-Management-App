"""
Scheduled collector: capture projections before each deadline, actuals after.

Run as ``python -m scripts.common.projection_snapshot``, the same shape as
``scripts.common.waiver_alerts``. Streamlit-free throughout, because the
workflow installs requirements best-effort.

**Why this is a scheduled job and not something the app does.** History that
only accrues when someone opens Streamlit is history with holes in it, and a
missed gameweek cannot be recovered -- FFP rolls its window forward and the old
week is gone from their site permanently. That is not hypothetical: it is
exactly how GW3's FFP projections were lost.

Two independent pieces of work per run:

*Pre-deadline*  Within ``PRE_DEADLINE_WINDOW_HOURS`` of the next deadline, build
                the full projection frame and store it. Rewritten each run while
                the window is open and frozen once the deadline passes, so the
                committed file is the **last state a manager could have acted
                on** -- which is the thing worth scoring.

*Actuals*       For any finished gameweek with no actuals file, store what each
                player really scored. FPL is only trusted once it reports both
                ``finished`` and ``data_checked``: bonus and adjustments land
                well after full time, and a provisional score is not an actual.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import requests

from scripts.common import projection_archive
from scripts.common.projection_engine import build_projections
from scripts.common.projection_sources import (
    fetch_ffp_table,
    ffp_source,
    fpl_ep_source,
    rotowire_source,
)

_logger = logging.getLogger("fpl_app.projections")

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
LIVE_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"

#: How long before a deadline to start capturing. Wide enough that a run is
#: never missed, narrow enough that the stored frame reflects settled team news.
PRE_DEADLINE_WINDOW_HOURS = 8.0

_POS = {1: "G", 2: "D", 3: "M", 4: "F"}


def _get_json(url, timeout=30):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def build_pool(bootstrap) -> pd.DataFrame:
    """The canonical player universe, with both name forms.

    ``Player`` is the full legal name that projection sources are matched
    against; ``Display_Name`` is what a human calls him. Match on one, display
    the other -- swapping them silently degrades match rates.
    """
    from scripts.common.text_helpers import to_display_name

    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    rows = []
    for e in bootstrap.get("elements", []):
        first, second = e.get("first_name", ""), e.get("second_name", "")
        rows.append({
            "Player_ID": e.get("id"),
            "Player": f"{first} {second}".strip(),
            "Display_Name": to_display_name(first, second, e.get("web_name", "")),
            "Web_Name": e.get("web_name", ""),
            "Team": teams.get(e.get("team"), ""),
            "Position": _POS.get(e.get("element_type"), "M"),
            "ep_next": pd.to_numeric(e.get("ep_next"), errors="coerce"),
            "status": e.get("status"),
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
        })
    return pd.DataFrame(rows)


def _rotowire_url(gameweek) -> Optional[str]:
    """Resolve the Rotowire article **for this gameweek**, explicitly.

    Deliberately not ``config.ROTOWIRE_URL``, which discovers the article for
    ``config.CURRENT_GAMEWEEK``. That happens to equal the gameweek being
    captured, because the resolver rolls forward once the current gameweek
    finishes -- but it is a coincidence of timing, not a guarantee, and if it
    ever failed the collector would pair one gameweek's Rotowire article with
    another's FFP table and store the result as a projection. Asking for the
    gameweek we actually want costs nothing and cannot drift.
    """
    try:
        import config
        return config._discover_rotowire_article(int(gameweek))
    except Exception as e:
        _logger.warning("Could not resolve the Rotowire article for GW%s: %s",
                        gameweek, e)
        return None


def collect_pre(gameweek: int, bootstrap, deadline=None,
                before_deadline: bool = True) -> bool:
    """Build and store the projection frame for ``gameweek``.

    ``before_deadline`` records whether this capture actually happened before
    the deadline. A backfill run after the fact is not the same thing: it can
    see team news, and in the limit lineups, that no manager had when they
    picked. Scoring one as though it were a genuine pre-deadline projection
    would flatter every source in it, so the distinction is stored rather than
    inferred later.
    """
    pool = build_pool(bootstrap)
    if pool.empty:
        _logger.error("Bootstrap carried no players; nothing to snapshot")
        return False

    rw = rotowire_source(_rotowire_url(gameweek))
    ffp_df, ffp_gw, ffp_updated, ffp_prov, ffp_note = fetch_ffp_table(gameweek=gameweek)
    ffp = ffp_source(ffp_df, ffp_gw, ffp_updated, ffp_note)
    ep = fpl_ep_source(bootstrap=bootstrap, gameweek=gameweek)

    sources = [rw, ffp, ep]
    projections = build_projections(sources, gameweek=gameweek, pool=pool)
    if projections.empty:
        _logger.error("GW%s: engine produced no rows", gameweek)
        return False

    # Store the blend *and* every source that fed it. Scoring the blend alone
    # would answer "is the app right" but never "which source is right", and the
    # second question is the one that sets the weights.
    frame = projections.copy()
    frame.index.name = "player_id"
    frame = frame.reset_index()
    meta_cols = pool.set_index("Player_ID")[["Player", "Display_Name", "Team", "Position"]]
    frame = frame.join(meta_cols, on="player_id")
    frame = frame.rename(columns={
        "Player": "player", "Display_Name": "display_name",
        "Team": "team", "Position": "position",
        "Proj": "proj", "Proj_Start": "proj_start", "Start_Pct": "start_pct",
        "Proj_Next3": "proj_next3", "Proj_Src": "proj_src",
        "Proj_Spread": "proj_spread",
    })
    frame = frame.drop(columns=["Proj_GW"], errors="ignore")
    # Lowercase the per-source columns too. These files are a long-lived data
    # contract -- 38 of them a season -- so a mixed-case schema is worth fixing
    # before any history accrues rather than after.
    frame = frame.rename(columns={
        c: c.lower() for c in frame.columns
        if c.startswith("Proj_Start__") or c.startswith("Start_Pct__")
    })

    source_status = [{
        "name": s.name, "ok": bool(s.ok), "rows": int(len(s.df)) if s.ok else 0,
        "basis": s.basis, "gameweek": s.gameweek,
        "updated": s.updated.isoformat() if getattr(s, "updated", None) else None,
        "note": s.note,
    } for s in sources]

    meta = {
        "deadline": deadline,
        "captured_before_deadline": bool(before_deadline),
        "sources": source_status,
        "ffp_provenance": ffp_prov,
        # Odds describe whatever fixtures are *upcoming*, so they only line up
        # with this gameweek inside the pre-deadline window. On a backfill they
        # would be a later gameweek's matches filed under this one.
        "odds": collect_match_odds() if before_deadline else None,
    }
    written = projection_archive.save_pre(gameweek, frame, meta)
    _logger.info("GW%s pre-deadline snapshot: %s (%d players priced)",
                 gameweek, "written" if written else "unchanged",
                 int(frame["proj"].notna().sum()))
    return written


def collect_match_odds() -> Optional[list]:
    """Match odds for the upcoming fixtures, stored with the snapshot.

    One call per run, and only inside the pre-deadline window, so the free tier's
    500 requests a month is never a constraint. Stored rather than re-fetched so
    an odds-derived projection can be modelled later without spending the budget
    again on data that has already gone stale.

    Returns None when no key is configured, which is not an error.
    """
    from scripts.common.projection_sources import fetch_match_odds

    try:
        df = fetch_match_odds()
    except Exception as e:
        _logger.warning("Match odds unavailable: %s", e)
        return None
    if df is None or df.empty:
        return None
    out = df.copy()
    for col in out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        out[col] = out[col].astype(str)
    import json
    return json.loads(out.to_json(orient="records"))


def collect_actuals(gameweek: int) -> bool:
    """Store what each player actually scored in ``gameweek``."""
    try:
        live = _get_json(LIVE_URL.format(gw=gameweek))
    except Exception as e:
        _logger.warning("GW%s live data unavailable: %s", gameweek, e)
        return False

    rows = []
    for el in live.get("elements", []):
        stats = el.get("stats") or {}
        rows.append({
            "player_id": el.get("id"),
            "points": stats.get("total_points"),
            "minutes": stats.get("minutes"),
            "started": stats.get("starts"),
            "bonus": stats.get("bonus"),
            "goals": stats.get("goals_scored"),
            "assists": stats.get("assists"),
            "clean_sheet": stats.get("clean_sheets"),
        })
    if not rows:
        _logger.warning("GW%s live data carried no players", gameweek)
        return False

    written = projection_archive.save_actuals(gameweek, pd.DataFrame(rows))
    _logger.info("GW%s actuals: %s (%d players)", gameweek,
                 "written" if written else "already stored", len(rows))
    return written


def _events(bootstrap) -> list:
    return bootstrap.get("events") or []


def pending_actuals(bootstrap) -> List[int]:
    """Finished gameweeks with no actuals file yet.

    ``data_checked`` is the gate, not ``finished``: bonus points and any
    adjustments land well after full time, and storing a provisional score as an
    actual would bake a wrong number into the record permanently.
    """
    have = set(projection_archive.list_actuals())
    return [
        int(e["id"]) for e in _events(bootstrap)
        if e.get("finished") and e.get("data_checked") and int(e["id"]) not in have
    ]


def deadline_for(bootstrap, gameweek) -> Optional[str]:
    """The deadline of one specific gameweek, or None."""
    for e in _events(bootstrap):
        if int(e.get("id", 0)) == int(gameweek):
            return e.get("deadline_time")
    return None


def next_deadline(bootstrap):
    """``(gameweek, deadline)`` for the next gameweek, or ``(None, None)``."""
    for e in _events(bootstrap):
        if e.get("is_next"):
            return int(e["id"]), e.get("deadline_time")
    # Preseason: nothing is "next" yet, so take the first unfinished gameweek.
    for e in _events(bootstrap):
        if not e.get("finished"):
            return int(e["id"]), e.get("deadline_time")
    return None, None


def within_pre_deadline_window(deadline, now=None,
                               hours: float = PRE_DEADLINE_WINDOW_HOURS) -> bool:
    """Is ``now`` inside the capture window that ends at ``deadline``?

    False once the deadline passes, which is what freezes the stored file at the
    last state a manager could have acted on.
    """
    if not deadline:
        return False
    try:
        dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
    except ValueError:
        return False
    now = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt - timedelta(hours=hours) <= now < dt


def run(force_gameweek: Optional[int] = None, skip_actuals: bool = False) -> int:
    """One collection pass. Returns the number of files written."""
    try:
        bootstrap = _get_json(BOOTSTRAP_URL)
    except Exception as e:
        _logger.error("FPL bootstrap unreachable: %s", e)
        return 0

    written = 0

    gw, deadline = next_deadline(bootstrap)
    if force_gameweek is not None:
        gw = force_gameweek
        # The deadline of the gameweek being captured, not of whatever happens
        # to be next -- storing GW4's deadline on a GW3 snapshot is the kind of
        # metadata error nothing downstream can detect.
        deadline = deadline_for(bootstrap, gw)
        before = within_pre_deadline_window(deadline)
        _logger.info("Forced snapshot for GW%s (deadline %s, %s)", gw, deadline,
                     "pre-deadline" if before else "backfill after the deadline")
        written += bool(collect_pre(gw, bootstrap, deadline, before_deadline=before))
    elif gw and within_pre_deadline_window(deadline):
        _logger.info("GW%s deadline %s is within %.0fh — capturing",
                     gw, deadline, PRE_DEADLINE_WINDOW_HOURS)
        written += bool(collect_pre(gw, bootstrap, deadline))
    else:
        _logger.info("Outside the pre-deadline window (next: GW%s at %s)", gw, deadline)

    if not skip_actuals:
        for done_gw in pending_actuals(bootstrap):
            written += bool(collect_actuals(done_gw))

    pre, act = projection_archive.list_pre(), projection_archive.list_actuals()
    _logger.info("Archive holds pre=%s actuals=%s scoreable=%s",
                 pre, act, projection_archive.scoreable_gameweeks())
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameweek", type=int, default=None,
                        help="Force a pre-deadline snapshot for this gameweek, "
                             "ignoring the deadline window.")
    parser.add_argument("--skip-actuals", action="store_true",
                        help="Do not backfill actuals for finished gameweeks.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    written = run(force_gameweek=args.gameweek, skip_actuals=args.skip_actuals)
    print(f"{written} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
