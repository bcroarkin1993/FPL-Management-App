"""
Local archive of Fantasy Football Pundit projections, one file per gameweek.

**FFP rolls its window forward and the old gameweek is gone.** Measured
2026-09-05: the site payload covered GW4-GW9, published 03:17 that morning.
GW3 -- which the app was still scoring, because GW3 was in progress -- was no
longer in the payload at all, and FFP was persisted nowhere, so the app's GW3
projections had simply ceased to exist. Nothing was broken and nothing raised;
FFP just silently stopped contributing partway through every gameweek.

That is the shape of the problem this module fixes, and it is not a one-off:
FFP publishes GW N+1 as soon as GW N kicks off, so for most of every gameweek
the live payload is a week ahead of what the app is scoring, and the gameweek
gate (correctly) throws it away.

Two things follow, and both are load-bearing:

1. **Archive the whole window, not the current slice.** Each payload carries six
   gameweeks -- 2274 rows, of which ``to_sheet_schema`` kept 379 and discarded
   1895. Storing all six means the moment FFP publishes GW N it is captured for
   every week through GW N+5, so the app never depends on FFP still offering a
   gameweek by the time it needs it.

2. **Newest publication wins.** A forecast for GW6 taken from a GW4-window
   payload is superseded by one from a GW5-window payload: later publications
   are better informed. A stored entry with no publication timestamp (the
   spreadsheet fallback has none) loses to any entry that has one.

Stored in the **sheet schema**, not the raw site rows, because that schema is
the app's stable compatibility seam -- every consumer already reads ``Name`` /
``Team`` / ``StartingPredicted`` / ``Start`` -- and because a recovery from the
spreadsheet cannot be expressed as site rows at all.

Pure and Streamlit-free: the Actions snapshot collector imports this, and that
workflow installs requirements best-effort.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_logger = logging.getLogger("fpl_app.projections")

#: Provenance values, strongest first.
PROV_SITE = "site"                  # FFP's own payload, for this exact gameweek
PROV_SHEET_OFFSET = "sheet_offset"  # recovered from the spreadsheet's relative
                                    # offset columns -- real FFP data, but of an
                                    # earlier vintage than a direct publication


def _archive_dir() -> Path:
    """Locate (and ensure) archive/ffp/ at the repo root."""
    here = Path(__file__).resolve().parent          # scripts/common/
    d = here.parent.parent / "archive" / "ffp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(gameweek: int) -> Path:
    return _archive_dir() / f"GW{int(gameweek):02d}.json"


def _parse_dt(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def parse_updated(value) -> Optional[datetime]:
    """An archived ``updated`` stamp back as a datetime.

    JSON stores it as a string, but ``FFPFeed.updated`` is consumed by
    ``format_last_updated()``, which needs a real datetime to compute an age.
    """
    return _parse_dt(value)


def _is_newer(new_updated, old_updated) -> bool:
    """Should a payload stamped ``new_updated`` replace one stamped ``old_updated``?

    An entry carrying a publication time always beats one that does not: the
    spreadsheet publishes no revision time, so "no timestamp" means "vintage
    unknown", which must never displace a dated publication.
    """
    new_dt, old_dt = _parse_dt(new_updated), _parse_dt(old_updated)
    if new_dt is None:
        return False
    if old_dt is None:
        return True
    # Compare in UTC; FFP stamps Europe/London, the spreadsheet path none at all.
    if new_dt.tzinfo is None:
        new_dt = new_dt.replace(tzinfo=timezone.utc)
    if old_dt.tzinfo is None:
        old_dt = old_dt.replace(tzinfo=timezone.utc)
    return new_dt > old_dt


def list_archived() -> List[int]:
    """Gameweeks with an archived FFP table, ascending."""
    out = []
    for path in _archive_dir().glob("GW*.json"):
        try:
            out.append(int(path.stem[2:]))
        except ValueError:
            continue
    return sorted(out)


def load_gameweek(gameweek: int) -> Tuple[Optional[pd.DataFrame], dict]:
    """Load one archived gameweek. Returns ``(df, meta)``; df is None if absent.

    Never raises: a damaged archive file must degrade to "no archive", not take
    a page down.
    """
    path = _path(gameweek)
    if not path.exists():
        return None, {}
    try:
        payload = json.loads(path.read_text())
        rows = payload.get("rows") or []
        if not rows:
            return None, {}
        df = pd.DataFrame(rows)
        meta = {k: v for k, v in payload.items() if k != "rows"}
        return df, meta
    except (json.JSONDecodeError, OSError, ValueError) as e:
        _logger.warning("FFP archive: could not read %s: %s", path, e)
        return None, {}


def save_gameweek(df: pd.DataFrame, gameweek: int, updated=None,
                  provenance: str = PROV_SITE, window_gw: Optional[int] = None,
                  force: bool = False) -> bool:
    """Archive one gameweek's table. Returns True if written.

    Declines to overwrite a more recent publication unless ``force``.
    """
    if df is None or df.empty or gameweek is None:
        return False

    _, existing = load_gameweek(gameweek)
    if existing and not force:
        if not _is_newer(updated, existing.get("updated")):
            return False

    payload = {
        "gameweek": int(gameweek),
        # FFP's own publication stamp for the payload this came from. The whole
        # freshness comparison rests on it, so it is stored as written.
        "updated": _parse_dt(updated).isoformat() if _parse_dt(updated) else None,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        # Which gameweek's window this forecast was published in. A GW6 forecast
        # taken from a GW4 window is five weeks of team news less informed than
        # one taken from a GW6 window; without this the difference is invisible.
        "window_gw": int(window_gw) if window_gw is not None else None,
        "rows": json.loads(_compact(df).to_json(orient="records")),
    }
    try:
        # Compact separators, not pretty-printing. Each gameweek's file is
        # rewritten every time a fresher publication covers it -- six times as
        # the window passes over it -- so every byte is six git blobs over a
        # season. These are machine-written and machine-read; nobody diffs them.
        _path(gameweek).write_text(json.dumps(payload, separators=(",", ":")))
    except OSError as e:
        _logger.warning("FFP archive: could not write GW%s: %s", gameweek, e)
        return False
    _logger.info("FFP archive: stored GW%s (%d rows, provenance=%s, window=GW%s)",
                 gameweek, len(df), provenance, window_gw)
    return True


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Round floats before storing.

    FFP publishes one decimal place; pandas' JSON writer emits full float
    repr, which triples the file for digits that carry no information.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["float", "float64"]).columns:
        out[col] = out[col].round(3)
    return out


def archive_payload(rows, window_gw, updated, code_to_id=None) -> List[int]:
    """Archive every gameweek in a site payload. Returns the gameweeks written.

    A payload covers six gameweeks and the app only ever renders one of them, so
    archiving just the current slice would leave the next five to be lost the
    same way GW3 was.
    """
    from scripts.common import ffp_feed

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if "gw" not in frame.columns:
        return []

    written = []
    for gw in sorted(pd.to_numeric(frame["gw"], errors="coerce").dropna().unique()):
        gw = int(gw)
        try:
            df = ffp_feed.to_sheet_schema(rows, gw, code_to_id)
        except Exception as e:                      # pragma: no cover - defensive
            _logger.warning("FFP archive: could not build GW%s schema: %s", gw, e)
            continue
        if save_gameweek(df, gw, updated=updated, provenance=PROV_SITE,
                         window_gw=window_gw):
            written.append(gw)
    return written


def recover_from_sheet_offsets(sheet: pd.DataFrame, sheet_gw: int,
                               target_gw: int) -> Optional[pd.DataFrame]:
    """Recover one gameweek's FFP forecast from the spreadsheet's offset columns.

    The spreadsheet carries ``GW2..GW6``, which are **relative offsets** -- the
    2nd..6th gameweek of its window, on the conditional (if-he-starts) basis.
    So a sheet published for GW2 holds a GW3 forecast in its ``GW2`` column.
    Verified live on 2026-09-05: ``Next2GWsStart == StartingPredicted + GW2`` at
    MAE 0.029, against 0.45 for ``GW2 + GW3``.

    This exists to salvage a gameweek that FFP's live payload has already rolled
    past and that was never archived -- which is how GW3 was lost. It is a
    genuine FFP forecast, but of an **earlier vintage** than a direct
    publication for that gameweek: it predates the team news of the weeks
    between. It is therefore stored under ``PROV_SHEET_OFFSET`` with no
    publication timestamp, so any real payload for that gameweek replaces it.

    Returns None when the offset is outside the sheet's window, which is the
    honest answer rather than an extrapolation.
    """
    if sheet is None or sheet.empty or sheet_gw is None or target_gw is None:
        return None
    offset = int(target_gw) - int(sheet_gw)
    if offset <= 0:
        return None
    # The window's own gameweek is StartingPredicted/Predicted; offsets 1..5 are
    # the columns GW2..GW6. Anything beyond is not in the sheet.
    col = "StartingPredicted" if offset == 0 else f"GW{offset + 1}"
    start_col = "Start" if offset == 0 else f"GW{offset + 1}s"
    if col not in sheet.columns:
        return None

    out = sheet.copy()
    conditional = pd.to_numeric(out[col], errors="coerce")
    # `GWNs` is the same week already discounted by start likelihood. Recovering
    # Start from the pair keeps the two columns on the bases the rest of the app
    # expects, instead of carrying the window week's start rate into a later one.
    if start_col in out.columns and offset > 0:
        discounted = pd.to_numeric(out[start_col], errors="coerce")
        ratio = (discounted / conditional.where(conditional.gt(0)))
        out["Start"] = (ratio.clip(0, 1) * 100).round(0)
    out["StartingPredicted"] = conditional
    out["Predicted"] = conditional * pd.to_numeric(out["Start"], errors="coerce") / 100.0
    out["FFP_GW"] = int(target_gw)

    # Rebuild the multi-gameweek totals from the offsets that now lead the
    # window. `Next3GWs` is 40% of the ROS score, so leaving it behind would
    # quietly drop every archived player onto the `single_gw * 3` fallback.
    # NextN *includes* the current gameweek, per the pinned sheet semantics:
    # Next2GWsStart == StartingPredicted + GW2, not GW2 + GW3.
    for n in (2, 3, 4, 5, 6):
        cond_cols, disc_cols = [], []
        for step in range(n):
            o = offset + step
            cond_cols.append("StartingPredicted" if o == 0 else f"GW{o + 1}")
            disc_cols.append("Start" if o == 0 else f"GW{o + 1}s")
        if all(c in sheet.columns for c in cond_cols):
            out[f"Next{n}GWsStart"] = sum(
                pd.to_numeric(sheet[c], errors="coerce") for c in cond_cols)
        if all(c in sheet.columns for c in disc_cols if not c == "Start"):
            parts = [pd.to_numeric(out["Predicted"], errors="coerce") if c == "Start"
                     else pd.to_numeric(sheet[c], errors="coerce") for c in disc_cols]
            out[f"Next{n}GWs"] = sum(parts)

    # The raw window-relative columns describe the wrong weeks now. Shifting
    # them in place would leave a silently mis-aligned multi-GW column, which is
    # exactly the plausible-but-wrong value nothing downstream can catch, so
    # they are dropped once the totals above have been derived from them.
    drop = [c for c in out.columns
            if (c.startswith("GW") and c != "FFP_GW")]
    out = out.drop(columns=[c for c in drop if c in out.columns])

    # Re-point the fixture at the gameweek actually being described. Carrying
    # the window week's fixture under a later week's label is precisely the
    # failure `resolve_ffp_gameweek()` votes against -- a table whose stated
    # gameweek its own rows contradict. Without this the recovered table claims
    # GW3 while every fixture string is GW2's.
    out = _repoint_fixtures(out, target_gw)
    return out


def _repoint_fixtures(df: pd.DataFrame, gameweek: int) -> pd.DataFrame:
    """Replace ``Fixture`` with the real one for ``gameweek``, per club.

    Written in the same shape the spreadsheet uses -- ``"Aston Villa (a)"``, the
    club's **full name** -- because ``resolve_ffp_gameweek()`` resolves both
    sides through ``TEAM_FULL_TO_SHORT``, and a short code is not a key in it.
    Getting this wrong makes the recovered table unable to prove its own
    gameweek, which is the very check it has to pass.

    Blanks ``Fixture`` if the fixture list cannot be read: a blank is honest,
    last week's opponent is not.
    """
    from scripts.common import ffp_feed

    out = df.copy()
    try:
        boot = ffp_feed._get(ffp_feed.BOOTSTRAP_URL)
        fixtures = ffp_feed._get(ffp_feed.FIXTURES_URL)
        teams = boot.json().get("teams", []) if boot is not None else []
        full_by_id = {t["id"]: t["name"] for t in teams}
        short_by_id = {t["id"]: t["short_name"] for t in teams}
        rows = fixtures.json() if fixtures is not None else []
    except Exception as e:                          # pragma: no cover - defensive
        _logger.warning("FFP archive: could not read the fixture list: %s", e)
        rows = []

    by_club: dict = {}
    for fx in rows:
        if fx.get("event") != int(gameweek):
            continue
        h, a = fx.get("team_h"), fx.get("team_a")
        if h not in full_by_id or a not in full_by_id:
            continue
        # A club can appear twice in a double gameweek; keep both rather than
        # silently picking one.
        by_club.setdefault(short_by_id[h], []).append(f"{full_by_id[a]} (h)")
        by_club.setdefault(short_by_id[a], []).append(f"{full_by_id[h]} (a)")
        by_club.setdefault(full_by_id[h], []).append(f"{full_by_id[a]} (h)")
        by_club.setdefault(full_by_id[a], []).append(f"{full_by_id[h]} (a)")

    if not by_club:
        out["Fixture"] = ""
        return out

    out["Fixture"] = out["Team"].map(
        lambda t: ", ".join(by_club.get(str(t).strip(), []))
    )
    return out


