"""
Per-gameweek record of what we projected, and what actually happened.

Nothing in this app has ever been scored after the fact. The 60/40 Rotowire/FFP
split was assumed, not measured, and there was no way to find out whether it was
right because no projection was ever written down -- ``cache.py`` is a TTL blob
store and every projection the app computed vanished at process exit.

This is the other half of that problem from :mod:`scripts.common.ffp_archive`.
That module keeps FFP's table so the *app* can still use it once FFP's window
rolls forward; this one keeps the **blended output and every source that fed
it**, plus the actual points, so accuracy can be measured.

Two files per gameweek:

``GW03_pre.json``     what every source said before the deadline, and what the
                      engine blended them into. Rewritten while the pre-deadline
                      window is open, frozen the moment the deadline passes --
                      so the committed file is the last state a manager could
                      have acted on.
``GW03_actual.json``  what each player actually scored. Written once, only after
                      FPL marks the gameweek finished *and* ``data_checked``,
                      because bonus and adjustments land after full time and a
                      provisional score is not an actual one.

**A write that changes nothing is skipped.** These files are committed by a
scheduled workflow, so an hourly job that rewrote an identical file would
produce an hourly commit. Rows are compared before writing and ``captured_at``
is preserved when they match, which makes every commit in the history a real
change in the projections.

Pure and Streamlit-free: the Actions collector imports this, and that workflow
installs requirements best-effort.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_logger = logging.getLogger("fpl_app.projections")

KIND_PRE = "pre"
KIND_ACTUAL = "actual"


def _archive_dir() -> Path:
    """Locate (and ensure) archive/projections/ at the repo root."""
    here = Path(__file__).resolve().parent          # scripts/common/
    d = here.parent.parent / "archive" / "projections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(gameweek: int, kind: str) -> Path:
    return _archive_dir() / f"GW{int(gameweek):02d}_{kind}.json"


def _rows_of(df: pd.DataFrame) -> list:
    """Deterministic JSON records: sorted, rounded, stable column order.

    Determinism is the point. An unstable serialisation makes every scheduled
    run look like a change and fills the history with empty commits.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["float", "float64"]).columns:
        out[col] = out[col].round(3)
    if "player_id" in out.columns:
        out = out.sort_values("player_id")
    out = out[sorted(out.columns)]
    return json.loads(out.to_json(orient="records"))


def _save(gameweek: int, kind: str, df: pd.DataFrame, meta: Optional[dict] = None,
          overwrite: bool = True) -> bool:
    if df is None or df.empty or gameweek is None:
        return False

    path = _path(gameweek, kind)
    rows = _rows_of(df)
    captured_at = datetime.now(timezone.utc).isoformat()

    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing is not None:
            if not overwrite:
                return False
            if existing.get("rows") == rows:
                # Nothing changed. Keep the original capture time so the file is
                # byte-identical and the scheduled commit is a no-op.
                return False
            captured_at = existing.get("captured_at", captured_at)

    payload = {"gameweek": int(gameweek), "kind": kind,
               "captured_at": captured_at,
               "updated_at": datetime.now(timezone.utc).isoformat()}
    payload.update(meta or {})
    payload["rows"] = rows
    try:
        path.write_text(json.dumps(payload, separators=(",", ":")))
    except OSError as e:
        _logger.warning("Projection archive: could not write %s: %s", path, e)
        return False
    _logger.info("Projection archive: wrote GW%s %s (%d rows)", gameweek, kind, len(rows))
    return True


def _load(gameweek: int, kind: str) -> Tuple[Optional[pd.DataFrame], dict]:
    path = _path(gameweek, kind)
    if not path.exists():
        return None, {}
    try:
        payload = json.loads(path.read_text())
        rows = payload.get("rows") or []
        if not rows:
            return None, {}
        meta = {k: v for k, v in payload.items() if k != "rows"}
        return pd.DataFrame(rows), meta
    except (json.JSONDecodeError, OSError, ValueError) as e:
        _logger.warning("Projection archive: could not read %s: %s", path, e)
        return None, {}


def save_pre(gameweek: int, df: pd.DataFrame, meta: Optional[dict] = None) -> bool:
    """Store the pre-deadline projections for ``gameweek``."""
    return _save(gameweek, KIND_PRE, df, meta)


def load_pre(gameweek: int) -> Tuple[Optional[pd.DataFrame], dict]:
    return _load(gameweek, KIND_PRE)


def save_actuals(gameweek: int, df: pd.DataFrame, meta: Optional[dict] = None) -> bool:
    """Store what each player actually scored. Written once and never revised.

    Points are final by the time this is called -- the caller waits for FPL's
    ``data_checked`` -- so a second write would either be identical or wrong.
    """
    if _path(gameweek, KIND_ACTUAL).exists():
        return False
    return _save(gameweek, KIND_ACTUAL, df, meta, overwrite=False)


def load_actuals(gameweek: int) -> Tuple[Optional[pd.DataFrame], dict]:
    return _load(gameweek, KIND_ACTUAL)


def _list(kind: str) -> List[int]:
    out = []
    for path in _archive_dir().glob(f"GW*_{kind}.json"):
        try:
            out.append(int(path.stem[2:4]))
        except ValueError:
            continue
    return sorted(out)


def list_pre() -> List[int]:
    """Gameweeks with archived pre-deadline projections."""
    return _list(KIND_PRE)


def list_actuals() -> List[int]:
    """Gameweeks with archived actual points."""
    return _list(KIND_ACTUAL)


def scoreable_gameweeks() -> List[int]:
    """Gameweeks holding both halves of the pair, so accuracy can be measured."""
    return sorted(set(list_pre()) & set(list_actuals()))
