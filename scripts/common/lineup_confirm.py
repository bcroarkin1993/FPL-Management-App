"""Did the manager say they have set this gameweek's lineup?

FPL publishes no pending squad without credentials, so between deadlines the app
is reading *last* deadline's eleven. That makes "your XI is not optimal" a claim
it cannot retract on its own: the manager may have fixed it minutes ago in FPL's
own UI and the app would go on saying otherwise for the rest of the week.

So the alert can be confirmed away, and the confirmation is **tied to the
fifteen it was made about**. A transfer changes the squad's membership, which
invalidates it automatically -- which is the behaviour asked for: *"if I made a
transfer after that, the alert should re-appear, because I'd need to reset my
lineup again based on the new team"*. Nothing here can mark a *different* squad
as checked.

Its own file beside `.fpl_pending_transfers.json`, gitignored: `league_settings.json`
holds non-secret ids in a public repo and this is per-gameweek state, which is a
different purpose and a different lifetime.

Note what this is **not**: evidence about the squad. It records that a human
said they had looked, which is why `week_check` renders a confirmed row as "you
confirmed this XI" rather than as "your XI is optimal".
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.common.lineup_confirm")

CONFIRM_FILE = Path(__file__).resolve().parents[2] / ".fpl_lineup_confirmations.json"


def squad_signature(element_ids: Iterable) -> str:
    """A stable fingerprint of *which fifteen*, ignoring their order.

    Membership rather than arrangement: re-ordering a bench is not a reason to
    re-ask whether the lineup has been looked at, but a transfer is.
    """
    ids = sorted(int(e) for e in element_ids if e is not None)
    return hashlib.sha1(",".join(str(i) for i in ids).encode()).hexdigest()[:16]


def _load() -> dict:
    try:
        if CONFIRM_FILE.exists():
            data = json.loads(CONFIRM_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        _logger.warning("Could not read %s", CONFIRM_FILE.name, exc_info=True)
    return {}


def _key(team_id, gameweek) -> str:
    return "%s:%s" % (int(team_id), int(gameweek))


def confirm_lineup(team_id, gameweek, signature: str) -> bool:
    """Record that this manager has set the lineup for this squad and gameweek."""
    data = _load()
    data[_key(team_id, gameweek)] = {"signature": signature}
    try:
        CONFIRM_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(CONFIRM_FILE, 0o600)
        except OSError:
            pass
        return True
    except OSError:
        _logger.warning("Could not write %s", CONFIRM_FILE.name, exc_info=True)
        return False


def is_confirmed(team_id, gameweek, signature: Optional[str]) -> bool:
    """Has *this* squad been confirmed for this gameweek?

    A confirmation recorded against a different signature does not count, so a
    transfer re-arms the alert without anything having to clear the file.
    """
    if not signature:
        return False
    entry = _load().get(_key(team_id, gameweek)) or {}
    return entry.get("signature") == signature


def clear_confirmation(team_id, gameweek) -> None:
    """Forget a confirmation -- used when the squad it described has changed."""
    data = _load()
    if data.pop(_key(team_id, gameweek), None) is None:
        return
    try:
        CONFIRM_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        _logger.warning("Could not write %s", CONFIRM_FILE.name, exc_info=True)
