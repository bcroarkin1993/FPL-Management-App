# scripts/common/fpl_auth.py
#
# Authenticated access to the FPL Classic "my team" endpoint.
#
# WHY THIS EXISTS
# ---------------
# The public endpoints publish the squad as it stood at the *last* deadline.
#
# CONFIRMED, re-checked on 2026-09-04 either side of the GW3 deadline:
#   entry/{id}/event/{gw}/picks/  -> 404 for any gameweek whose deadline has
#                                    not passed. Pinned by tests/live/.
#   my-team/{id}/                 -> 403 without credentials.
#
# NOT CONFIRMED, and the reason this module hedges: whether
# entry/{id}/transfers/ and history.chips publish a move *before* the deadline
# or only after it. A manager who wildcarded at 13:39 UTC saw the old squad in
# the app at ~17:20, 10 minutes before the 17:30 deadline, and both were
# populated when re-checked an hour after it — but no pre-deadline reading of
# that team was taken, so publication delay is inferred, not observed. (An
# earlier reading that appeared to show it was taken against the wrong entry id
# and proves nothing.) The next pre-deadline transfer settles it.
#
# Either way this module is the belt-and-braces answer: my-team is the one
# endpoint that is *defined* to return the live squad, so it is correct whether
# or not the others lag.
#
# The credential is a session cookie the user pastes from their browser. FPL's
# login flow is Cloudflare-protected, so scripted username/password login is not
# a viable path.

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

_logger = logging.getLogger(__name__)

__all__ = [
    "AUTH_FILE",
    "load_auth",
    "save_auth",
    "clear_auth",
    "has_auth",
    "build_headers",
    "fetch_my_team",
    "test_credentials",
    "normalise_my_team",
]

# Repo root, alongside league_settings.json / .fpl_pending_transfers.json.
# A credential gets its own file rather than joining league_settings.json:
# that file holds non-secret IDs, and this repo is public.
AUTH_FILE = Path(__file__).resolve().parents[2] / ".fpl_auth.json"

MY_TEAM_URL = "https://fantasy.premierleague.com/api/my-team/{team_id}/"
_ALLOWED_HOST = "fantasy.premierleague.com"

# FPL 403s a bare python-requests User-Agent on some paths.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Auth outcome codes. These are a closed set — callers branch on them, and in
# particular `expired` must never be collapsed into `error` or into a silent
# None: a stale cookie failing quietly reproduces the exact bug this module
# exists to fix (a stale squad rendered as if it were current).
STATUS_OK = "ok"
STATUS_NO_AUTH = "no_auth"
STATUS_EXPIRED = "expired"
STATUS_ERROR = "error"


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------

def load_auth() -> Dict[str, Any]:
    """Read the stored credential. Returns {} when absent or unreadable."""
    try:
        if AUTH_FILE.exists():
            data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        _logger.warning("Could not read %s", AUTH_FILE.name, exc_info=True)
    return {}


def save_auth(cookie: str, bearer: Optional[str] = None) -> None:
    """Persist the credential to the gitignored auth file, owner-readable only."""
    payload = {
        "version": 1,
        "cookie": (cookie or "").strip(),
        "bearer": (bearer or "").strip() or None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    AUTH_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        # Non-POSIX filesystem — the file is still gitignored.
        _logger.debug("Could not chmod %s", AUTH_FILE.name, exc_info=True)


def clear_auth() -> None:
    """Delete the stored credential."""
    try:
        AUTH_FILE.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        _logger.warning("Could not delete %s", AUTH_FILE.name, exc_info=True)


def has_auth() -> bool:
    return bool(load_auth().get("cookie"))


def build_headers(auth: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Build request headers from a stored credential.

    Both auth schemes are supported because FPL has used both and the live one
    cannot be confirmed without a real credential: historically the `sessionid`
    cookie sufficed, while newer web clients also send an
    `X-API-Authorization: Bearer` token. Sending the cookie alone is the common
    case; the bearer is added only when the user supplied one. The League Setup
    page's "Test connection" button is what settles which is required.
    """
    auth = auth if auth is not None else load_auth()
    cookie = (auth.get("cookie") or "").strip()
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }
    if cookie:
        headers["Cookie"] = cookie
    bearer = (auth.get("bearer") or "").strip()
    if bearer:
        # Accept a token pasted with or without the "Bearer " prefix.
        if not bearer.lower().startswith("bearer "):
            bearer = "Bearer " + bearer
        headers["X-API-Authorization"] = bearer
    return headers


# ---------------------------------------------------------------------------
# Fetch + normalise
# ---------------------------------------------------------------------------

def _fetch_my_team_uncached(team_id: int,
                           auth: Optional[Dict[str, Any]] = None,
                           ) -> Tuple[Optional[dict], str]:
    auth = auth if auth is not None else load_auth()
    if not auth.get("cookie"):
        return None, STATUS_NO_AUTH
    url = MY_TEAM_URL.format(team_id=team_id)
    # The credential grants full access to the user's FPL account. Pin the host
    # so it can never be attached to the Rotowire / FFP / odds fetchers.
    if urlparse(url).hostname != _ALLOWED_HOST:
        _logger.error("Refusing to send FPL credentials to %s", urlparse(url).hostname)
        return None, STATUS_ERROR
    try:
        resp = requests.get(url, headers=build_headers(auth), timeout=30)
    except Exception:
        _logger.warning("my-team request failed for team %s", team_id, exc_info=True)
        return None, STATUS_ERROR

    if resp.status_code in (401, 403):
        _logger.info("my-team auth rejected for team %s (HTTP %s)", team_id, resp.status_code)
        return None, STATUS_EXPIRED
    if not resp.ok:
        _logger.warning("my-team returned HTTP %s for team %s", resp.status_code, team_id)
        return None, STATUS_ERROR
    try:
        return resp.json(), STATUS_OK
    except ValueError:
        # An HTML login page with a 200 is what a half-valid cookie looks like.
        _logger.warning("my-team returned non-JSON for team %s", team_id)
        return None, STATUS_EXPIRED


@st.cache_data(show_spinner=False, ttl=60)
def fetch_my_team(team_id: int) -> Tuple[Optional[dict], str]:
    """Fetch the live squad for the authenticated user's own team.

    Returns (normalised_payload, status) where status is one of
    ok / no_auth / expired / error.

    Deliberately NOT routed through `cached_api_call` (the SQLite layer): this
    payload changes the instant the user makes a transfer, and that layer's
    permanent-TTL branch would pin a squad forever.
    """
    if not team_id:
        return None, STATUS_NO_AUTH
    raw, status = _fetch_my_team_uncached(int(team_id))
    if status != STATUS_OK or not raw:
        return None, status
    return normalise_my_team(raw), STATUS_OK


def test_credentials(team_id: int,
                     creds: Optional[Dict[str, Any]] = None,
                     ) -> Tuple[Optional[dict], str]:
    """Validate a credential against the live endpoint, uncached.

    Takes the candidate credential as an argument rather than reading the file,
    so the League Setup page can verify a pasted value *before* saving it. Note
    the cached fetch_my_team() deliberately takes no credential argument:
    Streamlit hashes cache_data arguments into the key and surfaces them in
    cache-introspection and hash-error messages, which is no place for a
    session cookie.
    """
    if not team_id:
        return None, STATUS_NO_AUTH
    if creds is not None and not (creds.get("cookie") or "").strip():
        return None, STATUS_NO_AUTH
    raw, status = _fetch_my_team_uncached(int(team_id), auth=creds)
    if status != STATUS_OK or not raw:
        return None, status
    return normalise_my_team(raw), STATUS_OK


def normalise_my_team(raw: dict) -> dict:
    """Reshape a my-team payload into the shape get_classic_team_picks() returns.

    The two payloads differ in three ways that all fail silently if ignored:
      - bank/value live under a top-level `transfers` object, not `entry_history`
      - the active chip is a `status_for_entry == "active"` entry in `chips`
      - picks carry NO `multiplier` key, and `_build_squad_dataframe()` reads
        `pick.get("multiplier", 1)` — so an un-normalised payload gives the
        captain a 1x multiplier and every bench player 1x instead of 0.
    """
    picks_in = raw.get("picks") or []

    active_chip = None
    for chip in raw.get("chips") or []:
        if chip.get("status_for_entry") == "active":
            active_chip = chip.get("name")
            break

    captain_multiplier = 3 if active_chip == "3xc" else 2
    # Bench Boost pays the bench, so bench players are not zeroed under it.
    bench_multiplier = 1 if active_chip == "bboost" else 0

    picks_out = []
    for pick in picks_in:
        position = pick.get("position", 0)
        if position > 11:
            multiplier = bench_multiplier
        elif pick.get("is_captain"):
            multiplier = captain_multiplier
        else:
            multiplier = 1
        picks_out.append({
            "element": pick.get("element"),
            "position": position,
            "is_captain": bool(pick.get("is_captain", False)),
            "is_vice_captain": bool(pick.get("is_vice_captain", False)),
            "multiplier": multiplier,
            "selling_price": pick.get("selling_price"),
            "purchase_price": pick.get("purchase_price"),
        })

    transfers = raw.get("transfers") or {}
    return {
        "picks": picks_out,
        "active_chip": active_chip,
        "entry_history": {
            "value": transfers.get("value", 0),
            "bank": transfers.get("bank", 0),
            "event_transfers": transfers.get("made", 0),
            "event_transfers_cost": transfers.get("cost", 0),
        },
        "chips": raw.get("chips") or [],
        "_source": "my_team",
    }
